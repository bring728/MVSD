import numpy as np
import torch
import glob
import os.path as osp
from utils import loadHdr, loadImage, loadBinary, get_hdr_scale
import torch.nn.functional as F

ALPHA_EPS = 1e-10


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, down, pos):
    z_norm = normalize(z)
    r = normalize(np.cross(down, z_norm))
    d = normalize(np.cross(z_norm, r))
    m = np.stack([r, d, z_norm, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def _34_to_44(proj):
    if proj.ndim == 3:
        bottom = np.array([0, 0, 0, 1], dtype=float).reshape([1, 1, 4])
        bottom = np.repeat(bottom, proj.shape[0], axis=0)
        return np.concatenate([proj, bottom], 1)
    else:
        bottom = np.array([0, 0, 0, 1], dtype=float).reshape([1, 4])
        return np.concatenate([proj, bottom], 0)


def recenter(mat_list):
    src_mat = mat_list[:4, ...]
    s = np.array([1.0, 1.0, 1.0], dtype=float)
    c2w_mean = np.mean(src_mat, axis=0)
    u, _, vh = np.linalg.svd(c2w_mean[:3, :3])
    
    ref2w = np.concatenate([u @ np.diag(s) @ vh, c2w_mean[:3, 3:]], 1)
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    ref2w = np.concatenate([ref2w[:3, :4], bottom], -2)[None]
    return np.linalg.inv(ref2w) @ mat_list

def make_reference_view(mat_list):
    s = np.array([1.0, 1.0, 1.0], dtype=float)
    c2w_mean = np.mean(mat_list, axis=0)
    u, _, vh = np.linalg.svd(c2w_mean[:3, :3])
    return np.concatenate([u @ np.diag(s) @ vh, c2w_mean[:3, 3:]], 1)


def poses_avg(poses):
    hwf = poses[:3, -1:, 0]
    center = poses[:3, 3, :].mean(-1)
    z = normalize(poses[:3, 2, :].sum(-1))
    down_avg = poses[:3, 1, :].sum(-1)
    c2w = np.concatenate([viewmatrix(z, down_avg, center), hwf], 1)
    return c2w


def InterpolateDepths(near_depth, far_depth, num_depths):
    """Returns num_depths from (far_depth, near_depth), interpolated in inv depth.
    Args:
      near_depth: The first depth.
      far_depth: The last depth.
      num_depths: The total number of depths to create, include near_depth and
        far_depth are always included and other depths are interpolated between
        them, in inverse depth space.

    Returns:
      The depths sorted in ascending order (so nearest first). This order is
      useful for back to front compositing.
    """

    inv_near_depth = 1.0 / near_depth
    inv_far_depth = 1.0 / far_depth
    depths = []
    for i in range(0, num_depths):
        fraction = float(i) / float(num_depths - 1)
        inv_depth = inv_far_depth + (inv_near_depth - inv_far_depth) * fraction
        depths.append(1.0 / inv_depth)

    # for i in range(0, num_depths):
    #     fraction = float(far_depth - near_depth) / float(num_depths - 1)
    #     depth = far_depth - i * fraction
    #     depths.append(depth)
    depths.reverse()
    return depths


def CreateDepthPlaneHomo(ref_c2w, tan_vh, shape, depths, src_int, src_c2w):
    ref_c2w = _34_to_44(ref_c2w)
    tan_v = tan_vh[0]
    tan_h = tan_vh[1]
    c_f_p = np.identity(4)
    intrinsics = np.identity(3)
    hom = np.identity(3)

    plane2image_H = []
    for depth in depths:
        x_size = tan_h * depth
        y_size = tan_v * depth
        c_f_p[0, 3] = -x_size
        c_f_p[1, 3] = -y_size
        c_f_p[2, 3] = depth
        intrinsics[0, 0] = shape[1] / (x_size * 2.0)
        intrinsics[1, 1] = shape[0] / (y_size * 2.0)

        dpc_inv_int = np.linalg.inv(intrinsics)
        dpc_p2w = ref_c2w @ c_f_p

        src_w2c = np.linalg.inv(src_c2w)
        p2c_src = src_w2c @ dpc_p2w
        hom[:, 0] = p2c_src[0:3, 0]
        hom[:, 1] = p2c_src[0:3, 1]
        hom[:, 2] = p2c_src[0:3, 3]

        src_int @ hom @ dpc_inv_int
    return np.stack(plane2image_H)


def CreateDepthPlaneCameras_2(tan_vh, shape, depths):
    """Creates depth plane Cameras for each of depths.

    Note that a depth plane is paramaterized by the extrinsic 3D transform and a
    2D mapping from the plane's coordinate system to pixels in the planes texture
    map. We slightly abuse the notion of a camera and use a Camera object as a
    container for these two transformations for depth planes.

    Creates depth plane cameras for the passed depth "centered" on the camera with
    transform w_f_c.
    A separate camera will be created for each depth in depths and each
    depth camera will have spatial size and intrinsics such that its
    "field of view" from the w_f_c origin will be fov_degrees.

    Args:
      w_f_c: The world from camera transform that these planes are created at.
      fov_degrees: Tuple of [vertical, horizontal] field of view for depth planes.
      shape: The shape of the depth planes (height, width, num_channels).
      depths: The depths along which to compute the planes.

    Returns:
      Returns a list of depth planes.
    """
    tan_v = tan_vh[0]
    tan_h = tan_vh[1]
    p2w_list = []
    inv_intrinsics_list = []
    for depth in depths:
        p2w = np.identity(4)
        x_size = tan_h * depth
        y_size = tan_v * depth
        p2w[0, 3] = -x_size
        p2w[1, 3] = -y_size
        p2w[2, 3] = depth
        p2w_list.append(p2w)
        
        inv_intrinsics = np.identity(3)
        inv_intrinsics[0, 0] = (x_size * 2.0) / shape[1]
        inv_intrinsics[1, 1] = (y_size * 2.0) / shape[0]
        inv_intrinsics_list.append(inv_intrinsics)
    return np.stack(inv_intrinsics_list), np.stack(p2w_list)

def getH_torch(dpc_inv_ints, dpc_p2w, src_int, src_w2c):
    dpc_inv_ints = torch.repeat_interleave(dpc_inv_ints[:, None, ...], src_int.shape[1], axis=1)
    dpc_p2w = torch.repeat_interleave(dpc_p2w[:, None, ...], src_int.shape[1], axis=1)

    src_int = torch.repeat_interleave(src_int[:, :, None, ...], dpc_p2w.shape[2], axis=2)
    src_w2c = torch.repeat_interleave(src_w2c[:, :, None, ...], dpc_p2w.shape[2], axis=2)

    p2c_src = src_w2c @ dpc_p2w
    hom = torch.zeros_like(dpc_inv_ints)
    hom[..., 0] = p2c_src[..., 0:3, 0]
    hom[..., 1] = p2c_src[..., 0:3, 1]
    hom[..., 2] = p2c_src[..., 0:3, 3]
    return src_int @ hom @ dpc_inv_ints


# def get_viewdir(depth, target_int, target_c2w, hom_image_coords_t):
#     n,c,h,w = depth.size()
#     depth = depth.permute(0, 2, 3, 1).contiguous()
#     normalized_coords = torch.einsum('ijk,nlk->nijl', hom_image_coords_t, torch.linalg.inv(target_int))
#     target_camera_coords = normalized_coords * depth
#     hom_target_camera_coords = torch.cat((target_camera_coords, torch.ones([n, h, w, 1], dtype=target_int.dtype, device=target_int.device)), -1)
#     world_coords = torch.einsum('nijk,nlk->nijl', hom_target_camera_coords, target_c2w[:,:3,:4]).view(n, -1, 3)
#     target_cam_pos = (target_c2w[:, :3, -1])[:, None, :]
#     viewdir_xyz = torch.nn.functional.normalize(world_coords - target_cam_pos, dim=-1)
#     return viewdir_xyz[..., :2]

def get_viewdir(target_int, target_c2w, hom_image_coords_t):
    n = target_int.size()[0]
    normalized_coords = torch.einsum('ijk,nlk->nijl', hom_image_coords_t, torch.linalg.inv(target_int))
    world_coords = torch.einsum('nijk,nlk->nijl', normalized_coords, target_c2w[:,:3,:3]).view(n, -1, 3)
    viewdir_xyz = torch.nn.functional.normalize(world_coords, dim=-1)
    return viewdir_xyz[..., :2]

def span_viewdir(nstep):
    step = float(1.4 / float(nstep))
    vx = torch.arange(-0.7, 0.7, step, dtype=torch.float, device=torch.device('cuda'))
    vy = torch.arange(-0.7, 0.7, step, dtype=torch.float, device=torch.device('cuda'))
    vxy = torch.stack(torch.meshgrid(vx, vy, indexing='ij'), -1).view(-1, 2)
    return vxy


def CreateDepthPlaneCameras(ref_c2w, tan_vh, shape, depths):
    """Creates depth plane Cameras for each of depths.

    Note that a depth plane is paramaterized by the extrinsic 3D transform and a
    2D mapping from the plane's coordinate system to pixels in the planes texture
    map. We slightly abuse the notion of a camera and use a Camera object as a
    container for these two transformations for depth planes.

    Creates depth plane cameras for the passed depth "centered" on the camera with
    transform w_f_c.
    A separate camera will be created for each depth in depths and each
    depth camera will have spatial size and intrinsics such that its
    "field of view" from the w_f_c origin will be fov_degrees.

    Args:
      w_f_c: The world from camera transform that these planes are created at.
      fov_degrees: Tuple of [vertical, horizontal] field of view for depth planes.
      shape: The shape of the depth planes (height, width, num_channels).
      depths: The depths along which to compute the planes.

    Returns:
      Returns a list of depth planes.
    """
    ref_c2w = _34_to_44(ref_c2w)
    tan_v = tan_vh[0]
    tan_h = tan_vh[1]
    c_f_p = np.identity(4)
    intrinsics_list = []
    p2w_list = []
    for depth in depths:
        x_size = tan_h * depth
        y_size = tan_v * depth
        c_f_p[0, 3] = -x_size
        c_f_p[1, 3] = -y_size
        c_f_p[2, 3] = depth
        intrinsics = np.identity(3)
        intrinsics[0, 0] = shape[1] / (x_size * 2.0)
        intrinsics[1, 1] = shape[0] / (y_size * 2.0)

        intrinsics_list.append(intrinsics)
        p2w_list.append(ref_c2w @ c_f_p)
    inv_intrinsics_list = np.linalg.inv(np.stack(intrinsics_list))
    return inv_intrinsics_list, np.stack(p2w_list)

def getH_parallel(dpc_inv_ints, dpc_p2ws, src_int, src_c2w):
    dpc_inv_ints = np.repeat(dpc_inv_ints[None, ...], src_int.shape[0], axis=0)
    dpc_p2ws = np.repeat(dpc_p2ws[None, ...], src_int.shape[0], axis=0)

    src_int = np.repeat(src_int[:, None, ...], dpc_p2ws.shape[1], axis=1)
    src_c2w = np.repeat(src_c2w[:, None, ...], dpc_p2ws.shape[1], axis=1)

    src_w2c = np.linalg.inv(src_c2w)
    p2c_src = src_w2c @ dpc_p2ws
    hom = np.zeros_like(dpc_inv_ints)
    hom[..., 0] = p2c_src[:, :, 0:3, 0]
    hom[..., 1] = p2c_src[:, :, 0:3, 1]
    hom[..., 2] = p2c_src[:, :, 0:3, 3]
    return src_int @ hom @ dpc_inv_ints


def ImageFromPlane(src_int, src_c2w, dpc_p2w):
    """Computes the homography from the plane's space to the camera's image.

    Points on the plane in the plane's space have coordinates (x, y, 0). The
    homography computed maps from x, y to image pixel coordinates in the camera.
    Note that this maps from the plane's geometric coordinate system (i.e. *not*
    the image associated with the plane) to the image's pixels.

    Args:
      camera: A camera instance.
      w_f_p: The transform from the plane to the world, see top of file.

    Returns:
      Returns a numpy 3x3 matrix representing the homography.
    """
    src_c2w = _34_to_44(src_c2w)
    src_w2c = np.linalg.inv(src_c2w)
    p2c_src = src_w2c * dpc_p2w
    hom = np.matrix(np.identity(3))
    hom[:, 0] = p2c_src[0:3, 0]
    hom[:, 1] = p2c_src[0:3, 1]
    hom[:, 2] = p2c_src[0:3, 3]
    return src_int * hom


def ComputeImageFromPlaneHomographies(dpc_inv_ints, dpc_p2ws, src_int, src_c2w):
    """Compute the homographies from the depth planes to the image.

    The returned homography will map a pixel on a depth plane to a (floating
    point) pixel in the image camera.

    Args:
      depth_cameras: A list of "depth" cameras instances.
      image_camera: Homographies are computed from this camera's coordinate system
        to each of the depth cameras.

    Returns:
      The list of length len(depth_cameras), containing the 3x3 float32 np.array
      representing the homographies.
    """
    plane2image_H = []
    for dpc_inv_int, dpc_p2w in zip(dpc_inv_ints, dpc_p2ws):
        plane2image_H.append(np.asarray(ImageFromPlane(
            src_int, src_c2w, dpc_p2w) * dpc_inv_int).astype(np.float32))
    return np.stack(plane2image_H)


def WarpCoordinatesWithHomography(homography, rect):
    """Computes the warped coordinates from rect through homography.

    Computes the corresponding coordinates on the image for each pixel of rect.
    Note that the returned coordinates are in x, y order.
    The returned image can be used to warp from the image to the
    pixels of the depth_plane within rect.
    warp_coordinates = ApplyHomographyToCoords(....)
    warped_from_image(x, y) = image(warp_coordinates(x, y)[0],
                                    warp_coordinates(x, y)[1])

    Args:
      homography: A 3x3 tensor representing the transform applied to the
        coordinates inside rect.
     rect: An integer tensor [start_y, start_x, end_y, end_x] representing a rect.

    Returns:
      Returns a rect.height * rect.width * 2 tensor filled with image
      coordinates.
    """
    ys = torch.arange(rect[0], rect[2], dtype=torch.float32,
                      device=torch.device('cuda'))
    xs = torch.arange(rect[1], rect[3], dtype=torch.float32,
                      device=torch.device('cuda'))

    # Adds 0.5, as pixel centers are assumed to be at half integer coordinates.
    image_coords_t = torch.transpose(torch.stack(
        torch.meshgrid(xs, ys), -1) + 0.5, 0, 1)
    hom_image_coords_t = torch.cat((image_coords_t, torch.ones(
        [rect[2] - rect[0], rect[3] - rect[1], 1], device=torch.device('cuda'))), -1)

    hom_warped_coords = torch.einsum(
        'ijk,lk->ijl', hom_image_coords_t, homography)
    return hom_warped_coords[:, :, :-1] / hom_warped_coords[:, :, 2:3]


def WarpCoordinatesWithHomography_np(homography, rect):
    """Computes the warped coordinates from rect through homography.

    Computes the corresponding coordinates on the image for each pixel of rect.
    Note that the returned coordinates are in x, y order.
    The returned image can be used to warp from the image to the
    pixels of the depth_plane within rect.
    warp_coordinates = ApplyHomographyToCoords(....)
    warped_from_image(x, y) = image(warp_coordinates(x, y)[0],
                                    warp_coordinates(x, y)[1])

    Args:
      homography: A 3x3 tensor representing the transform applied to the
        coordinates inside rect.
     rect: An integer tensor [start_y, start_x, end_y, end_x] representing a rect.

    Returns:
      Returns a rect.height * rect.width * 2 tensor filled with image
      coordinates.
    """
    ys = np.arange(rect[0], rect[2]).astype(np.float32)
    xs = np.arange(rect[1], rect[3]).astype(np.float32)

    # Adds 0.5, as pixel centers are assumed to be at half integer coordinates.
    image_coords_t = np.transpose(
        np.stack(np.meshgrid(xs, ys, indexing='ij'), -1) + 0.5, (1, 0, 2))
    hom_image_coords_t = np.concatenate(
        (image_coords_t, np.ones([rect[2] - rect[0], rect[3] - rect[1], 1])), -1)

    hom_warped_coords = np.einsum(
        'ijk,lk->ijl', hom_image_coords_t, homography)
    return hom_warped_coords[:, :, :-1] / hom_warped_coords[:, :, 2:3]


def get_warped_coord(H, rect):
    if H.dim() == 3:
        warped_coord = torch.stack([WarpCoordinatesWithHomography(
            H_each_plane, rect) for H_each_plane in H], 0)
        warped_X = warped_coord[..., 0:1] / rect[3] * 2 - 1
        warped_Y = warped_coord[..., 1:] / rect[2] * 2 - 1
        warped_normalized = torch.cat([warped_X, warped_Y], -1)
        return warped_normalized
    elif H.dim() == 4:
        wk = []
        for H_each_view in H:
            warped_coord = torch.stack([WarpCoordinatesWithHomography(
                H_each_plane, rect) for H_each_plane in H_each_view], 0)
            warped_X = warped_coord[..., 0:1] / rect[3] * 2 - 1
            warped_Y = warped_coord[..., 1:] / rect[2] * 2 - 1
            warped_normalized = torch.cat([warped_X, warped_Y], -1)
            wk.append(warped_normalized)
        return torch.stack(wk, 0)
    else:
        print('H shape Error')
        return None


def get_warped_coord3d_np(DPC_inv_ints, DPC_p2ws, src_int, src_c2w, rect, inverse=False):
    H = ComputeImageFromPlaneHomographies(
        DPC_inv_ints, DPC_p2ws, src_int, src_c2w)
    if inverse:
        H = np.linalg.inv(H)
    warped_coord = np.stack([WarpCoordinatesWithHomography_np(
        H_each_plane, rect) for H_each_plane in H], 0)
    warped_X = warped_coord[..., 0:1] / rect[3] * 2 - 1
    warped_Y = warped_coord[..., 1:] / rect[2] * 2 - 1
    warped_Z = np.ones_like(warped_X) * np.linspace(-1,
                                                    1, H.shape[0]).reshape((-1, 1, 1, 1))
    warped_normalized = np.concatenate([warped_X, warped_Y, warped_Z], -1)
    return warped_normalized


def get_warped_coord3d_parallel(H, hom_image_coords_t, rect):
    hom_warped_coords = torch.einsum('ijk,ndlk->ndijl', hom_image_coords_t, H)
    hom_warped_coords = hom_warped_coords[..., :-1] / hom_warped_coords[..., 2:3]

    warped_X = hom_warped_coords[..., 0:1] / rect[3] * 2 - 1
    warped_Y = hom_warped_coords[..., 1:] / rect[2] * 2 - 1
    warped_Z = torch.ones_like(warped_X) * torch.linspace(-1, 1,
                                                          H.shape[1], dtype=H.dtype, device=torch.device('cuda')).reshape((1, -1, 1, 1, 1))
    warped_normalized = torch.cat([warped_X, warped_Y, warped_Z], -1)
    return warped_normalized


def get_warped_coord3d(H, rect):
    warped_coord = torch.stack([WarpCoordinatesWithHomography(
        H_each_plane, rect) for H_each_plane in H], 0)
    warped_X = warped_coord[..., 0:1] / rect[3] * 2 - 1
    warped_Y = warped_coord[..., 1:] / rect[2] * 2 - 1
    warped_Z = torch.ones_like(warped_X) * torch.linspace(-1, 1,
                                                          H.shape[0], device=torch.device('cuda')).reshape((-1, 1, 1, 1))
    warped_normalized = torch.cat([warped_X, warped_Y, warped_Z], -1)
    return warped_normalized


def alpha_composition(mpi, get_accum=False, get_disp=False):
    mpiR_alpha = mpi[..., 3:4]  # D H W 1
    mpiR_color = mpi[..., 0:3]  # D H W 3

    tmp = torch.flip(torch.cumprod(torch.flip(1. - mpiR_alpha, [0]), 0), [0])
    tranmittance = torch.roll(tmp, -1, 0)  # exclusive = True
    tranmittance[-1:, ...] = 1

    weights = mpiR_alpha * tranmittance + ALPHA_EPS  # D H W 1
    alpha_acc = torch.sum(weights[..., 0], 0)
    alpha_acc = torch.unsqueeze(alpha_acc, -1)  # H W 1
    rendering = torch.sum(weights * mpiR_color, 0)
    if get_accum is True:
        accum = torch.flip(torch.cumsum(
            torch.flip(weights * mpiR_color, [0]), 0), [0])
    else:
        accum = None

    if get_disp is True:
        disp = torch.div(torch.argmax(mpiR_alpha, dim=0), float(mpi.shape[0]))
    else:
        disp = None
    return rendering, alpha_acc, accum, disp


def render_path_axis(c2w, down, ax, rad, focal, N):
    render_poses = []
    center = c2w[:, 3]
    v = c2w[:, ax] * rad
    for t in np.linspace(-1., 1., N + 1)[:-1]:
        c = center + t * v
        z = normalize((center + focal * c2w[:, 2]) - c)
        render_poses.append(viewmatrix(z, down, c))
    return render_poses


def render_path_spiral(c2w, down, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array(
            [-np.sin(theta), np.cos(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(np.dot(c2w[:3, :4], np.array([0, 0, focal, 1.])) - c)
        render_poses.append(viewmatrix(z, down, c))
    return render_poses


def scene_renderpath(scene, index, pathtype, N, hdr=False):
    im_list = []
    if hdr:
        num_img = len(glob.glob(osp.join(scene, '*.rgbe')))
        idx_list = list(range(num_img))
        im_names = [f'{scene}/im_{i + 1}.rgbe' for i in idx_list]
        seg_names = [f'{scene}/immask_{i + 1}.png' for i in idx_list]

        im = loadHdr(im_names[0])
        seg = 0.5 * (loadImage(seg_names[0]) + 1)[0:1, :, :]
        scale = get_hdr_scale(im, seg, 'TEST')

        for im_name, seg_name in zip(im_names, seg_names):
            im = loadHdr(im_name)
            # seg = 0.5 * (loadImage(seg_name) + 1)[0:1, :, :]
            # scale = get_hdr_scale(im, seg, 'TEST')
            im_list.append(np.clip(im * scale, 0, 1.0))
    else:
        num_img = len(glob.glob(osp.join(scene, '*.jpg')))
        idx_list = list(range(num_img))
        im_names = [f'{scene}/im_{i + 1}.jpg' for i in idx_list]
        for im_name in im_names:
            im_list.append(loadImage(im_name))

    im_list = np.stack(im_list, 0)
    im_list = im_list[index]
    poses_arr = np.load(osp.join(scene, 'poses_bounds.npy'))
    bds = poses_arr[:, -2:].transpose([1, 0])
    bds = [bds.min(), bds.max()]
    
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])[index]
    h, w, f = poses[0, :, -1]
    intrinsic = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=float)
    
    poses = poses[:, :3, :4]
    bottom = np.tile(np.reshape([0, 0, 0, 1.], [1, 4]), (4, 1, 1))
    poses = np.concatenate([poses, bottom], -2)
    poses = recenter(poses)[:, :3, :4].transpose(1, 2, 0) # ref_c2w is Identity

    # c2w = poses_avg(poses)
    # ref_c2w = make_reference_view(np.transpose(poses, [2, 0, 1]))
    ref_c2w = np.identity(4)[:3,:4]

    close_depth, inf_depth = bds[0], bds[1]

    dt = .75
    mean_dz = 1. / ((1. - dt) / close_depth + dt / inf_depth)
    focal = mean_dz

    shrink_factor = 1.0
    zdelta = close_depth * .2

    down = normalize(poses[:3, 1, :].sum(-1))

    tt = ptstocam(poses[:3, 3, :].T, ref_c2w).T
    rads = np.percentile(np.abs(tt), 90, -1) * 1.5

    render_poses = []
    if 'x' in pathtype:
        render_poses += render_path_axis(ref_c2w,down, 0, shrink_factor * rads[0], focal, N)
    if 'y' in pathtype:
        render_poses += render_path_axis(ref_c2w,down, 1, shrink_factor * rads[0], focal, N)
    if 'z' in pathtype:
        render_poses += render_path_axis(ref_c2w,down, 2, shrink_factor * zdelta, focal, N)
    if 'circle' in pathtype:
        render_poses += render_path_spiral(ref_c2w,down, rads, focal, zdelta, 0., 1, N * 2)
    if 'spiral' in pathtype:
        render_poses += render_path_spiral(ref_c2w,down, rads, focal, zdelta, .5, 2, N * 4)

    src_poses = np.transpose(poses, [2, 0, 1])
    src_ints = np.tile(intrinsic, [src_poses.shape[0], 1, 1])
    target_poses = np.array(render_poses)
    target_ints = np.tile(intrinsic, [target_poses.shape[0], 1, 1])

    src_poses = _34_to_44(src_poses)
    target_poses = _34_to_44(target_poses)
    return im_list, src_poses, src_ints, bds, ref_c2w, target_poses, target_ints, [int(h), int(w)]
