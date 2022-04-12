import torch
from utils import *
import torch.nn.functional as F
from collections import OrderedDict



def compute_projection(pixel_batch, pixel_depth, int_list, c2w_list, depth_list, im_list, featmaps):
    h, w = depth_list.shape[2:]
    w2c_list = torch.inverse(c2w_list)

    # get world coord
    pixel_coord = pixel_batch[:, :3]
    norm_coord = torch.inverse(int_list[0]) @ pixel_coord[..., None]
    cam_coord = pixel_depth[:, None] * norm_coord
    # because cam_0 is world
    world_coord = torch.cat([cam_coord, torch.ones_like(cam_coord[:, :1, :])], dim=1)

    # get projection coord
    cam_coord_k = (w2c_list[:, None] @ world_coord[None])[..., :3, :]
    pixel_coord_k = (int_list[:, None] @ cam_coord_k)[..., 0]
    pixel_depth_k = pixel_coord_k[..., 2:3]
    pixel_coord_k = pixel_coord_k[..., :2] / pixel_depth_k

    resize_factor = torch.tensor([w, h]).to(pixel_coord_k.device)[None, None, :]
    pixel_coord_k_norm = (2 * pixel_coord_k / resize_factor - 1.)[:, None]

    # get depth error
    pixel_depth_k_gt_rgb = F.grid_sample(torch.cat([depth_list, im_list], dim=1), pixel_coord_k_norm, align_corners=False)
    proj_err = pixel_depth_k - pixel_depth_k_gt_rgb[:, 0, 0, :, None]
    # torch.nonzero(torch.where(proj_err > 1, 1, 0))
    # torch.nonzero(torch.where(proj_err < -1, 1, 0))

    rgb_sampled = pixel_depth_k_gt_rgb[:, 1:, 0, :]
    feat_sampled = F.grid_sample(featmaps, pixel_coord_k_norm, align_corners=False)[..., 0, :]
    rgb_feat_sampled = torch.cat([rgb_sampled, feat_sampled], dim=1).permute(2, 0, 1)

    viewdir = F.normalize((c2w_list[:, None, :3, :3] @ cam_coord_k)[..., 0], dim=-1)
    rgb_feat_viewdir_err = torch.cat([rgb_feat_sampled, torch.cat([viewdir, proj_err], dim=-1).permute(1, 0, 2)], dim=-1)
    return rgb_feat_viewdir_err


def decompose_single_image(model, gpu, chunk_size, pixels, val_data):
    ret = OrderedDict()
    ret['albedo'] = []
    ret['normal'] = []
    ret['roughness'] = []

    depth_list = val_data['depth_list'][0].cuda(gpu)
    int_list = val_data['int_list'][0].cuda(gpu)
    c2w_list = val_data['c2w_list'][0].cuda(gpu)
    im_list = val_data['im_list'][0].cuda(gpu)
    target_gt = val_data['target_gt'][0].cuda(gpu)
    ch, h, w = target_gt.shape
    depth_gt_flat = depth_list[0].reshape(-1)

    featmaps = model.feature_net(im_list)
    N_rays = pixels.shape[0]
    for i in range(0, N_rays, chunk_size):
        pixel_batch = pixels[i:i + chunk_size]
        depth_gt = depth_gt_flat[i:i + chunk_size][:, None]

        rgb_feat_viewdir_err = compute_projection(pixel_batch, depth_gt, int_list, c2w_list, depth_list, im_list, featmaps)
        brdf = model.brdf_net(rgb_feat_viewdir_err)

        albedo = 0.5 * (brdf[..., :3] + 1)

        x_orig = brdf[..., 3:6]
        norm = torch.sqrt(torch.sum(x_orig * x_orig, dim=1).unsqueeze(1)).expand_as(x_orig)
        normal = x_orig / torch.clamp(norm, min=1e-6)

        roughness = brdf[..., 6:]

        ret['albedo'].append(albedo.cpu())
        ret['normal'].append(normal.cpu())
        ret['roughness'].append(roughness.cpu())

    for k in ret:
        tmp = torch.cat(ret[k], dim=0).reshape((h, w, -1))
        ret[k] = tmp.permute(2, 0, 1)

    albedo = ret['albedo']
    segBRDF = target_gt[8:9].cpu()
    albedo_gt = target_gt[:3].cpu()
    ret['albedo'] = LSregress(albedo * segBRDF, albedo_gt * segBRDF, albedo)

    ret['albedo_gt'] = albedo_gt
    ret['normal_gt'] = target_gt[3:6].cpu()
    ret['roughness_gt'] = target_gt[6:7].cpu()
    ret['segBRDF'] = segBRDF
    ret['segAll'] = target_gt[9:].cpu()
    return ret


def log_view_to_tb(writer, global_step, cfg, gpu, model, val_data, pixels, prefix=''):
    model.switch_to_eval()
    with torch.no_grad():
        ret = decompose_single_image(model, gpu, cfg.chunk_size, pixels, val_data)

    h, w = val_data['target_gt'].shape[2:]

    albedo_gt = ret['albedo_gt'] ** (1.0 / 2.2)
    albedo = ret['albedo'] ** (1.0 / 2.2)
    albedo_all = torch.zeros(3, h, 2 * w)
    albedo_all[:, :, :w] = albedo_gt
    albedo_all[:, :, w:] = albedo

    normal_gt = 0.5 * (ret['normal_gt'] + 1)
    normal = 0.5 * (ret['normal'] + 1)
    normal_all = torch.zeros(3, h, 2 * w)
    normal_all[:, :, :w] = normal_gt
    normal_all[:, :, w:] = normal

    roughness_gt = 0.5 * (ret['roughness_gt'] + 1)
    roughness = 0.5 * (ret['roughness'] + 1)
    rough = torch.zeros(1, h, 2 * w)
    rough[:, :, :w] = roughness_gt
    rough[:, :, w:] = roughness

    # write the pred/gt rgb images and depths
    writer.add_image(prefix + 'albedo', albedo_all, global_step)
    writer.add_image(prefix + 'normal', normal_all, global_step)
    writer.add_image(prefix + 'roughness', rough, global_step)

    psnr_albedo = img2psnr(ret['albedo'], ret['albedo_gt'], ret['segBRDF'])
    writer.add_scalar(prefix + 'psnr_albedo', psnr_albedo, global_step)
    psnr_normal = img2psnr(ret['normal'], ret['normal_gt'], ret['segAll'])
    writer.add_scalar(prefix + 'psnr_normal', psnr_normal, global_step)
    psnr_rough = img2psnr(ret['roughness'], ret['roughness_gt'], ret['segBRDF'])
    writer.add_scalar(prefix + 'psnr_rough', psnr_rough, global_step)

    model.switch_to_train()