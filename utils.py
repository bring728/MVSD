import numpy as np
import sys, re, struct
from PIL import Image
import os.path as osp
import imageio
from skimage.measure import block_reduce
import torch

HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6  # float32 only has 7 decimal digits precision
eps = 1e-7

img_HWC2CHW = lambda x: x.permute(2, 0, 1)
img_CHW2HWC = lambda x: x.permute(1, 2, 0)
img_rgb2bgr = lambda x: x[[2, 1, 0]]
img_bgr2rgb = lambda x: x[[2, 1, 0]]

gray2rgb = lambda x: x.unsqueeze(2).repeat(1, 1, 3)

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
mse2psnr = lambda x: -10. * np.log(x + TINY_NUMBER) / np.log(10.)
psnr2mse = lambda x: 10 ** -(x / (10.0))


# img is B C H W
def img2mse(x, y, mask=None):
    if mask is None:
        return torch.mean((x - y) ** 2)
    else:
        return torch.sum((x - y) ** 2 * mask) / (torch.sum(mask) + TINY_NUMBER)


# img is B C H W
def img2angerr(x, y, mask=None):
    tmp = torch.clamp(torch.sum(x * y, dim=1, keepdim=True), min=-1 + eps, max=1 - eps)
    if mask is None:
        return torch.mean(torch.acos(tmp))
    else:
        return torch.sum(torch.acos(tmp) * mask) / (torch.sum(mask) + TINY_NUMBER)


# img is B C H W
def img2log_mse(x, y, mask=None):
    if mask is None:
        return torch.mean(torch.log(x + 1.0) - torch.log(y + 1.0) ** 2)
    else:
        return torch.sum((torch.log(x + 1.0) - torch.log(y + 1.0)) ** 2 * mask) / (torch.sum(mask) + TINY_NUMBER)


def img2psnr(x, y, mask=None):
    return mse2psnr(img2mse(x, y, mask).item())


def env_vis(env, h, w, r, c):
    env = torch.permute(env / torch.max(env), (0, 1, 3, 2, 4)).reshape(3, r * h, c * w)
    return env


def th_save_img(img_name, img):
    imageio.imwrite(img_name, img)


def th_save_img_accum(img_name, img, accum_name, accum):
    imageio.imwrite(img_name, img)
    imageio.imwrite(accum_name, accum)


def tocuda(vars, gpu):
    if isinstance(vars, list):
        out = []
        for var in vars:
            if isinstance(var, torch.Tensor):
                out.append(var.to(gpu))
            elif isinstance(var, str):
                out.append(var)
            else:
                raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))
        return out
    elif isinstance(vars, dict):
        out = {}
        for k in vars:
            if isinstance(vars[k], torch.Tensor):
                out[k] = vars[k].to(gpu)
            elif isinstance(vars[k], str):
                out[k] = vars[k]
            elif isinstance(vars[k], list):
                out[k] = vars[k]
            else:
                raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))
        return out


def cvimg_from_torch(tensor, iscolor=True):
    if iscolor:
        tensor = img_rgb2bgr(tensor)
    tensor = img_CHW2HWC(tensor)
    image = np.clip((tensor.detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
    return image


def srgb2rgb(srgb):
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret


def loadImage(imName, isGama=False, resize=False, W=None, H=None):
    if not (osp.isfile(imName)):
        print(imName)
        assert (False)

    im = Image.open(imName)
    if resize:
        im = im.resize([W, H], Image.ANTIALIAS)

    im = np.asarray(im, dtype=float)
    if isGama:
        im = (im / 255.0) ** 2.2
        im = 2 * im - 1
    else:
        im = (im - 127.5) / 127.5
    if len(im.shape) == 2:
        im = im[:, np.newaxis]
    im = np.transpose(im, [2, 0, 1])
    return im


def loadHdr(imName):
    if not (osp.isfile(imName)):
        print(imName)
        raise Exception(imName, 'image doesnt exists')
    # im = cv2.imread(imName, -1)
    im = np.array(imageio.imread_v2(imName, format='HDR-FI'))
    if im is None:
        print(imName)
        raise Exception(imName, 'image doesnt exists 2')
    im = np.transpose(im, [2, 0, 1])
    # im = im[::-1, :, :]
    return im


def get_hdr_scale(hdr, seg, phase):
    length = hdr.shape[0] * hdr.shape[1] * hdr.shape[2]
    intensityArr = (hdr * seg).flatten()
    intensityArr.sort()
    intensity_almost_max = intensityArr[int(0.95 * length)]

    if phase.upper() == 'TRAIN' or phase.upper() == 'DEBUG':
        scale = (0.95 - 0.1 * np.random.random()) / np.clip(intensity_almost_max, 0.1, None)
    elif phase.upper() == 'TEST':
        scale = (0.95 - 0.05) / np.clip(intensity_almost_max, 0.1, None)
    else:
        raise Exception('!!')
    return scale


def loadBinary(imName):
    if not (osp.isfile(imName)):
        print(imName)
        assert (False)
    with open(imName, 'rb') as fIn:
        hBuffer = fIn.read(4)
        height = struct.unpack('i', hBuffer)[0]
        wBuffer = fIn.read(4)
        width = struct.unpack('i', wBuffer)[0]
        dBuffer = fIn.read(4 * width * height)
        depth = np.asarray(struct.unpack('f' * height * width, dBuffer))
        depth = depth.reshape([height, width])
    return depth[np.newaxis, :, :]


def loadEnvmap(envName, env_height, env_width, env_rows, env_cols):
    envHeightOrig, envWidthOrig = 16, 32
    assert ((envHeightOrig / env_height) == (envWidthOrig / env_width))
    assert (envHeightOrig % env_height == 0)

    # env = cv2.imread(envName, -1)
    env = np.array(imageio.imread_v2(envName, format='HDR-FI'))
    if not env is None:
        env = env.reshape(env_rows, envHeightOrig, env_cols, envWidthOrig, 3)
        env = np.ascontiguousarray(env.transpose([4, 0, 2, 1, 3]))
        scale = envHeightOrig / env_height
        if scale > 1:
            env = block_reduce(env, block_size=(1, 1, 1, 2, 2), func=np.mean)
        envInd = np.ones([1, 1, 1], dtype=np.float32)
        return env, envInd

    else:
        raise Exception('env does not exist')


def predToShading(pred, envWidth=32, envHeight=16, SGNum=12):
    Az = ((np.arange(envWidth) + 0.5) / envWidth - 0.5) * 2 * np.pi
    El = ((np.arange(envHeight) + 0.5) / envHeight) * np.pi / 2.0
    Az, El = np.meshgrid(Az, El)
    Az = Az[np.newaxis, :, :]
    El = El[np.newaxis, :, :]
    lx = np.sin(El) * np.cos(Az)
    ly = np.sin(El) * np.sin(Az)
    lz = np.cos(El)
    ls = np.concatenate((lx, ly, lz), axis=0)
    ls = ls[np.newaxis, :, np.newaxis, np.newaxis, :, :]
    envWeight = np.cos(El) * np.sin(El)
    envWeight = envWeight[np.newaxis, np.newaxis, np.newaxis, :, :]

    envRow, envCol = pred.shape[2], pred.shape[3]
    pred = pred.squeeze(0)
    axisOrig = pred[0:3 * SGNum, :, :]
    lambOrig = pred[3 * SGNum: 4 * SGNum, :, :]
    weightOrig = pred[4 * SGNum: 7 * SGNum, :, :]

    weight = weightOrig.reshape([SGNum, 3, envRow, envCol]) * 0.999
    weight = np.tan(np.pi / 2.0 * weight)
    weight = weight[:, :, :, :, np.newaxis, np.newaxis]

    axisDir = axisOrig.reshape([SGNum, 3, envRow, envCol])
    axisDir = axisDir[:, :, :, :, np.newaxis, np.newaxis]

    lamb = lambOrig.reshape([SGNum, 1, envRow, envCol]) * 0.999
    lamb = np.tan(np.pi / 2.0 * lamb)
    lamb = lamb[:, :, :, :, np.newaxis, np.newaxis]

    mi = lamb * (np.sum(axisDir * ls, axis=1)[:, np.newaxis, :, :, :, :] - 1)
    envmaps = np.sum(weight * np.exp(mi), axis=0)

    shading = (envmaps * envWeight).reshape([3, envRow, envCol, -1])
    shading = np.sum(shading, axis=3)
    shading = np.maximum(shading, 0.0)

    return shading


def LSregress(pred, gt, origin):
    nb = pred.size(0)
    origSize = pred.size()
    pred = pred.reshape(nb, -1)
    gt = gt.reshape(nb, -1)

    coef = (torch.sum(pred * gt, dim=1) / torch.clamp(torch.sum(pred * pred, dim=1), min=1e-5)).detach()
    coef = torch.clamp(coef, 0.001, 1000)
    for n in range(0, len(origSize) - 1):
        coef = coef.unsqueeze(-1)
    # pred = pred.reshape(origSize)
    predNew = origin * coef.expand(origSize)
    return predNew


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def outdir2xml(scene):
    a = scene.split('data_FF_10_640')
    b = a[1].split('/')
    scene_name = b[2]
    return a[0] + 'scenes/' + b[1].split('_')[1] + '/' + scene_name + '/' + b[1].split('_')[0] + '_FF.xml'


def xml2camtxt(xml, k):
    scene_type = xml.split('/')[-1].split('_')[0]
    return f'{osp.dirname(xml)}/{k}_cam_{scene_type}_FF.txt'


def xml2outdir(xml):
    split = xml.split('.')[0].split('/')
    root = xml.split('scenes')[0] + 'data_FF_10_640'
    if 'mainDiffLight' in split[-1]:
        scene_type = 'mainDiffLight'
    elif 'mainDiffMat' in split[-1]:
        scene_type = 'mainDiffMat'
    else:
        scene_type = 'main'
    scene_name = split[-2]
    xml_name = split[-3]
    return osp.join(root, scene_type + '_' + xml_name, scene_name)


def camtxt2outdir(camtxt):
    split = camtxt.split('.')[0].split('/')
    root = camtxt.split('scenes')[0] + 'data_FF_10_640'
    if 'mainDiffLight' in split[-1]:
        scene_type = 'mainDiffLight'
    elif 'mainDiffMat' in split[-1]:
        scene_type = 'mainDiffMat'
    else:
        scene_type = 'main'
    scene_name = split[-2]
    xml_name = split[-3]
    return osp.join(root, scene_type + '_' + xml_name, scene_name)
