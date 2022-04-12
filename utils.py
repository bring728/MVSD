import numpy as np
import struct
from PIL import Image
import cv2
import h5py
import os.path as osp
import imageio
from skimage.measure import block_reduce
import torch


HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision

img_HWC2CHW = lambda x: x.permute(2, 0, 1)
gray2rgb = lambda x: x.unsqueeze(2).repeat(1, 1, 3)


to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
mse2psnr = lambda x: -10. * np.log(x+TINY_NUMBER) / np.log(10.)


def img2mse(x, y, mask=None):
    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        return torch.sum((x - y) * (x - y) * mask) / (torch.sum(mask) * x.shape[0] + TINY_NUMBER)


def img2psnr(x, y, mask=None):
    return mse2psnr(img2mse(x, y, mask).item())


def th_save_img(img_name, img):
    imageio.imwrite(img_name, img)


def th_save_img_accum(img_name, img, accum_name, accum):
    imageio.imwrite(img_name, img)
    imageio.imwrite(accum_name, accum)


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
    im = cv2.imread(imName, -1)
    im = np.transpose(im, [2, 0, 1])
    im = im[::-1, :, :]
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

def loadBinary(imName, resize=False, W=None, H=None):
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
        if resize:
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_AREA)
    return depth[np.newaxis, :, :]

def loadH5(imName):
    try:
        hf = h5py.File(imName, 'r')
        im = np.array(hf.get('data'))
        return im
    except:
        return None

def loadEnvmap(envName, envRow = 120, envCol = 160):
    envHeight = 8
    envWidth = 16

    envHeightOrig, envWidthOrig = 16, 32

    env = cv2.imread(envName, -1)
    env = env.reshape(envRow, envHeightOrig, envCol, envWidthOrig, 3)
    env = np.ascontiguousarray(env.transpose([4, 0, 2, 1, 3]))

    scale = envHeightOrig / envHeight
    if scale > 1:
        env = block_reduce(env, block_size=(1, 1, 1, 2, 2), func=np.mean)

    envInd = np.ones([1, 1, 1], dtype=np.float32)
    return env, envInd


def writeEnvToFile(envmaps, envId, envName, nrows=12, ncols=8, envHeight=8, envWidth=16, gap=1):
    envmap = envmaps[envId, :, :, :, :, :].data.cpu().numpy()
    envmap = np.transpose(envmap, [1, 2, 3, 4, 0])
    envRow, envCol = envmap.shape[0], envmap.shape[1]

    interY = int(envRow / nrows)
    interX = int(envCol / ncols)

    lnrows = len(np.arange(0, envRow, interY))
    lncols = len(np.arange(0, envCol, interX))

    lenvHeight = lnrows * (envHeight + gap) + gap
    lenvWidth = lncols * (envWidth + gap) + gap

    envmapLarge = np.zeros([lenvHeight, lenvWidth, 3], dtype=np.float32) + 1.0
    for r in range(0, envRow, interY):
        for c in range(0, envCol, interX):
            rId = int(r / interY)
            cId = int(c / interX)

            rs = rId * (envHeight + gap)
            cs = cId * (envWidth + gap)
            envmapLarge[rs: rs + envHeight, cs: cs + envWidth, :] = envmap[r, c, :, :, :]

    envmapLarge = np.clip(envmapLarge, 0, 1)
    envmapLarge = (255 * (envmapLarge ** (1.0 / 2.2))).astype(np.uint8)
    cv2.imwrite(envName, envmapLarge[:, :, ::-1])


def writeNumpyEnvToFile(envmap, envName, nrows=12, ncols=8, envHeight=8, envWidth=16, gap=1):
    envRow, envCol = envmap.shape[0], envmap.shape[1]

    interY = int(envRow / nrows)
    interX = int(envCol / ncols)

    lnrows = len(np.arange(0, envRow, interY))
    lncols = len(np.arange(0, envCol, interX))

    lenvHeight = lnrows * (envHeight + gap) + gap
    lenvWidth = lncols * (envWidth + gap) + gap

    envmapLarge = np.zeros([lenvHeight, lenvWidth, 3], dtype=np.float32) + 1.0
    for r in range(0, envRow, interY):
        for c in range(0, envCol, interX):
            rId = int(r / interY)
            cId = int(c / interX)

            rs = rId * (envHeight + gap)
            cs = cId * (envWidth + gap)
            envmapLarge[rs: rs + envHeight, cs: cs + envWidth, :] = envmap[r, c, :, :, :]

    envmapLarge = np.clip(envmapLarge, 0, 1)
    envmapLarge = (255 * envmapLarge ** (1.0 / 2.2)).astype(np.uint8)
    cv2.imwrite(envName, envmapLarge[:, :, ::-1])


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
