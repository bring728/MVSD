import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *
import random
from utils_geometry import *
from utils_geometry import _34_to_44
import scipy.ndimage as ndimage
import os.path as osp


class Openrooms_FF(Dataset):
    def __init__(self, dataRoot, cfg, stage, phase='TRAIN', debug=True):
        self.num_view_min = cfg.num_view_min
        self.num_view_all = cfg.num_view_all
        self.range = self.num_view_all - self.num_view_min + 1
        self.dataRoot = dataRoot
        self.cfg = cfg
        self.stage = stage
        self.phase = phase

        if phase == 'TRAIN' and debug:
            sceneFile = osp.join(dataRoot, 'train_debug.txt')
        elif phase == 'TEST' and debug:
            sceneFile = osp.join(dataRoot, 'test_debug.txt')
        elif phase == 'TRAIN' and not debug:
            sceneFile = osp.join(dataRoot, 'train.txt')
        elif phase == 'TEST' and not debug:
            sceneFile = osp.join(dataRoot, 'test.txt')
        else:
            raise Exception('Unrecognized phase for data loader')

        with open(sceneFile, 'r') as fIn:
            sceneList = fIn.readlines()
        sceneList = [a.strip() for a in sceneList]

        self.nameList = []
        self.matrixList = []

        idx_list = list(range(self.num_view_all))
        for scene in sceneList:
            self.nameList.append([scene + '{}_' + f'{i + 1}' + '.{}' for i in idx_list])
            self.matrixList.append(np.load(osp.join(dataRoot ,scene + 'poses_bounds.npy'))[:3, :5, :])
        self.matrixList = np.stack(self.matrixList)
        self.count = len(self.nameList)

    def __len__(self):
        return self.count

    def __getitem__(self, ind):
        batchDict = {}
        name_list = [osp.join(self.dataRoot, a) for a in self.nameList[ind]]
        batchDict['name'] = name_list[0]

        num_view = self.num_view_min + np.random.choice(self.range, 1)[0]
        # num_view = self.num_view_all
        training_pair_idx = np.random.choice(self.num_view_all, num_view, replace=False)
        target_idx = training_pair_idx[0]

        if self.cfg.depth_gt:
            target_depth_norm = loadBinary(name_list[target_idx].format('imdepth', 'dat')).astype(np.float32)
            batchDict['target_depth_norm'] = target_depth_norm / target_depth_norm.max()
        else:
            target_depth_norm = loadBinary(name_list[target_idx].format('depthestnorm', 'dat')).astype(np.float32)
            batchDict['target_depth_norm'] = target_depth_norm

        target_im = loadHdr(name_list[target_idx].format('im', 'rgbe'))
        seg = loadImage(name_list[target_idx].format('immask', 'png'))[0:1, :, :]
        scene_scale = get_hdr_scale(target_im, seg, self.phase)

        segArea = np.logical_and(seg > 0.49, seg < 0.51)
        segObj = (seg > 0.9)
        if self.stage == '3':
            segObj = ndimage.binary_erosion(segObj.squeeze(), structure=np.ones((7, 7)), border_value=1)[np.newaxis, :, :]
            seg = np.concatenate([segObj, segArea + segObj], axis=0).astype(np.float32)  # segBRDF, segAll
        else:
            seg = np.concatenate([segObj, segArea + segObj], axis=0).astype(np.float32)  # segBRDF, segAll
        batchDict['mask'] = seg

        src_c2w_list = []
        src_int_list = []
        rgb_list = []
        # depth_list = []
        depthest_list = []
        conf_list = []
        depth_norm_list = []
        for idx in training_pair_idx:
            im = loadHdr(name_list[idx].format('im', 'rgbe'))
            im = np.clip(im * scene_scale, 0, 1.0)
            poses = self.matrixList[ind, ..., idx]
            src_c2w_list.append(_34_to_44(poses[:, :4]))
            h, w, f = poses[:, -1]
            intrinsic = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=float)
            src_int_list.append(intrinsic)
            rgb_list.append(im)

            # if self.cfg.depth_gt:
            #     depth = loadBinary(name_list[idx].format('imdepth', 'dat'))
            # else:
            #     depth = loadBinary(name_list[idx].format('depthest', 'dat'))
            # depth = loadBinary(name_list[idx].format('imdepth', 'dat'))
            # depth_list.append(depth)

            depthest = loadBinary(name_list[idx].format('depthest', 'dat'))
            depthest_list.append(depthest)
            conf_list.append(loadBinary(name_list[idx].format('conf', 'dat')))

            if self.cfg.BRDF.input_feature == 'rgbdc':
                depth_norm_list.append(loadBinary(name_list[idx].format('depthestnorm', 'dat')))

        batchDict['rgb'] = np.stack(rgb_list, 0).astype(np.float32)
        # batchDict['depth'] = np.stack(depth_list, 0).astype(np.float32)
        batchDict['depthest'] = np.stack(depthest_list, 0).astype(np.float32)

        batchDict['conf'] = np.stack(conf_list, 0).astype(np.float32)
        if self.cfg.BRDF.input_feature == 'rgbdc':
            batchDict['depth_norm'] = np.stack(depth_norm_list, 0).astype(np.float32)

        batchDict['cam'] = np.stack(src_int_list, 0).astype(np.float32)
        # recenter to cam_0
        w2target = np.linalg.inv(src_c2w_list[0])
        batchDict['c2w'] = (w2target @ np.stack(src_c2w_list, 0)).astype(np.float32)

        normal_est = loadH5(name_list[target_idx].format('normalest', 'h5'))
        batchDict['normal'] = normal_est
        DL = loadH5(name_list[target_idx].format('DLest', 'h5'))
        batchDict['DL'] = DL

        if self.stage == '2':
            albedo = loadImage(name_list[target_idx].format('imbaseColor', 'png'))
            batchDict['albedo_gt'] = (albedo ** 2.2).astype(np.float32)
            rough = loadImage(name_list[target_idx].format('imroughness', 'png'))[0:1, :, :].astype(np.float32)
            batchDict['rough_gt'] = rough

        if self.stage == '3':
            envmaps, envmapsInd = loadEnvmap(name_list[target_idx].format('imenv', 'hdr'), self.cfg.GL.env_height, self.cfg.GL.env_width,
                                             self.cfg.GL.env_rows, self.cfg.GL.env_cols)
            envmaps = envmaps * scene_scale
            batchDict['envmaps_gt'] = envmaps.astype(np.float32)
            batchDict['envmapsInd'] = envmapsInd
        return batchDict


class Openrooms_FF_single(Dataset):
    def __init__(self, dataRoot, cfg, stage, phase='TRAIN', debug=True):
        self.dataRoot = dataRoot
        self.cfg = cfg
        self.stage = stage
        self.phase = phase

        if phase == 'TRAIN' and debug:
            sceneFile = osp.join(dataRoot, 'train_debug.txt')
        elif phase == 'TEST' and debug:
            sceneFile = osp.join(dataRoot, 'test_debug.txt')
        elif phase == 'TRAIN' and not debug:
            sceneFile = osp.join(dataRoot, 'train.txt')
        elif phase == 'TEST' and not debug:
            sceneFile = osp.join(dataRoot, 'test.txt')
        elif phase == 'ALL':
            sceneFile = osp.join(dataRoot, 'all.txt')
        else:
            raise Exception('Unrecognized phase for data loader')

        with open(sceneFile, 'r') as fIn:
            sceneList = fIn.readlines()
        sceneList = [a.strip() for a in sceneList]

        self.nameList = []
        idx_list = list(range(self.cfg.num_view_all))
        for scene in sceneList:
            self.nameList += [scene + '{}_' + f'{i + 1}' + '.{}' for i in idx_list]

        self.count = len(self.nameList)

    def __len__(self):
        return self.count

    def __getitem__(self, ind):
        batchDict = {}
        name = osp.join(self.dataRoot, self.nameList[ind])
        batchDict['name'] = name

        im = loadHdr(name.format('im', 'rgbe'))
        seg = loadImage(name.format('immask', 'png'))[0:1, :, :]
        scene_scale = get_hdr_scale(im, seg, self.phase)

        segArea = np.logical_and(seg > 0.49, seg < 0.51)
        segObj = (seg > 0.9)
        if self.stage == '1-2':
            segObj = ndimage.binary_erosion(segObj.squeeze(), structure=np.ones((7, 7)), border_value=1)[np.newaxis, :, :]
            seg = np.concatenate([segObj, segArea + segObj], axis=0).astype(np.float32)  # segBRDF, segAll
        else:
            seg = np.concatenate([segObj, segArea + segObj], axis=0).astype(np.float32)  # segBRDF, segAll
        batchDict['mask'] = seg

        # depthest = loadBinary(name.format('depthest', 'dat'))
        # conf = loadBinary(name.format('conf', 'dat'))
        # np.unravel_index(np.argmax(conf, axis=None), conf.shape)
        # depthgt = loadBinary(name.format('imdepth', 'dat'))
        # mask = conf > 0.8
        # depthest2, coef = LSregress(torch.from_numpy(depthest * mask), torch.from_numpy(depthgt * mask), torch.from_numpy(depthest))
        # depthest2 = depthest2.numpy()
        # depthest3 = depthest / 4 * 3
        # a = np.sum(np.abs(depthest - depthgt) * mask) / (np.sum(mask) * (depthest.shape[1] / mask.shape[1]) + TINY_NUMBER)
        # b = np.mean(np.abs(depthest - depthgt))
        # c = np.sum(np.abs(depthest2 - depthgt) * mask) / (np.sum(mask) * (depthest2.shape[1] / mask.shape[1]) + TINY_NUMBER)
        # d = np.mean(np.abs(depthest2 - depthgt))
        # e = np.sum(np.abs(depthest3 - depthgt) * mask) / (np.sum(mask) * (depthest3.shape[1] / mask.shape[1]) + TINY_NUMBER)
        # f = np.mean(np.abs(depthest3 - depthgt))
        # tt = {'a': a}
        # return tt

        im = np.clip(im * scene_scale, 0, 1.0)
        if self.cfg.normal.depth_type == 'mvs':
            conf = loadBinary(name.format('conf', 'dat'))
            if self.cfg.normal.norm_type == 'max':
                depthmvs_normalized = loadBinary(name.format('depthnormmax', 'dat'))
                input_data = np.concatenate([im, depthmvs_normalized, conf], axis=0).astype(np.float32)
            else:
                depthmvs_normalized = loadBinary(name.format('depthnormzero', 'dat'))
                input_data = np.concatenate([im, depthmvs_normalized, conf], axis=0).astype(np.float32)
        else:
            midas_normalized = loadBinary(name.format('midas', 'dat'))
            input_data = np.concatenate([im, midas_normalized], axis=0).astype(np.float32)
        batchDict['input'] = input_data

        if self.stage == '1-1' and not self.phase == 'ALL':
            normal = loadImage(name.format('imnormal', 'png'), normalize_01=False)
            normal = (normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5))[np.newaxis, :]).astype(np.float32)
            batchDict['normal_gt'] = normal

        if self.stage == '1-2' and not self.phase == 'ALL':
            envmaps, envmapsInd = loadEnvmap(name.format('imenvDirect', 'hdr'), self.cfg.DL.env_height, self.cfg.DL.env_width,
                                             self.cfg.DL.env_rows,
                                             self.cfg.DL.env_cols)
            envmaps = envmaps * scene_scale
            batchDict['envmaps_gt'] = envmaps.astype(np.float32)
            batchDict['envmapsInd'] = envmapsInd
        return batchDict
