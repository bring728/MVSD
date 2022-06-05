import os

from torch.utils.data import Dataset
import time
from utils import *
from utils_geometry import *
from utils_geometry import _34_to_44
import scipy.ndimage as ndimage
import os.path as osp
import six
import lmdb
from PIL import Image
import pyarrow as pa
import pickle

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
        # sceneList = ['mainDiffLight_xml1/scene0107_00/18_', ]
        self.nameList = []
        self.matrixList = []

        idx_list = list(range(self.num_view_all))
        for scene in sceneList:
            self.nameList.append([scene + '{}_' + f'{i + 1}' + '.{}' for i in idx_list])
            self.matrixList.append(np.load(osp.join(dataRoot, scene + 'poses_bounds.npy'))[:3, :5, :])
        self.matrixList = np.stack(self.matrixList)
        self.length = len(self.nameList)

    def __len__(self):
        return self.length

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
        depth_list = []
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

            depth = loadBinary(name_list[idx].format('imdepth', 'dat'))
            depth_list.append(depth)

            depthest = loadBinary(name_list[idx].format('depthest', 'dat'))
            depthest_list.append(depthest)
            conf_list.append(loadBinary(name_list[idx].format('conf', 'dat')))

            if self.cfg.BRDF.input_feature == 'rgbdc':
                depth_norm_list.append(loadBinary(name_list[idx].format('depthestnorm', 'dat')))

        batchDict['rgb'] = np.stack(rgb_list, 0).astype(np.float32)
        batchDict['depth'] = np.stack(depth_list, 0).astype(np.float32)
        batchDict['depthest'] = np.stack(depthest_list, 0).astype(np.float32)

        batchDict['conf'] = np.stack(conf_list, 0).astype(np.float32)
        if self.cfg.BRDF.input_feature == 'rgbdc':
            batchDict['depth_norm'] = np.stack(depth_norm_list, 0).astype(np.float32)

        batchDict['cam'] = np.stack(src_int_list, 0).astype(np.float32)
        # recenter to cam_0
        w2target = np.linalg.inv(src_c2w_list[0])
        batchDict['c2w'] = (w2target @ np.stack(src_c2w_list, 0)).astype(np.float32)

        # normal_est = loadH5(name_list[target_idx].format('normalest', 'h5'))
        # batchDict['normal'] = normal_est
        # DL = loadH5(name_list[target_idx].format('DLest', 'h5'))
        # batchDict['DL'] = DL

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
    def __init__(self, dataRoot, cfg, stage, phase='TRAIN', debug=True, gpu=0):
        self.gpu = gpu
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
        idx_list = list(range(self.cfg.num_view_all))
        for scene in sceneList:
            self.nameList += [scene + '{}_' + f'{i + 1}' + '.{}' for i in idx_list]
        # self.nameList = np.array(self.nameList).astype(np.string_)

        self.length = len(self.nameList)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        batchDict = {}
        name = osp.join(self.dataRoot, self.nameList[ind])
        # name = osp.join(self.dataRoot, str(self.nameList[ind], encoding='utf-8'))
        batchDict['name'] = name
        im = loadHdr(name.format('im', 'rgbe'))
        seg = loadImage(name.format('immask', 'png'))[0:1, :, :]
        scene_scale = get_hdr_scale(im, seg, self.phase)
        im = np.clip(im * scene_scale, 0, 1.0)

        segArea = np.logical_and(seg > 0.49, seg < 0.51)
        segObj = (seg > 0.9)
        segAll = segArea + segObj
        segBRDF = ndimage.binary_erosion(segObj.squeeze(), structure=np.ones((7, 7)), border_value=1)[np.newaxis, :, :]
        seg = np.concatenate([segBRDF, segAll], axis=0).astype(np.float32)
        batchDict['mask'] = seg

        conf = loadBinary(name.format('conf', 'dat'))
        depthmvs_normalized = loadBinary(name.format('depthnorm', 'dat'))
        input_data = np.concatenate([im, depthmvs_normalized, conf], axis=0).astype(np.float32)
        batchDict['input'] = input_data

        if self.stage == '1-1':
            normal = loadImage(name.format('imnormal', 'png'), normalize_01=False)
            normal = (normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5))[np.newaxis, :]).astype(np.float32)
            batchDict['normal_gt'] = normal

        if self.stage == '1-2':
            envmaps, envmapsInd = loadEnvmap(name.format('imenvDirect', 'hdr'), self.cfg.DL.env_height, self.cfg.DL.env_width,
                                             self.cfg.DL.env_rows,
                                             self.cfg.DL.env_cols)
            envmaps = envmaps * scene_scale
            batchDict['envmaps_gt'] = envmaps.astype(np.float32)
            batchDict['envmapsInd'] = envmapsInd

        return batchDict




# class Openrooms_LMDB_single(Dataset):
#     def __init__(self, db_path, cfg, stage, gpu, lock):
#         self.gpu = gpu
#         self.lock = lock
#         self.cfg = cfg
#         self.stage = stage
#         self.db_path = db_path
#         env = lmdb.open(self.db_path, subdir=osp.isdir(self.db_path),
#                         readonly=True, lock=False,
#                         readahead=False, meminit=False)
#         with env.begin(write=False) as txn:
#             # self.length = pickle.loads(txn.get(b'__len__'))
#             # self.keys = pickle.loads(txn.get(b'__keys__'))
#             self.length = pa.deserialize(txn.get(b'__len__'))
#             self.keys = pa.deserialize(txn.get(b'__keys__'))
#
#     def open_lmdb(self):
#          # print('reset')
#          self.env = lmdb.open(self.db_path, subdir=osp.isdir(self.db_path),
#                               readonly=True, lock=False,
#                               readahead=False, meminit=False)
#          self.txn = self.env.begin(write=False, buffers=True)
#
#     def __getitem__(self, index):
#         if not hasattr(self, 'txn'):
#             self.open_lmdb()
#
#         self.lock.acquire()
#         print('get lock: ', self.gpu, torch.utils.data.get_worker_info().id)
#         start = time.time()
#         byteflow = self.txn.get(self.keys[index])
#         print(time.time() - start)
#         self.lock.release()
#         # unpacked = pickle.loads(byteflow)
#         unpacked = pa.deserialize(byteflow)
#
#         rgb, mask, conf, normmax, normzero, midas, normal_gt, envmaps_gt, envmapsInd, name = unpacked
#         # imgbuf = unpacked[0]
#         # buf = six.BytesIO()
#         # buf.write(imgbuf)
#         # buf.seek(0)
#         # img = Image.open(buf).convert('RGB')
#         batchDict = {}
#         if self.cfg.normal.depth_type == 'mvs':
#             if self.cfg.normal.norm_type == 'max':
#                 input_data = np.concatenate([rgb, normmax, conf], axis=0).astype(np.float32)
#             else:
#                 input_data = np.concatenate([rgb, normzero, conf], axis=0).astype(np.float32)
#         else:
#             input_data = np.concatenate([rgb, midas], axis=0).astype(np.float32)
#         batchDict['input'] = torch.tensor(input_data)
#         batchDict['name'] = name
#         batchDict['mask'] = torch.tensor(mask)
#         if self.stage == '1-1':
#             batchDict['normal_gt'] = torch.tensor(normal_gt)
#         if self.stage == '1-2':
#             batchDict['envmaps_gt'] = torch.tensor(envmaps_gt)
#             batchDict['envmapsInd'] = torch.tensor(envmapsInd)
#         return batchDict
#
#     def __len__(self):
#         return self.length
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + self.db_path + ')'
#
# class Openrooms_FF_h5(Dataset):
#     def __init__(self, dataRoot, cfg, stage, phase='TRAIN', debug=True):
#         self.stage = stage
#         if stage == '1-1' or stage == '1-2':
#             stage = '1'
#         self.phase = phase
#         self.dataRoot = osp.join(dataRoot, 'stage-' + stage, phase.lower())
#         self.cfg = cfg
#         self.nameList = os.listdir(self.dataRoot)
#         if debug:
#             self.nameList = self.nameList[:100]
#         self.length = len(self.nameList)
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, ind):
#         batchDict = {}
#         if self.stage == '1-1':
#             name, mask, dc, hdr, max_intensity, normal, _, _ = loadH5_stage(osp.join(self.dataRoot, self.nameList[ind]), self.stage)
#         elif self.stage == '1-2':
#             name, mask, dc, hdr, max_intensity, _, DL_gt, DL_ind = loadH5_stage(osp.join(self.dataRoot, self.nameList[ind]), self.stage)
#
#         if self.phase == 'TRAIN':
#             scale = (0.95 - 0.1 * np.random.random()) / np.clip(max_intensity, 0.1, None)
#         elif self.phase == 'TEST':
#             scale = (0.95 - 0.05) / np.clip(max_intensity, 0.1, None)
#         else:
#             raise Exception('phase error')
#         hdr = np.clip(hdr * scale, 0, 1)
#         batchDict['name'] = name
#         batchDict['mask'] = mask
#         batchDict['input'] = np.concatenate([hdr, dc], axis=1).astype(np.float32)
#         if self.stage == '1-1':
#             batchDict['normal_gt'] = normal
#         elif self.stage == '1-2':
#             batchDict['envmaps_gt'] = DL_gt
#             batchDict['envmapsInd'] = DL_ind
#         return batchDict
#
# class Openrooms_FF_single_offline(Dataset):
#     def __init__(self, dataRoot, stage, phase):
#         self.dataRoot = dataRoot
#         self.stage = stage
#         self.phase = phase
#
#         if phase == 'TRAIN':
#             sceneFile = osp.join(dataRoot, 'train.txt')
#         elif phase == 'TEST':
#             sceneFile = osp.join(dataRoot, 'test.txt')
#         else:
#             raise Exception('Unrecognized phase for data loader')
#
#         with open(sceneFile, 'r') as fIn:
#             sceneList = fIn.readlines()
#         sceneList = [a.strip() for a in sceneList]
#
#         self.nameList = []
#         idx_list = list(range(9))
#         for scene in sceneList:
#             self.nameList += [scene + '{}_' + f'{i + 1}' + '.{}' for i in idx_list]
#         # self.nameList = np.array(self.nameList).astype(np.string_)
#         self.length = len(self.nameList)
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, ind):
#         batchDict = {}
#         name = osp.join(self.dataRoot, self.nameList[ind])
#         batchDict['name'] = name
#
#         im = loadHdr(name.format('im', 'rgbe'))
#         batchDict['hdr'] = im.astype(np.float32)
#         seg = loadImage(name.format('immask', 'png'))[0:1, :, :]
#         # scene_scale = get_hdr_scale(im, seg, self.phase)
#         length = im.shape[0] * im.shape[1] * im.shape[2]
#         intensityArr = (im * seg).flatten()
#         intensityArr.sort()
#         intensity_almost_max = intensityArr[int(0.95 * length)]
#         batchDict['max_intensity'] = np.array(intensity_almost_max).reshape((1))
#
#         segArea = np.logical_and(seg > 0.49, seg < 0.51)
#         segObj = (seg > 0.9)
#         segAll = segArea + segObj
#         segBRDF = ndimage.binary_erosion(segObj.squeeze(), structure=np.ones((7, 7)), border_value=1)[np.newaxis, :, :]
#         seg = np.concatenate([segBRDF, segAll], axis=0).astype(np.float32)  # segBRDF, segAll
#         batchDict['mask'] = seg
#
#         conf = loadBinary(name.format('conf', 'dat'))
#         depthmvs_normalized = loadBinary(name.format('depthnormzero', 'dat'))
#         dc = np.concatenate([depthmvs_normalized, conf], axis=0).astype(np.float32)
#         batchDict['dc'] = dc
#
#         normal = loadImage(name.format('imnormal', 'png'), normalize_01=False)
#         normal = (normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5))[np.newaxis, :]).astype(np.float32)
#         batchDict['normal_gt'] = normal
#
#         envmaps, envmapsInd = loadEnvmap(name.format('imenvDirect', 'hdr'), self.cfg.DL.env_height, self.cfg.DL.env_width,
#                                          self.cfg.DL.env_rows,
#                                          self.cfg.DL.env_cols)
#         # envmaps = envmaps * scene_scale
#         batchDict['envmaps_gt'] = envmaps.astype(np.float32)
#         batchDict['envmapsInd'] = envmapsInd
#         return batchDict
