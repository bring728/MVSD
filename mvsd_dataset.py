import os

from torch.utils.data import Dataset
import time
import random
from utils import *
from utils_geometry import *
from utils_geometry import _34_to_44
import scipy.ndimage as ndimage
import os.path as osp


class Openrooms_FF(Dataset):
    def __init__(self, dataRoot, cfg, stage, phase='TRAIN', debug=False):
        self.num_view_all = cfg.num_view_all
        self.dataRoot = dataRoot
        self.cfg = cfg
        self.mode = cfg.mode
        self.stage = stage
        self.phase = phase
        self.idx_list = list(range(1, self.num_view_all + 1))
        self.idx_list = list(map(str, self.idx_list))

        if phase == 'TRAIN':
            sceneFile = osp.join(dataRoot, 'train.txt')
        elif phase == 'TEST':
            sceneFile = osp.join(dataRoot, 'test.txt')
        else:
            raise Exception('Unrecognized phase for data loader')

        with open(sceneFile, 'r') as fIn:
            sceneList = fIn.readlines()
        sceneList = [a.strip() for a in sceneList]
        if debug:
            sceneList = sceneList[:30]
        self.nameList = []
        self.matrixList = []
        for scene in sceneList:
            # self.nameList += [scene + '{}_' + f'{i + 1}' + '.{}' for i in idx_list]
            if phase == 'TRAIN':
                self.nameList += [scene + '$' + i for i in self.idx_list]
            elif phase == 'TEST':
                self.nameList += [scene + '$5']
        # self.matrixList.append(np.load(osp.join(dataRoot, scene + 'poses_bounds.npy'))[:3, :5, :])
        # self.matrixList = np.stack(self.matrixList)
        self.length = len(self.nameList)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        batchDict = {}
        scene, target_idx = self.nameList[ind].split('$')
        training_idx = self.idx_list.copy()
        training_idx.remove(target_idx)
        # random.shuffle(training_idx)
        all_idx = [target_idx, ] + training_idx
        scene = osp.join(self.dataRoot, scene)
        name_list = [scene + '{}_' + a + '.{}' for a in all_idx]
        batchDict['name'] = name_list[0]
        cam_mats = np.load(scene + 'cam_mats.npy')
        # if not cam_mats.shape == (3, 6, 9):
        #     raise Exception(scene, ' cam mat shape error!')

        max_depth = cam_mats[1, -1, int(target_idx) - 1].astype(np.float32)
        target_im = loadHdr(name_list[0].format('im', 'hdr'))
        seg = loadImage(name_list[0].format('immask', 'png'))[0:1, :, :]
        scene_scale = get_hdr_scale(target_im, seg, self.phase)

        segObj_BRDF = (seg > 0.9)
        segObj_SVL = ndimage.binary_erosion(segObj_BRDF.squeeze(), structure=np.ones((7, 7)), border_value=1)[np.newaxis, :, :]
        seg = np.concatenate([segObj_BRDF, segObj_SVL], axis=0).astype(np.float32)
        batchDict['mask'] = seg  # segBRDF

        src_c2w_list = []
        src_int_list = []
        rgb_list = []
        depthest_list = []
        depthgt_list = []
        # conf_list = []
        # depth_norm_list = []
        for name, idx in zip(name_list, all_idx):
            idx = int(idx) - 1
            im = loadHdr(name.format('im', 'hdr'))
            im = np.clip(im * scene_scale, 0, 1.0)
            rgb_list.append(im)
            poses_hwf_bounds = cam_mats[..., idx]
            src_c2w_list.append(_34_to_44(poses_hwf_bounds[:, :4]))
            h, w, f = poses_hwf_bounds[:, -2]
            intrinsic = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=float)
            src_int_list.append(intrinsic)
            # conf_list.append(loadBinary(name.format('conf', 'dat')))
            depthest = loadBinary(name.format('depthest', 'dat'))
            depthest_list.append(depthest)
            if self.cfg.BRDF.gt:
                depthgt = loadBinary(name.format('imdepth', 'dat'))
                depthgt_list.append(depthgt)

        conf = loadBinary(name_list[0].format('conf', 'dat'))
        depth_norm = np.clip(depthest_list[0] / max_depth, 0, 1)

        batchDict['rgb'] = np.stack(rgb_list, 0).astype(np.float32)
        batchDict['depth_est'] = np.stack(depthest_list, 0).astype(np.float32)
        if self.cfg.BRDF.gt:
            batchDict['depth_gt'] = np.stack(depthgt_list, 0).astype(np.float32)
        # batchDict['conf'] = np.stack(conf_list, 0).astype(np.float32)
        batchDict['conf'] = conf.astype(np.float32)
        batchDict['depth_norm'] = depth_norm.astype(np.float32)
        batchDict['cam'] = np.stack(src_int_list, 0).astype(np.float32)
        # recenter to cam_0
        w2target = np.linalg.inv(src_c2w_list[0])
        batchDict['c2w'] = (w2target @ np.stack(src_c2w_list, 0)).astype(np.float32)
        # batchDict['max_depth'] = max_depth.reshape(1, 1, 1, 1)

        # scene scale changes during training time
        # normal_est = loadH5(name_list[0].format('normalest', 'h5'))
        # batchDict['normal'] = normal_est
        # DL = loadH5(name_list[0].format('DLest', 'h5'))
        # batchDict['DL'] = DL

        albedo = loadImage(name_list[0].format('imbaseColor', 'png'))
        batchDict['albedo_gt'] = (albedo ** 2.2).astype(np.float32)
        rough = loadImage(name_list[0].format('imroughness', 'png'))[0:1, :, :].astype(np.float32)
        batchDict['rough_gt'] = rough

        if self.mode == 'SVL' or self.mode == 'finetune':
            # envmaps = loadEnvmap(name_list[0].replace('_320', '').format('imenv', 'hdr'), self.cfg.SVL.env_height,
            #                      self.cfg.SVL.env_width, self.cfg.SVL.env_rows, self.cfg.SVL.env_cols)
            envmaps = loadEnvmap(name_list[0].replace('_320', '_env').format('imenv', 'hdr'), self.cfg.SVL.env_height,
                                 self.cfg.SVL.env_width, self.cfg.SVL.env_rows, self.cfg.SVL.env_cols)
            envmaps = envmaps * scene_scale
            batchDict['envmaps_SVL_gt'] = envmaps.astype(np.float32)
        return batchDict


class Openrooms_FF_single(Dataset):
    def __init__(self, dataRoot, cfg, stage, phase='TRAIN'):
        self.dataRoot = dataRoot
        self.cfg = cfg
        self.mode = cfg.mode
        self.stage = stage
        self.phase = phase

        if phase == 'TRAIN':
            sceneFile = osp.join(dataRoot, 'train.txt')
        elif phase == 'TEST':
            sceneFile = osp.join(dataRoot, 'test.txt')
        else:
            raise Exception('Unrecognized phase for data loader')

        with open(sceneFile, 'r') as fIn:
            sceneList = fIn.readlines()
        sceneList = [a.strip() for a in sceneList]

        idx_list = list(range(1, cfg.num_view_all + 1))
        self.idx_list = list(map(str, idx_list))
        self.nameList = []
        # saving maxdepth array in ram is very slow...
        # maybe saving floating array is difficult for computer.
        self.maxdepthList = np.array([])
        for scene in sceneList:
            cam_mats = np.load(osp.join(dataRoot, scene + 'cam_mats.npy'))
            if not cam_mats.shape == (3, 6, 9):
                raise Exception(scene, ' cam mat shape error!')
            max_depth = cam_mats[1, -1]
            self.maxdepthList = np.concatenate([self.maxdepthList, max_depth])
            # self.nameList += [scene + '{}_' + f'{i}' + '.{}' for i in idx_list]
            self.nameList += [scene + '$' + i for i in self.idx_list]
        # self.nameList = np.array(self.nameList).astype(np.string_)
        self.maxdepthList = self.maxdepthList.astype(np.float32)
        self.length = len(self.nameList)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        batchDict = {}
        scene, target_idx = self.nameList[ind].split('$')
        scene = osp.join(self.dataRoot, scene)
        # cam_mats = np.load(scene + 'cam_mats.npy')
        # max_depth = cam_mats[1, -1, int(target_idx) - 1]
        name = osp.join(scene + '{}_' + target_idx + '.{}')
        max_depth = self.maxdepthList[ind]
        # name = osp.join(self.dataRoot, str(self.nameList[ind], encoding='utf-8'))
        # name = osp.join(self.dataRoot, self.nameList[ind])
        batchDict['name'] = name
        im = loadHdr(name.format('im', 'hdr'))
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
        depthest = loadBinary(name.format('depthest', 'dat'))
        depth_norm = np.clip(depthest / max_depth, 0, 1)
        # depth_norm = loadBinary(name.format('midas', 'dat'))
        batchDict['input'] = np.concatenate([im, depth_norm, conf], axis=0).astype(np.float32)

        # if self.stage == '1-1':
        #     normal = loadImage(name.format('imnormal', 'png'), normalize_01=False)
        #     normal = (normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0, keepdims=True), 1e-5))).astype(np.float32)
        #     batchDict['normal_gt'] = normal
        #
        # elif self.stage == '1-2':
        #     envmaps_DL = loadEnvmap(name.format('imenvDirect', 'hdr'), self.cfg.DL.env_height, self.cfg.DL.env_width,
        #                                      self.cfg.DL.env_rows, self.cfg.DL.env_cols)
        #     envmaps_DL = envmaps_DL * scene_scale
        #     batchDict['envmaps_DL_gt'] = envmaps_DL.astype(np.float32)
        normal = loadImage(name.format('imnormal', 'png'), normalize_01=False)
        normal = (normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0, keepdims=True), 1e-5))).astype(np.float32)
        batchDict['normal_gt'] = normal
        if self.mode == 'DL' or self.mode == 'finetune':
            envmaps_DL = loadEnvmap(name.format('imenvDirect', 'hdr'), self.cfg.DL.env_height, self.cfg.DL.env_width,
                                 self.cfg.DL.env_rows, self.cfg.DL.env_cols)
            envmaps_DL = envmaps_DL * scene_scale
            batchDict['envmaps_DL_gt'] = envmaps_DL.astype(np.float32)
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
