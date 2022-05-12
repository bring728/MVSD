from torch.utils.data import Dataset
from utils import *
import random
from utils_geometry import *
from utils_geometry import _34_to_44
import scipy.ndimage as ndimage
import os.path as osp


class Openrooms_org(Dataset):
    def __init__(self, dataRoot, cfg, phase='TRAIN', debug=True):
        self.phase = phase
        self.env_height = cfg.env_height
        self.env_width = cfg.env_width
        self.env_rows = cfg.env_rows
        self.env_cols = cfg.env_cols

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
        sceneList = [osp.join(dataRoot, a.strip()) for a in sceneList]

        self.imList = []
        for shape in sceneList:
            imNames = sorted(glob.glob(osp.join(shape, 'im_*.hdr')))
            self.imList += imNames

        self.directlightList = [x.replace('im_', 'imenvDirect_') for x in self.imList]
        self.directlightList = [x.replace('DiffMat', '') for x in self.directlightList]

        self.normalList = [x.replace('im_', 'imnormal_').replace('hdr', 'png') for x in self.imList]
        self.normalList = [x.replace('DiffLight', '') for x in self.normalList]

        self.depthList = [x.replace('im_', 'imdepth_').replace('hdr', 'dat') for x in self.imList]
        self.depthList = [x.replace('DiffLight', '') for x in self.depthList]
        self.depthList = [x.replace('DiffMat', '') for x in self.depthList]

        self.segList = [x.replace('im_', 'immask_').replace('hdr', 'png') for x in self.imList]
        self.segList = [x.replace('DiffMat', '') for x in self.segList]

        self.count = len(self.imList)
        print('sample Num: %d' % self.count)

        self.imList = np.array(self.imList).astype(np.string_)
        self.directlightList = np.array(self.directlightList).astype(np.string_)
        self.normalList = np.array(self.normalList).astype(np.string_)
        self.depthList = np.array(self.depthList).astype(np.string_)
        self.segList = np.array(self.segList).astype(np.string_)

    def __len__(self):
        return self.count

    def __getitem__(self, ind):
        name = str(self.imList[ind], encoding='utf-8')

        im = loadHdr(str(self.imList[ind], encoding='utf-8'))
        seg = 0.5 * (loadImage(str(self.segList[ind], encoding='utf-8')) + 1)[0:1, :, :]
        scale = get_hdr_scale(im, seg, self.phase)
        im = np.clip(im * scale, 0, 1.0)

        segArea = np.logical_and(seg > 0.49, seg < 0.51).astype(np.float32)
        # segEnv = (seg < 0.1).astype(np.float32)
        segObj = (seg > 0.9)
        # segObj = ndimage.binary_erosion(segObj.squeeze(), structure=np.ones((7, 7)), border_value=1)[np.newaxis, :, :]
        # segObj = segObj.astype(np.float32)
        target_seg = np.concatenate([segObj, segArea + segObj], axis=0)

        # normalize the normal vector so that it will be unit length
        normal = loadImage(str(self.normalList[ind], encoding='utf-8'))
        normal = (normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5))[np.newaxis, :]).astype(np.float32)

        depth = loadBinary(str(self.depthList[ind], encoding='utf-8'))
        depth = depth / depth.max()

        envmaps, envmapsInd = loadEnvmap(str(self.directlightList[ind], encoding='utf-8'), self.env_height, self.env_width, self.env_rows,
                                         self.env_cols)
        envmaps = envmaps * scale

        rgbd = np.concatenate([im, depth], axis=0).astype(np.float32)
        batchDict = {'rgbd': rgbd,
                     'seg': target_seg,
                     'normal': normal,
                     'envmaps': envmaps.astype(np.float32),
                     'envmapsInd': envmapsInd,
                     'name': name,
                     }
        return batchDict


class Openrooms_FF(Dataset):
    def __init__(self, dataRoot, cfg, phase='TRAIN', debug=True):
        self.num_view_min = cfg.num_view_min
        self.range = cfg.num_view_max - cfg.num_view_min + 1
        self.num_view_all = cfg.num_view_all
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
        sceneList = [osp.join(dataRoot, a.strip()) for a in sceneList]

        self.viewList = []
        self.matrixList = []
        self.albedoList = []
        self.normalList = []
        self.roughList = []
        self.depthmvsList = []
        self.confList = []
        self.depthGTList = []
        self.segList = []

        # bds = np.load(osp.join(sceneList[0], 'poses_bounds.npy'))[:, -2:].transpose([1, 0])
        # self.near_depth = bds.min() * 0.9
        # self.far_depth = bds.max() * 2
        # self.plane_depths = np.array(InterpolateDepths(self.near_depth, self.far_depth, self.num_planes))

        idx_list = list(range(self.num_view_all))
        for scene in sceneList:
            poses = np.load(scene + 'poses_bounds.npy').transpose([1, 2, 0])
            # poses = poses_arr[:, :-2].reshape([-1, 3, 5])
            self.viewList.append([f'{scene}im_{i + 1}.rgbe' for i in idx_list])
            self.depthmvsList.append([f'{scene}depthest_{i + 1}.dat' for i in idx_list])
            self.confList.append([f'{scene}conf_{i + 1}.dat' for i in idx_list])
            self.albedoList.append([f'{scene}imbaseColor_{i + 1}.png' for i in idx_list])
            self.normalList.append([f'{scene}imnormal_{i + 1}.png' for i in idx_list])
            self.roughList.append([f'{scene}imroughness_{i + 1}.png' for i in idx_list])
            self.depthGTList.append([f'{scene}imdepth_{i + 1}.dat' for i in idx_list])
            self.segList.append([f'{scene}immask_{i + 1}.png' for i in idx_list])
            self.matrixList.append(poses)

        self.count = len(self.viewList)
        print('sample Num: %d' % self.count)

        self.viewList = np.array(self.viewList).astype(np.string_)
        self.albedoList = np.array(self.albedoList).astype(np.string_)
        self.normalList = np.array(self.normalList).astype(np.string_)
        self.roughList = np.array(self.roughList).astype(np.string_)
        self.depthmvsList = np.array(self.depthmvsList).astype(np.string_)
        self.confList = np.array(self.confList).astype(np.string_)
        self.depthGTList = np.array(self.depthGTList).astype(np.string_)
        self.segList = np.array(self.segList).astype(np.string_)
        self.matrixList = np.stack(self.matrixList)

    def __len__(self):
        return self.count

    def __getitem__(self, ind):
        name = str(self.viewList[ind, 0], encoding='utf-8').split('rendering')[1]
        num_view = self.num_view_min + np.random.choice(self.range, 1)[0]
        training_pair_idx = np.random.choice(self.num_view_all, num_view, replace=False)
        target_idx = training_pair_idx[0]

        target_im = loadHdr(str(self.viewList[ind, target_idx], encoding='utf-8'))
        seg = 0.5 * (loadImage(str(self.segList[ind, target_idx], encoding='utf-8')) + 1)[0:1, :, :]
        scale = get_hdr_scale(target_im, seg, self.phase)

        target_segArea = np.logical_and(seg > 0.49, seg < 0.51)
        target_segEnv = (seg < 0.1)
        target_segObj = (seg > 0.9)
        target_seg = np.concatenate([target_segObj, target_segArea + target_segObj], axis=0)

        # Read albedo
        albedo = loadImage(str(self.albedoList[ind, target_idx], encoding='utf-8'), isGama=False)
        target_albedo = (0.5 * (albedo + 1)) ** 2.2

        # normalize the normal vector so that it will be unit length
        normal = loadImage(str(self.normalList[ind, target_idx], encoding='utf-8'))
        target_normal = (normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5))[np.newaxis, :])

        target_rough = loadImage(str(self.roughList[ind, target_idx], encoding='utf-8'))[0:1, :, :]

        src_im_list = []
        src_c2w_list = []
        src_int_list = []
        src_depth_list = []
        for idx in training_pair_idx:
            # Read Image
            im = loadHdr(str(self.viewList[ind, idx], encoding='utf-8'))
            im = np.clip(im * scale, 0, 1.0)
            # im = np.repeat(im[:, None, ...], self.num_planes, axis=1)
            src_im_list.append(im)
            # Read Poses
            poses = self.matrixList[ind, ..., idx]
            src_c2w_list.append(_34_to_44(poses[:, :-1]))
            h, w, f = poses[:, -1]
            intrinsic = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=float)
            src_int_list.append(intrinsic)
            src_depth_list.append(loadBinary(str(self.depthList[ind, idx], encoding='utf-8')))

        src_int_list = np.stack(src_int_list)
        src_c2w_list = np.stack(src_c2w_list)

        # recenter to cam_0
        w2target = np.linalg.inv(src_c2w_list[0])
        src_c2w_list = w2target @ src_c2w_list

        target_gt = np.concatenate([target_albedo.astype(np.float32), target_normal.astype(np.float32), target_rough.astype(np.float32),
                                    src_depth_list[0].astype(np.float32),
                                    target_seg.astype(np.float32)], axis=0)
        batchDict = {'target_gt': target_gt,
                     'depth_list': np.stack(src_depth_list, 0).astype(np.float32),
                     'im_list': np.stack(src_im_list, 0).astype(np.float32),
                     'int_list': src_int_list.astype(np.float32),
                     'c2w_list': src_c2w_list.astype(np.float32),
                     'shape': np.array([h, w], dtype=np.int),
                     'name': name,
                     'idx': training_pair_idx,
                     }
        return batchDict


class Openrooms_FF_single(Dataset):
    def __init__(self, dataRoot, cfg, phase='TRAIN', debug=True):
        self.dataRoot = dataRoot
        self.cfg = cfg
        self.phase = phase
        self.prob_threshold = 0.8

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

        self.count = len(self.nameList)
        print('sample Num: %d' % self.count)

    def __len__(self):
        return self.count

    def __getitem__(self, ind):
        name = osp.join(self.dataRoot, self.nameList[ind])
        im = loadHdr(name.format('im', 'rgbe'))
        seg = 0.5 * (loadImage(name.format('immask', 'png')) + 1)[0:1, :, :]
        scale = get_hdr_scale(im, seg, self.phase)
        im = np.clip(im * scale, 0, 1.0)

        if self.cfg.is_Light:
            segArea = np.logical_and(seg > 0.49, seg < 0.51).astype(np.float32)
            segObj = (seg > 0.9)
            segObj = ndimage.binary_erosion(segObj.squeeze(), structure=np.ones((7, 7)), border_value=1)[np.newaxis, :, :]
            segObj = segObj.astype(np.float32)
            seg = np.concatenate([segObj, segArea + segObj], axis=0)  # segBRDF, segAll
        else:
            segArea = np.logical_and(seg > 0.49, seg < 0.51).astype(np.float32)
            segObj = (seg > 0.9)
            seg = np.concatenate([segObj, segArea + segObj], axis=0)  # segBRDF, segAll

        albedo = loadImage(name.format('imbaseColor', 'png'), isGama=False)
        albedo = ((0.5 * (albedo + 1)) ** 2.2).astype(np.float32)
        normal = loadImage(name.format('imnormal', 'png'))
        normal = (normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5))[np.newaxis, :]).astype(np.float32)
        rough = loadImage(name.format('imroughness', 'png'))[0:1, :, :].astype(np.float32)

        envmaps, envmapsInd = loadEnvmap(name.format('imenvDirect', 'hdr'), self.cfg.env_height, self.cfg.env_width, self.cfg.env_rows, self.cfg.env_cols)
        envmaps = envmaps * scale

        if self.cfg.depth_type == 'mvs':
            depthmvs = loadBinary(name.format('depthest', 'dat'))
            conf = loadBinary(name.format('conf', 'dat'))
            depth_filtered = depthmvs.copy()
            mask = (conf > self.prob_threshold)
            depth_filtered[~mask] = 0
            depth_max = np.percentile(depth_filtered, 96)
            depthmvs_normalized = np.clip(depthmvs / depth_max, 0, 1)
            input_data = np.concatenate([im, depthmvs_normalized, conf], axis=0).astype(np.float32)

            depthGT = loadBinary(name.format('imdepth', 'dat'))
            depthGT_normalized = np.clip((depthGT / depth_max), 0, 1).astype(np.float32)
            # depth_err = np.mean(np.abs(depthGT_normalized - depthmvs_normalized))
            # cv2.hconcat([depthGT_normalized.transpose([1,2,0]), depthmvs_normalized.transpose([1,2,0])])
        elif self.cfg.depth_type == 'midas':
            depthmidas = loadBinary(name.format('midas', 'dat'))
            depthmidas_normalized = depthmidas / np.percentile(depthmidas, 96)
            depthmidas_normalized = np.clip(depthmidas_normalized, 0, 1)
            input_data = np.concatenate([im, depthmidas_normalized], axis=0).astype(np.float32)
        else:
            raise Exception('depth type error')


        batchDict = {'input': input_data,
                     'normal_gt': normal,
                     'depth_gt': depthGT_normalized,
                     'seg': seg,
                     'name': name,
                     'envmaps_gt': envmaps.astype(np.float32),
                     'envmapsInd': envmapsInd,
                     # 'depth_err': depth_err,
                     }
        return batchDict
