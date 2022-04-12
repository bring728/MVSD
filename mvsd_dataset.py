from torch.utils.data import Dataset
from utils import loadHdr, loadImage, loadBinary, get_hdr_scale
import random
from utils_geometry import *
from utils_geometry import _34_to_44


class Openrooms_dataset(Dataset):
    def __init__(self, dataRoot, num_view_min, num_view_max, num_view_all=9, phase='TRAIN', debug=True):
        self.num_view_min = num_view_min
        self.range = num_view_max - num_view_min + 1
        self.num_view_all = num_view_all
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
        sceneList = [dataRoot + a.strip() for a in sceneList]

        self.viewList = []
        self.matrixList = []
        self.albedoList = []
        self.normalList = []
        self.roughList = []
        self.depthList = []
        self.segList = []

        # bds = np.load(osp.join(sceneList[0], 'poses_bounds.npy'))[:, -2:].transpose([1, 0])
        # self.near_depth = bds.min() * 0.9
        # self.far_depth = bds.max() * 2
        # self.plane_depths = np.array(InterpolateDepths(self.near_depth, self.far_depth, self.num_planes))

        idx_list = list(range(num_view_all))
        for scene in sceneList:
            poses_arr = np.load(osp.join(scene, 'poses_bounds.npy'))
            poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
            self.viewList.append([f'{scene}/im_{i + 1}.rgbe' for i in idx_list])
            self.albedoList.append([f'{scene}/imbaseColor_{i + 1}.png' for i in idx_list])
            self.normalList.append([f'{scene}/imnormal_{i + 1}.png' for i in idx_list])
            self.roughList.append([f'{scene}/imroughness_{i + 1}.png' for i in idx_list])
            self.depthList.append([f'{scene}/imdepth_{i + 1}.dat' for i in idx_list])
            self.segList.append([f'{scene}/immask_{i + 1}.png' for i in idx_list])
            self.matrixList.append(poses)

        self.count = len(self.viewList)
        print('group Num: %d' % self.count)

        self.viewList = np.array(self.viewList).astype(np.string_)
        self.albedoList = np.array(self.albedoList).astype(np.string_)
        self.normalList = np.array(self.normalList).astype(np.string_)
        self.roughList = np.array(self.roughList).astype(np.string_)
        self.depthList = np.array(self.depthList).astype(np.string_)
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

        target_gt = np.concatenate([target_albedo.astype(np.float32), target_normal.astype(np.float32), target_rough.astype(np.float32), src_depth_list[0].astype(np.float32),
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
