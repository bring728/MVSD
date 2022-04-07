from torch.utils.data import Dataset
from utils import loadHdr, loadImage, loadBinary, get_hdr_scale
import random
from utils_geometry import *
from utils_geometry import _34_to_44

training_group = [[0, 2, 3, 5], [3, 5, 6, 8], [0, 1, 6, 7], [1, 2, 7, 8], [0, 2, 6, 7], [0, 1, 6, 8], [1, 2, 6, 8], [0, 2, 7, 8], [0, 2, 6, 8],
                  [1, 3, 5, 7],
                  [0, 2, 5, 6], [0, 5, 6, 8], [0, 2, 3, 8], [2, 3, 6, 8], [0, 5, 6, 7], [0, 1, 5, 6], [2, 3, 7, 8], [1, 2, 3, 8], [0, 2, 3, 4],
                  [0, 2, 4, 5],
                  [1, 2, 3, 5], [0, 1, 3, 5], [3, 4, 6, 8], [3, 5, 6, 7], [3, 5, 7, 8], [4, 5, 6, 8], [0, 1, 4, 6], [0, 1, 3, 7], [1, 3, 6, 7],
                  [0, 4, 6, 7],
                  [1, 2, 4, 8], [1, 5, 7, 8], [1, 2, 5, 7], [2, 4, 7, 8], [0, 3, 7, 8], [0, 1, 5, 8], [1, 2, 3, 6], [2, 5, 6, 7]]

target_group = [[1, 4], [4, 7], [3, 4], [4, 5], [1, 3, 4], [3, 4, 7], [4, 5, 7], [1, 4, 5], [1, 3, 4, 5, 7], [4], [1, 3, 4], [3, 4, 7], [1, 4, 5],
                [4, 5, 7], [3, 4], [3, 4], [4, 5], [4, 5], [1], [1], [4], [4], [7], [4], [4], [7], [3], [4], [4], [3], [5], [4], [4], [5], [4], [4],
                [4], [4]]


class Openrooms_dataset(Dataset):
    def __init__(self, dataRoot, num_view_all=9, num_planes=24, phase='DEBUG', rseed=None):

        self.num_view_all = num_view_all
        self.num_planes = num_planes
        self.phase = phase.upper()

        if phase.upper() == 'TRAIN':
            sceneFile = osp.join(dataRoot, 'train.txt')
        elif phase.upper() == 'TEST':
            sceneFile = osp.join(dataRoot, 'test.txt')
        elif phase.upper() == 'DEBUG':
            sceneFile = osp.join(dataRoot, 'debug.txt')
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

        self.num_group = len(training_group)

    def __len__(self):
        return self.count

    def __getitem__(self, ind):
        name = str(self.viewList[ind, 0], encoding='utf-8').split('rendering')[1]

        group_idx = random.randrange(self.num_group)
        training_pair_idx = training_group[group_idx]
        target_idx = random.choice(target_group[group_idx])
        
        target_im = loadHdr(str(self.viewList[ind, target_idx], encoding='utf-8'))
        seg = 0.5 * (loadImage(str(self.segList[ind, target_idx], encoding='utf-8')) + 1)[0:1, :, :]
        scale = get_hdr_scale(target_im, seg, self.phase)
        target_im = np.clip(target_im * scale, 0, 1.0)

        target_segArea = np.logical_and(seg > 0.49, seg < 0.51)
        target_segEnv = (seg < 0.1)
        target_segObj = (seg > 0.9)
        target_seg = np.concatenate([target_segArea, target_segEnv, target_segObj], axis=0)

        # Read albedo
        albedo = loadImage(str(self.albedoList[ind, target_idx], encoding='utf-8'), isGama=False)
        target_albedo = (0.5 * (albedo + 1)) ** 2.2

        # normalize the normal vector so that it will be unit length
        normal = loadImage(str(self.normalList[ind, target_idx], encoding='utf-8'))
        target_normal = (normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5))[np.newaxis, :])

        target_rough = loadImage(str(self.roughList[ind, target_idx], encoding='utf-8'))[0:1, :, :]
        target_depth = loadBinary(str(self.depthList[ind, target_idx], encoding='utf-8'))

        im_list = []
        src_c2w_list = []
        src_int_list = []
        # Read Poses
        pose = self.matrixList[ind, ..., target_idx]
        target_c2w = _34_to_44(pose[:, :-1])
        h, w, f = pose[:, -1]
        target_int = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=float)
        for idx in training_pair_idx:
            # Read Image
            im = loadHdr(str(self.viewList[ind, idx], encoding='utf-8'))
            im = np.clip(im * scale, 0, 1.0)
            # im = np.repeat(im[:, None, ...], self.num_planes, axis=1)
            im_list.append(im)
            # Read Poses
            poses = self.matrixList[ind, ..., idx]
            src_c2w_list.append(_34_to_44(poses[:, :-1]))
            h, w, f = poses[:, -1]
            intrinsic = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=float)
            src_int_list.append(intrinsic)

        src_int_list = np.stack(src_int_list)
        src_c2w_list = np.stack(src_c2w_list)
        
        int_list = np.concatenate([src_int_list, target_int[None, ...]], 0)
        c2w_list = np.concatenate([src_c2w_list, target_c2w[None, ...]], 0)
        c2w_list = recenter(c2w_list) # ref_c2w is Identity
        w2c_list = np.linalg.inv(c2w_list)
        # ref_c2w = make_reference_view(src_c2w_list)
        cx, cy, f = src_int_list[0, 0, -1], src_int_list[0, 1, -1], (src_int_list[0, 0, 0])
        tan_vh = [(cy / f), (cx / f)]

        # CreateDepthPlaneHomo(ref_c2w, tan_vh, [h, w], self.plane_depths, src_int_list, src_c2w_list)
        DPC_inv_ints, DPC_p2ws = CreateDepthPlaneCameras_2(tan_vh, [h, w], self.plane_depths)
        
        # H_tmp = getH_parallel(DPC_inv_ints, DPC_p2ws, tmp_int_list, tmp_c2w_list)
        # Hk_inv_list = H_tmp[:-1, ...]
        # Ht = np.linalg.inv(H_tmp[-1, ...])

        # normal vector rotation
        target2ref = (c2w_list[-1, :3, :3])[None, None, :, :]
        target_normal = np.transpose((target2ref @ np.transpose(target_normal, [1, 2, 0])[..., None])[..., 0], [2, 0, 1])

        batchDict = {'albedo': target_albedo.astype(np.float32),
                     'normal': target_normal.astype(np.float32),
                     'rough': target_rough.astype(np.float32),
                     'depth': target_depth.astype(np.float32),
                     'seg': target_seg.astype(np.float32),
                    #  'segArea': target_segArea.astype(np.float32),
                    #  'segEnv': target_segEnv.astype(np.float32),
                    #  'segObj': target_segObj.astype(np.float32),
                     'im': np.stack(im_list, 0).astype(np.float32),
                     'im_target': target_im.astype(np.float32),
                     'DPC_inv_ints': DPC_inv_ints.astype(np.float32),
                     'DPC_p2ws': DPC_p2ws.astype(np.float32),
                     'int_list': int_list.astype(np.float32),
                     'w2c_list': w2c_list.astype(np.float32),
                     'target_c2w': target_c2w.astype(np.float32),
                     'plane_depth': self.plane_depths[:, None, None].astype(np.float32),
                     'shape': np.array([h, w], dtype=np.int),
                     'name': name,
                     'idx': training_pair_idx + [target_idx],
                     }
        return batchDict