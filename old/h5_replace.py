import os.path as osp
import scipy.ndimage as ndimage
import glob
from utils import *
from tqdm import tqdm

stage = '1-1'
dataRoot = '/new_disk/happily/Data/OpenRooms_FF/'
train_out_root = f'/new_disk/happily/Data/OpenRooms_h5/{stage}/train'
test_out_root = f'/new_disk/happily/Data/OpenRooms_h5/{stage}/test'


def replace_func():
    h5List = sorted(glob.glob('/new_disk/happily/Data/OpenRooms_h5/stage-1/train/*'))

    for h5_name in tqdm(h5List):
        hf = h5py.File(h5_name, 'a')
        name = hf.attrs['name']
        name = osp.join(dataRoot, name).split(',')[0]
        seg = loadImage(name.format('immask', 'png'))[0:1, :, :]
        segArea = np.logical_and(seg > 0.49, seg < 0.51)
        segObj = (seg > 0.9)
        segAll = segArea + segObj
        segBRDF = ndimage.binary_erosion(segObj.squeeze(), structure=np.ones((7, 7)), border_value=1)[np.newaxis, :, :]
        seg = np.concatenate([segBRDF, segAll], axis=0).astype(np.float32)  # segBRDF, segAll
        del hf['mask']
        hf.create_dataset('mask', data=seg[None], compression='lzf')
        envmaps, envmapsInd = loadEnvmap(name.format('imenvDirect', 'hdr'), 8, 16, 30, 40)
        if 'DL_gt' in hf.keys():
            del hf['DL_gt']
        hf.create_dataset('DL_gt', data=envmaps[None], compression='lzf')
        if 'DL_ind' in hf.keys():
            del hf['DL_ind']
        hf.create_dataset('DL_ind', data=envmapsInd[None], compression='lzf')
        hf.close()


if __name__ == "__main__":
    replace_func()
