import os.path as osp
import time
import pickle
from cfgnode import CfgNode
import yaml
from mvsd_dataset import Openrooms_FF, Openrooms_FF_single
import os
import lmdb
from torch.utils.data import DataLoader

root = '/new_disk/happily/Data'
write_frequency = 1000
lmdb_path = '/new_disk/happily/Data/OpenRooms_lmdb_tmp'


# def dumps_pyarrow(obj):
#     """
#     Serialize an object.
#     Returns:
#         Implementation-dependent bytes-like object
#     """
#     return pa.serialize(obj).to_buffer()

def dumps_pickle(obj):
    return pickle.dumps(obj)

def openrooms_to_lmdb(config, phase='TRAIN'):
    stage = config.split('stage')[1].split('_')[0]
    with open(os.getcwd() + '/config/' + config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    worker_per_gpu = 16
    batch_per_gpu = 1
    dataRoot = osp.join(root, 'OpenRooms_FF')

    if stage.startswith('1'):
        dataset = Openrooms_FF_single(dataRoot, cfg, stage, phase, False)
    else:
        dataset = Openrooms_FF(dataRoot, cfg, stage, phase, False)
    data_loader = DataLoader(dataset, batch_size=batch_per_gpu, shuffle=False, num_workers=worker_per_gpu, collate_fn=lambda x: x)

    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        data = data[0]
        rgb, mask, conf, normmax, normzero, midas, normal_gt, envmaps_gt, envmapsInd, name = data['rgb'], data['mask'], data['conf'], \
                                                                                       data['depthnormmax'], data['depthnormzero'], \
                                                                                       data['midas'], data['normal_gt'], \
                                                                                       data['envmaps_gt'], data['envmapsInd'], data['name']
        # start = time.time()
        pack = dumps_pickle((rgb, mask, conf, normmax, normzero, midas, normal_gt, envmaps_gt, envmapsInd, name))
        # print(time.time() - start)
        # pack = dumps_pickle5((rgb, mask, conf, normmax, normzero, midas, normal_gt, envmaps_gt, envmapsInd, name))
        txn.put(u'{}'.format(idx).encode('ascii'), pack)
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pickle(keys))
        txn.put(b'__len__', dumps_pickle(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    config = 'stage1-1_0.yml'
    phase = 'TRAIN'
    openrooms_to_lmdb(phase='TRAIN', config=config)
