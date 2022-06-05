from forward import model_forward
import os
from utils import *
from mvsd_dataset import Openrooms_FF_single_offline
from tqdm import tqdm
import threading
from torch.utils.data import DataLoader

stage = '1'
dataRoot = '/new_disk/happily/Data/OpenRooms_FF/'
train_out_root = f'/new_disk/happily/Data/OpenRooms_h5/stage-{stage}/train'
test_out_root = f'/new_disk/happily/Data/OpenRooms_h5/stage-{stage}/test'

if cfg.h5:
    for k, v in data.items():
        if isinstance(data[k], torch.Tensor):
            data_shape = data[k].shape
            data[k] = data[k].reshape(-1, data_shape[-3], data_shape[-2], data_shape[-1])

def preprocess_stage_func(gpu, num_gpu, is_DDP):
    if is_DDP:
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:2958', world_size=num_gpu, rank=gpu)
        torch.cuda.set_device(gpu)

    os.makedirs(train_out_root, exist_ok=True)
    os.makedirs(test_out_root, exist_ok=True)
    train_dataset = Openrooms_FF_single_offline(dataRoot, stage, 'TRAIN')
    test_dataset = Openrooms_FF_single_offline(dataRoot, stage, 'TEST')
    train_sampler = None
    if is_DDP:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    is_shuffle = not is_DDP
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=is_shuffle, num_workers=3, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=3, shuffle=False)
    print(gpu, len(train_loader), len(test_loader))
    if stage == '1-1':
        data_key = ['hdr', 'mask', 'dc', 'normal_gt', 'max_intensity', 'DL_gt', 'DL_ind']
    elif stage == '1-2':
        data_key = ['hdr', 'mask', 'dc', 'normal_gt']

    if gpu == 0:
        for i, all_data in enumerate(tqdm(test_loader)):
            h5_name = osp.join(test_out_root, f'{i:06}.h5')
            hf = h5py.File(h5_name, 'w')
            name = all_data['name']
            name = ''.join([n.split('OpenRooms_FF/')[1] + ',' for n in name])
            hf.attrs['name'] = name
            for k in data_key:
                v = all_data[k].detach().cpu().numpy()
                hf.create_dataset(k, data=v, compression='lzf')
            hf.close()

    prefix = gpu * len(train_loader)
    for i, all_data in enumerate(tqdm(train_loader)):
        h5_name = osp.join(train_out_root, f'{prefix + i:06}.h5')
        hf = h5py.File(h5_name, 'w')
        name = all_data['name']
        name = ''.join([n.split('OpenRooms_FF/')[1] + ',' for n in name])
        hf.attrs['name'] = name
        for k in data_key:
            v = all_data[k].detach().cpu().numpy()
            hf.create_dataset(k, data=v, compression='lzf')
        hf.close()
        # loadH5_stage(h5_name, stage)

    # scalars_to_log = {}
    # for all_data in tqdm(all_loader):
    #     all_data = tocuda(all_data, gpu, cfg.pinned, not stage.startswith('1'))
    #     with torch.no_grad():
    #         total_loss, pred = model_forward(stage, phase, curr_model, helper_dict, all_data, cfg, scalars_to_log, False)
    #         name = all_data['name']
    #         normal_name = [n.format('normalest', 'h5') for n in name]
    #         dl_name = [n.format('DLest', 'h5') for n in name]
    #         normal_pred = pred['normal']
    #         dl_pred = pred['DL']
    #         threads = [threading.Thread(target=th_save_h5, args=(name, img,)) for name, img in zip(normal_name, normal_pred)]
    #         for thread in threads:
    #             thread.start()
    #         for thread in threads:
    #             thread.join()
    #
    #         threads = [threading.Thread(target=th_save_h5, args=(name, img,)) for name, img in zip(dl_name, dl_pred)]
    #         for thread in threads:
    #             thread.start()
    #         for thread in threads:
    #             thread.join()

    if is_DDP:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    preprocess_stage_func(gpu=0, num_gpu=1, is_DDP=False)
