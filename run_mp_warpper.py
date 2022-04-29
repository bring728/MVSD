import torch.distributed as dist
import torch.multiprocessing as mp
from train_stage_1_single import train
import os

def init_process(rank, size, config, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, config)


config_files = ['stage1_0.yml', 'stage1_2.yml']

if __name__ == "__main__":
    mp.set_start_method('spawn')
    size = len(config_files)
    processes = []
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, config_files[rank], train))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()