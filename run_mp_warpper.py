import torch.distributed as dist
import torch.multiprocessing as mp
from train_stage_1_1 import train
import os

def init_process(rank, size, config, debug, phase, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, config, debug, phase)


config_files = ['stage1-1_0.yml', 'stage1-1_1.yml']
num_gpu = 2

if __name__ == "__main__":
    mp.set_start_method('spawn')

    processes = []
    for rank in range(num_gpu):
        p = mp.Process(target=init_process, args=(rank, num_gpu, config_files[rank], False, 'TRAIN', train))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # processes = []
    # for rank in range(num_gpu):
    #     p = mp.Process(target=init_process, args=(rank, num_gpu, config_files[rank + num_gpu], train))
    #     p.start()
    #     processes.append(p)
    #
    # for p in processes:
    #     p.join()