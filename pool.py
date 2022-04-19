import os
import shutil
import logging
from tqdm import tqdm
import sys
import os.path as osp
from multiprocessing import Pool, Queue, current_process
import time
from subprocess import check_output, STDOUT, CalledProcessError
import glob



def work(asset):
    gpu_id = queue.get()
    try:
        ident = current_process().ident
        renderer = '/Lib/OptixRendererLight/src/bin/optixRenderer'

        xml, cam, outdir = asset.split(' ')
        outdir_name = outdir.split('data_FF_10_640')[1]
        if len(glob.glob(outdir + '*')) != 65:
            print(f'{outdir_name} is wrong')
            return

        print(f'{outdir_name} start! {ident}: starting process on GPU {gpu_id}')
        optixrenderer_arg = [renderer, '-f', xml, '-c', cam, '-o', outdir + 'im', '-m', '7', '--gpuIds', str(gpu_id),
                             '--forceOutput']
        start = time.time()
        try:
            output = check_output(optixrenderer_arg, stderr=STDOUT, universal_newlines=True)
        except CalledProcessError as exc:
            print(exc.returncode)
            print(exc.output)
        # output = check_output(optixrenderer_arg, stderr=STDOUT, universal_newlines=True)
        print(f'{outdir_name} is done_{time.time() - start:.2f}, {ident}: finished')
    finally:
        queue.put(gpu_id)
    return


queue = Queue()

if __name__ == "__main__":
    num_p = 8
    for gpu_ids in range(num_p):
        queue.put(gpu_ids)

    txt = open('good_scenes.txt', 'r')
    a = txt.readlines()
    txt.close()

    a = [b.strip() for b in a]
    xmlFiles = a[::3]
    camFiles = a[1::3]
    outdirs = a[2::3]

    asset = [z + ' ' + x + ' ' + c for z, x, c in zip(xmlFiles, camFiles, outdirs)]

    total = len(asset)
    p = Pool(num_p)
    with tqdm(total=total) as pbar:
        for ret in tqdm(p.imap_unordered(work, asset)):
            pbar.update()

    p.close()
    p.join()






