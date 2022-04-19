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
from xml.etree.ElementTree import ElementTree, parse


def work(asset):
    gpu_id = queue.get()
    try:
        ident = current_process().ident
        renderer = '/Lib/OptixRendererLight/src/bin/optixRenderer'

        xml, cam, outdir = asset.split(' ')
        outdir_name = outdir.split('data_FF_10_640')[1]

        if len(glob.glob(outdir + '*.rgbe')) == 9:
            print(f'{outdir_name} is already rendered')
        else:
            tree = parse(xml)
            root = tree.getroot()
            root.find('sensor').find('sampler').find('integer').attrib['value'] = '2048'
            xml_tmp = outdir + 'tmp.xml'
            ElementTree(root).write(xml_tmp)

            # print(f'{outdir_name} start! {ident}: starting process on GPU {gpu_id}')
            # optixrenderer_arg = [renderer, '-f', xml, '-c', cam, '-o', outdir + 'im', '-m', '7', '--gpuIds', str(gpu_id), '--forceOutput']
            # start = time.time()
            # output = check_output(optixrenderer_arg, stderr=STDOUT, universal_newlines=True)
            # print(f'{outdir_name} is done_{time.time() - start:.2f}, {ident}: finished')
            # os.remove(xml_tmp)
    finally:
        queue.put(gpu_id)
        return


queue = Queue()

if __name__ == "__main__":
    num_p = 1
    for gpu_ids in range(num_p):
        queue.put(gpu_ids)

    txt = open('noisy_good_scenes.txt', 'r')
    a = txt.readlines()
    txt.close()

    a = [b.strip() for b in a]
    xmlFiles = a[::3]
    camFiles = a[1::3]
    outdirs = a[2::3]

    asset = [z + ' ' + x + ' ' + c for z, x, c in zip(xmlFiles, camFiles, outdirs)][:1]

    total = len(asset)
    p = Pool(num_p)
    with tqdm(total=total) as pbar:
        for ret in tqdm(p.imap_unordered(work, asset)):
            pbar.update()

    p.close()
    p.join()






