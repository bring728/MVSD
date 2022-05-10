import os
import shutil
import traceback
import logging
import os.path as osp
from multiprocessing import Pool, Manager
import time
from subprocess import check_output, STDOUT, CalledProcessError
import glob
import random

def outdir2xml(scene):
    a = scene.split('data_FF_10_640')
    b = a[1].split('/')
    scene_name = b[2]
    return a[0] + 'scenes/' + b[1].split('_')[1] + '/' + scene_name + '/' + b[1].split('_')[0] + '_FF.xml'

def xml2camtxt(xml, k):
    scene_type = xml.split('/')[-1].split('_')[0]
    return f'{osp.dirname(xml)}/{k}_cam_{scene_type}_FF.txt'


renderer_directlight = '/Lib/OptixRendererDirectLight/src/bin/optixRenderer'
num_p = 2


def work(xml, camtxt, outdir, k, num_pushed, q):
    gpu_id = q.get()
    start = time.time()
    try:
        print(f'{outdir} - {k} is start on gpu {gpu_id}')
        optixrenderer_arg = [renderer_directlight, '-f', xml, '-c', camtxt, '-o', osp.join(outdir, f'{k}_im'), '-m', '7', '--imWidth', '40',
                             '--imHeight', '30', '--maxPathLength', '2 ', '--gpuIds', str(gpu_id), '--forceOutput']
        output = check_output(optixrenderer_arg, stderr=STDOUT, universal_newlines=True)
    except Exception as e:
        logging.error(traceback.format_exc())
    finally:
        q.put(gpu_id)
        logger.debug(f'{camtxt} - is done, count : {num_pushed}, {time.time() - start}')
        return


logger = logging.getLogger()
if __name__ == "__main__":
    m = Manager()
    q = m.Queue()
    for gpu in range(num_p):
        q.put(gpu)
    pool = Pool(processes=num_p)

    root = '/Data/OpenRoomsDataset/data/rendering/data_FF_10_640'
    renderer_directlight = '/Lib/OptixRendererDirectLight/src/bin/optixRenderer'

    f = open('train.txt', 'r')
    outdir = f.readlines()
    f.close()

    outdir_list = [osp.join(root, a.strip()) for a in outdir]
    xml_list = [outdir2xml(a) for a in outdir_list]
    num_pushed = 0
    for outdir, xml in zip(outdir_list, xml_list):
        num_pushed += 1
        k = osp.basename(outdir).split('_')[0]
        outdir = osp.dirname(outdir)
        os.makedirs(outdir, exist_ok=True)
        cam_file_name = xml2camtxt(xml, k)
        pool.apply_async(work, (xml, cam_file_name, outdir, k, num_pushed, q))

    pool.close()
    pool.join()
