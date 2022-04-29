import os
import shutil
import traceback
import logging
import os.path as osp
from multiprocessing import Pool, Manager
import time
from subprocess import check_output, STDOUT, CalledProcessError
import glob
from termcolor import colored


def scene2xml(scene):
    a = scene.split('data_FF_10_640')
    b = a[1].split('/')
    scene_name = b[2]
    return a[0] + 'scenes/' + b[1].split('_')[1] + '/' + scene_name + '/' + b[1].split('_')[0] + '_FF.xml'


def xml2camtxt(xml, k):
    scene_type = xml.split('/')[-1].split('_')[0]
    return f'{os.path.dirname(xml)}/{k}_cam_{scene_type}_FF.txt'


renderer = '/Lib/OptixRenderer-master/build_650/bin/optixRenderer'
renderer_light = '/Lib/OptixRendererLight/src/bin/optixRenderer'
repository_root = '/Data/optix_output/'
# dataroot = '/home/vig-titan2/Data/@@FF_renew/rendering/data_FF_10_640'
dataroot = '/Data/OpenRoomsDataset/data/rendering/data_FF_10_640'
num_p = 8
mode = ['1', '2', '3', '4', '5', '0']
# mode = ['1', '2',]


def work(xml, camtxt, outdir, k, num_pushed, q):
    gpu_id = q.get()
    try:
        start = time.time()
        for m in mode:
            optixrenderer_arg = [renderer, '-f', xml, '-c', camtxt, '-o', osp.join(outdir, f'{k}_im'), '-m', m, '--gpuIds', str(gpu_id),
                                 '--forceOutput', '--maxIteration', '4', '--medianFilter']
            output = check_output(optixrenderer_arg, stderr=STDOUT, universal_newlines=True)

        optixrenderer_arg = [renderer_light, '-f', xml, '-c', camtxt, '-o', osp.join(outdir, f'{k}_im'), '-m', '7', '--gpuIds', str(gpu_id),
                             '--forceOutput']
        output = check_output(optixrenderer_arg, stderr=STDOUT, universal_newlines=True)
        print(f'{outdir} is done, {time.time() - start}')

    except Exception as e:
        logging.error(traceback.format_exc())
    finally:
        q.put(gpu_id)
        text = colored(f'{camtxt} - is done, count : {num_pushed}')
        print(text)
        return


if __name__ == "__main__":
    m = Manager()
    q = m.Queue()
    for gpu in range(num_p):
        q.put(gpu)
    pool = Pool(processes=num_p)

    num_pushed = 0
    while True:
        while True:
            num_all_scene = 0
            scene_type_list = glob.glob(osp.join(dataroot, '*'))
            for scene_type in scene_type_list:
                num_all_scene += len(os.listdir(scene_type))

            if num_all_scene > 0:
                break
            print(num_all_scene, 'waiting additional data...')
            time.sleep(60)

        outdir = glob.glob(osp.join(glob.glob(osp.join(dataroot, '*'))[0], '*'))[0]
        xml = scene2xml(outdir)

        outdir_type = outdir.split('/')[-2]
        final_repository = osp.join(repository_root, outdir_type)

        ff_list = os.listdir(outdir)
        ff_list = [f.split('_')[0] for f in ff_list]

        os.makedirs(final_repository, exist_ok=True)
        time.sleep(1)
        shutil.move(outdir, final_repository)
        time.sleep(1)
        outdir = os.path.join(final_repository, osp.basename(outdir))
        for k in ff_list:
            num_pushed += 1
            camtxt = xml2camtxt(xml, k)
            pool.apply_async(work, (xml, camtxt, outdir, k, num_pushed, q))


    pool.close()
    pool.join()
