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
renderer_directlight = '/Lib/OptixRendererDirectLight/src/bin/optixRenderer'

dataroot = '/Data/OpenRoomsDataset/data/rendering/data_FF_10_640/'
intermediate_path = '/Data/intermediate_path/'
optix_output_path = '/Data/optix_output/'
logroot = '/Data/logs/'
num_p = 2


def work(xml, camtxt, outdir, k, num_pushed, mode, outdir_tmp, q):
    gpu_id = q.get()
    start = time.time()
    try:
        if outdir_tmp == None:
            logger.debug(f'{outdir} - {k} is start on gpu {gpu_id}')
            for m in mode:
                if m == '7':
                    optixrenderer_arg = [renderer_directlight, '-f', xml, '-c', camtxt, '-o', osp.join(outdir, f'{k}_im'), '-m', m, '--imWidth', '40',
                                         '--imHeight', '30', '--maxPathLength', '2 ', '--gpuIds', str(gpu_id), '--forceOutput']
                    output = check_output(optixrenderer_arg, stderr=STDOUT, universal_newlines=True)

                    optixrenderer_arg = [renderer_light, '-f', xml, '-c', camtxt, '-o', osp.join(outdir, f'{k}_im'), '-m', m, '--gpuIds',
                                         str(gpu_id), '--forceOutput']
                    output = check_output(optixrenderer_arg, stderr=STDOUT, universal_newlines=True)
                else:
                    optixrenderer_arg = [renderer, '-f', xml, '-c', camtxt, '-o', osp.join(outdir, f'{k}_im'), '-m', m, '--gpuIds',
                                         str(gpu_id), '--forceOutput', '--maxIteration', '4', '--medianFilter']
                    output = check_output(optixrenderer_arg, stderr=STDOUT, universal_newlines=True)

        else:
            logger.debug(f'{outdir} - {k} is start from middle on gpu {gpu_id}')
            optixrenderer_arg = [renderer, '-f', xml, '-c', camtxt, '-o', osp.join(outdir_tmp, f'{k}_im'), '-m', '0', '--gpuIds',
                                 str(gpu_id), '--forceOutput', '--maxIteration', '4', '--medianFilter']
            output = check_output(optixrenderer_arg, stderr=STDOUT, universal_newlines=True)

            rendered = sorted(glob.glob(osp.join(outdir_tmp, '*.rgbe')))
            num_rendered = 9 - len(rendered)
            for idx, i in enumerate(rendered):
                shutil.move(i, osp.join(outdir, f'{k}_im_{idx + num_rendered + 1}.rgbe'))

            os.remove(camtxt)
            os.removedirs(outdir_tmp)

    except Exception as e:
        logging.error(traceback.format_exc())
    finally:
        q.put(gpu_id)
        logger.debug(f'{camtxt} - is done, count : {num_pushed}, {time.time() - start}')
        return


logger = logging.getLogger()
if __name__ == "__main__":
    intermediate_scenes = []
    scene_type_list = glob.glob(osp.join(intermediate_path, '*'))
    for scene_type in scene_type_list:
        intermediate_scenes += sorted(glob.glob(osp.join(scene_type, '*')))

    for scene in intermediate_scenes:
        new_scene = scene.replace(intermediate_path, dataroot)
        shutil.move(scene, new_scene)

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(u'%(asctime)s [%(levelname)8s] %(message)s')
    # StreamHandler
    streamingHandler = logging.StreamHandler()
    streamingHandler.setFormatter(formatter)
    # FileHandler
    file_handler = logging.FileHandler(osp.join(logroot, 'output.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(streamingHandler)

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

            intermediate_scenes = []
            scene_type_list = glob.glob(osp.join(intermediate_path, '*'))
            for scene_type in scene_type_list:
                intermediate_scenes += sorted(glob.glob(osp.join(scene_type, '*')))

            for scene in intermediate_scenes:
                outdir_type = osp.basename(osp.dirname(scene))

                all_files = os.listdir(scene)
                num_rgbe = len([a for a in all_files if a.endswith('.rgbe')])
                num_hdr = len([a for a in all_files if a.endswith('.hdr')])
                num_dat = len([a for a in all_files if a.endswith('.dat')])
                num_png = len([a for a in all_files if a.endswith('.png')])
                num_npy = len([a for a in all_files if a.endswith('.npy')])

                if num_rgbe == 9 * num_npy and num_hdr == 18 * num_npy and num_dat == 9 * num_npy and num_png == 36 * num_npy:
                    logger.debug(f'{scene} is done. moving to optix_output.')
                    #command = f'sshpass -p 8501 scp -r {scene} vig-titan-103@161.122.115.103:/media/vig-titan-103/mybookduo/OpenRooms_FF/{outdir_type}/'
                    #os.system(command)

                    final_repository = osp.join(optix_output_path, outdir_type)
                    os.makedirs(final_repository, exist_ok=True)
                    time.sleep(1)
                    shutil.move(scene, final_repository)

            # rendered_scene_list = glob.glob(osp.join(optix_output_path, '*'))
            # for scene in rendered_scene_list:
            #     all_files = os.listdir(scene)
            #     ff_list = sorted(list(set([f.split('_')[0] for f in all_files])))
            #     for k in ff_list:
            #         k_all_files = glob.glob(osp.join(scene, f'{k}_*.rgbe')) + glob.glob(osp.join(scene, f'{k}_*.dat')) + glob.glob(
            #             osp.join(scene, f'{k}_*.hdr')) + glob.glob(osp.join(scene, f'{k}_*.png')) + glob.glob(osp.join(scene, f'{k}_*.npy'))
            #         if len(k_all_files) == 64:
            #             done_dir = osp.join(optix_output_path, scene.split(intermediate_path)[1])
            #             os.makedirs(done_dir, exist_ok=True)
            #             k_new_files = [f.replace(intermediate_path, optix_output_path) for f in k_all_files]
            #             for f_old, f_new in zip(k_all_files, k_new_files):
            #                 os.rename(f_old, f_new)

            logger.debug(f'all scenes : {num_pushed}, waiting additional data...')
            time.sleep(120)

        all_scene = []
        scene_type_list = sorted(glob.glob(osp.join(dataroot, '*')))
        for scene_type in scene_type_list:
            all_scene += sorted(glob.glob(osp.join(scene_type, '*')))

        outdir = all_scene[0]
        xml = scene2xml(outdir)

        outdir_type = outdir.split('/')[-2]
        final_repository = osp.join(intermediate_path, outdir_type)

        os.makedirs(final_repository, exist_ok=True)
        time.sleep(1)
        shutil.move(outdir, final_repository)
        time.sleep(1)
        outdir = os.path.join(final_repository, osp.basename(outdir))

        ff_list = os.listdir(outdir)
        ff_list = sorted(list(set([f.split('_')[0] for f in ff_list])))

        for k in ff_list:
            if len(glob.glob(osp.join(outdir, f'{k}_*.rgbe'))) == 9:
                logger.debug(f'{outdir} - {k} is already made.')
                continue

            num_pushed += 1
            num_rendered_rgbe = len(glob.glob(osp.join(outdir, f'{k}_*.rgbe')))
            cam_file_name = xml2camtxt(xml, k)

            if num_rendered_rgbe == 0:
                logger.debug(f'{outdir} - {k} is pushed')
                mode = ['1', '2', '3', '4', '5', '7', '0']
                pool.apply_async(work, (xml, cam_file_name, outdir, k, num_pushed, mode, None, q))
                continue

            if num_rendered_rgbe > 0:
                with open(cam_file_name, 'r') as fIn:
                    txt = fIn.readlines()
                camtxt = [a.strip() for a in txt]

                cam_tmp_txt = [str(int(camtxt[0]) - num_rendered_rgbe)] + camtxt[num_rendered_rgbe * 3 + 1:]
                cam_file_name_tmp = osp.join(outdir, f'{k}_tmp.txt')
                with open(cam_file_name_tmp, 'w') as f:
                    f.write('\n'.join(cam_tmp_txt) + '\n')
                outdir_tmp = osp.join(outdir, f'{k}_tmp')
                os.makedirs(outdir_tmp, exist_ok=True)

                logger.debug(f'{outdir} - {k} is stopped during rgbe, pushed')
                pool.apply_async(work, (xml, cam_file_name_tmp, outdir, k, num_pushed, None, outdir_tmp, q))

    pool.close()
    pool.join()
