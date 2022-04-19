import os
import os.path as osp
import glob
from xml.etree.ElementTree import parse
from utils import *
from utils_geometry import *
import numpy as np
import platform
from transformation import *
from subprocess import check_output, STDOUT, CalledProcessError
from PIL import Image
import random
import time
import sys


def make_camtxt(camera, xmlFile, k, var=None, baseline=0.3):
    tree = parse(xmlFile)
    root = tree.getroot()
    sensor = root.findall('sensor')
    assert len(sensor) == 1, f'{xmlFile}, multiple sensor'
    sensor = sensor[0]
    assert sensor.find('film').attrib['type'] == 'hdrfilm', f'{xmlFile}, not hdr film'
    fov = float(sensor.find('float').attrib['value'])
    fovaxis = sensor.find('string').attrib['value']
    if sensor.find('film').findall('integer')[0].attrib['name'] == 'height':
        W = float(root.find('sensor').find('film').findall('integer')[1].attrib['value'])
        H = float(root.find('sensor').find('film').findall('integer')[0].attrib['value'])
    else:
        W = float(root.find('sensor').find('film').findall('integer')[0].attrib['value'])
        H = float(root.find('sensor').find('film').findall('integer')[1].attrib['value'])

    if fovaxis == 'x':
        focal = .5 * W / np.tan(.5 * fov)
    elif fovaxis == 'y':
        focal = .5 * H / np.tan(.5 * fov)
    else:
        raise Exception('fov axis error!')

    camera = camera[3 * k:3 * k + 3]

    # [right, down, forward] order
    c2w_list = []
    pos_list = []
    up_list = []
    tgt_pt_list = []

    pos_org = np.array(camera[0].replace('\n', '').split(' '), dtype=float)
    tgt_pt_org = np.array(camera[1].replace('\n', '').split(' '), dtype=float)
    up_org = np.array(camera[2].replace('\n', '').split(' '), dtype=float)

    # print(np.linalg.norm(tgt_pt_org - pos_org))
    z = normalize(tgt_pt_org - pos_org)
    y = normalize(-up_org)
    x = normalize(np.cross(y, z))
    c2w = np.concatenate([x[..., np.newaxis], y[..., np.newaxis], z[..., np.newaxis]], axis=1)

    if var is not None:
        pos_org = pos_org + x * var['x'] + y * var['y']

    # left-up
    c2w_list.append(c2w)
    pos_list.append(pos_org - x * baseline - y * baseline)

    # right-up
    c2w_list.append(c2w)
    pos_list.append(pos_org + x * baseline - y * baseline)

    # center
    c2w_list.append(c2w)
    pos_list.append(pos_org)

    # left-down
    c2w_list.append(c2w)
    pos_list.append(pos_org - x * baseline + y * baseline)

    # right-down
    c2w_list.append(c2w)
    pos_list.append(pos_org + x * baseline + y * baseline)

    for c2w, pos in zip(c2w_list, pos_list):
        up_list.append(normalize(-c2w[:, 1]))
        tgt_pt_list.append(pos + normalize(c2w[:, 2]))

    ref2c = []
    bottom = np.array([0, 0, 0, 1], dtype=float).reshape([1, 4])

    ref2w = np.concatenate([c2w_list[2], pos_list[2][..., np.newaxis]], axis=1)
    ref2w = np.concatenate([ref2w, bottom], 0)
    for c2w, pos in zip(c2w_list, pos_list):
        c2w = np.concatenate([c2w, pos[..., np.newaxis]], axis=1)
        w2c = np.linalg.inv(np.concatenate([c2w, bottom], 0))
        ref2c.append(w2c @ ref2w)

    pos_list = [str(x)[1:-1] for x in pos_list]
    tgt_pt_list = [str(x)[1:-1] for x in tgt_pt_list]
    up_list = [str(x)[1:-1] for x in up_list]

    camtxt = []
    camtxt.append('5')
    for pos, tgt_pt, up in zip(pos_list, tgt_pt_list, up_list):
        camtxt.append(pos)
        camtxt.append(tgt_pt)
        camtxt.append(up)

    return camtxt, ref2c, focal


def make_full_FFcam(camera, xmlFile, k, var=None, baseline=0.3, variance=0.03):
    tree = parse(xmlFile)
    root = tree.getroot()
    sensor = root.findall('sensor')
    assert len(sensor) == 1, f'{xmlFile}, multiple sensor'
    sensor = sensor[0]
    assert sensor.find('film').attrib['type'] == 'hdrfilm', f'{xmlFile}, not hdr film'
    fov = float(sensor.find('float').attrib['value'])
    fovaxis = sensor.find('string').attrib['value']
    if sensor.find('film').findall('integer')[0].attrib['name'] == 'height':
        W = float(root.find('sensor').find('film').findall('integer')[1].attrib['value'])
        H = float(root.find('sensor').find('film').findall('integer')[0].attrib['value'])
    else:
        W = float(root.find('sensor').find('film').findall('integer')[0].attrib['value'])
        H = float(root.find('sensor').find('film').findall('integer')[1].attrib['value'])

    if fovaxis == 'x':
        focal = .5 * W / np.tan(.5 * fov)
    elif fovaxis == 'y':
        focal = .5 * H / np.tan(.5 * fov)
    else:
        raise Exception('fov axis error!')

    hwf = np.array([H, W, focal], dtype=float).reshape([3, 1])

    camera = camera[3 * k:3 * k + 3]

    baseline_low = baseline - variance
    baseline_high = baseline + variance
    angle_low = -0.5
    angle_high = 0.5

    tgt_pt_list = []
    up_list = []
    # [right, down, forward] order
    c2w_list = []
    pos_list = []

    pos_org = np.array(camera[0].replace('\n', '').split(' '), dtype=float)
    tgt_pt_org = np.array(camera[1].replace('\n', '').split(' '), dtype=float)
    up_org = np.array(camera[2].replace('\n', '').split(' '), dtype=float)

    # print(np.linalg.norm(tgt_pt_org - pos_org))
    z = normalize(tgt_pt_org - pos_org)
    y = normalize(-up_org)
    x = normalize(np.cross(y, z))
    c2w = np.concatenate([x[..., np.newaxis], y[..., np.newaxis], z[..., np.newaxis]], axis=1)

    if var is not None:
        pos_org = pos_org + x * var['x'] + y * var['y']

    # left-up
    R_x = rotation_matrix(angle=np.deg2rad(np.random.uniform(angle_low, 0)), direction=x)[:3, :3]
    R_y = rotation_matrix(angle=np.deg2rad(np.random.uniform(0, angle_high)), direction=y)[:3, :3]
    c2w_list.append(R_y @ R_x @ c2w)
    pos_list.append(pos_org - x * np.random.uniform(baseline_low, baseline_high) - y * np.random.uniform(baseline_low, baseline_high))

    # up
    R_x = rotation_matrix(angle=np.deg2rad(np.random.uniform(angle_low, 0)), direction=x)[:3, :3]
    R_y = rotation_matrix(angle=np.deg2rad(np.random.uniform(angle_low, angle_high)), direction=y)[:3, :3]
    c2w_list.append(R_y @ R_x @ c2w)
    pos_list.append(pos_org - y * np.random.uniform(baseline_low, baseline_high))

    # right-up
    R_x = rotation_matrix(angle=np.deg2rad(np.random.uniform(angle_low, 0)), direction=x)[:3, :3]
    R_y = rotation_matrix(angle=np.deg2rad(np.random.uniform(angle_low, 0)), direction=y)[:3, :3]
    c2w_list.append(R_y @ R_x @ c2w)
    pos_list.append(pos_org + x * np.random.uniform(baseline_low, baseline_high) - y * np.random.uniform(baseline_low, baseline_high))

    # left
    R_x = rotation_matrix(angle=np.deg2rad(np.random.uniform(angle_low, angle_high)), direction=x)[:3, :3]
    R_y = rotation_matrix(angle=np.deg2rad(np.random.uniform(0, angle_high)), direction=y)[:3, :3]
    c2w_list.append(R_y @ R_x @ c2w)
    pos_list.append(pos_org - x * np.random.uniform(baseline_low, baseline_high))

    # center
    c2w_list.append(c2w)
    pos_list.append(pos_org)

    # right
    R_x = rotation_matrix(angle=np.deg2rad(np.random.uniform(angle_low, angle_high)), direction=x)[:3, :3]
    R_y = rotation_matrix(angle=np.deg2rad(np.random.uniform(angle_low, 0)), direction=y)[:3, :3]
    c2w_list.append(R_y @ R_x @ c2w)
    pos_list.append(pos_org + x * np.random.uniform(baseline_low, baseline_high))

    # left-down
    R_x = rotation_matrix(angle=np.deg2rad(np.random.uniform(0, angle_high)), direction=x)[:3, :3]
    R_y = rotation_matrix(angle=np.deg2rad(np.random.uniform(0, angle_high)), direction=y)[:3, :3]
    c2w_list.append(R_y @ R_x @ c2w)
    pos_list.append(pos_org - x * np.random.uniform(baseline_low, baseline_high) + y * np.random.uniform(baseline_low, baseline_high))

    # down
    R_x = rotation_matrix(angle=np.deg2rad(np.random.uniform(0, angle_high)), direction=x)[:3, :3]
    R_y = rotation_matrix(angle=np.deg2rad(np.random.uniform(angle_low, angle_high)), direction=y)[:3, :3]
    c2w_list.append(R_y @ R_x @ c2w)
    pos_list.append(pos_org + y * np.random.uniform(baseline_low, baseline_high))

    # right-down
    R_x = rotation_matrix(angle=np.deg2rad(np.random.uniform(0, angle_high)), direction=x)[:3, :3]
    R_y = rotation_matrix(angle=np.deg2rad(np.random.uniform(angle_low, 0)), direction=y)[:3, :3]
    c2w_list.append(R_y @ R_x @ c2w)
    pos_list.append(pos_org + x * np.random.uniform(baseline_low, baseline_high) + y * np.random.uniform(baseline_low, baseline_high))

    ret_c2w_list = []
    for c2w, pos in zip(c2w_list, pos_list):
        up_list.append(normalize(-c2w[:, 1]))
        tgt_pt_list.append(pos + normalize(c2w[:, 2]))
        ret_c2w_list.append(np.concatenate([c2w, pos[..., np.newaxis]], axis=1))

    tgt_pt_list = [str(x)[1:-1] for x in tgt_pt_list]
    up_list = [str(x)[1:-1] for x in up_list]
    pos_list = [str(x)[1:-1] for x in pos_list]

    camtxt = []
    camtxt.append('9')
    for pos, tgt_pt, up in zip(pos_list, tgt_pt_list, up_list):
        camtxt.append(pos)
        camtxt.append(tgt_pt)
        camtxt.append(up)

    return hwf, np.array(ret_c2w_list), camtxt


root = '/media/vig-titan2/My Book Duo'
root_renderer = '/home/vig-titan2/Downloads'
# root = '/home/happily/Data'
# root_renderer = '/home/happily/Lib/'
if platform.system() == 'Linux':
    openroomsRoot = root + '/OpenRoomsDataset/data/rendering/'
    sep = '/'
    renderer = root_renderer + '/OptixRenderer-master/build/bin/optixRenderer'
elif platform.system() == 'Windows':
    openroomsRoot = 'D:/OpenRoomsDataset_demo/data/rendering'
    sep = '\\'
    renderer = '/home/happily/Lib/OptixRenderer-master/build/bin/optixRenderer'

else:
    assert f'where is here? : {platform.system()}'

baseline = 15
variance = 1
z_min = 1.1

def main():
    # if len(sys.argv) < 4:
    #     print('argv error')
    #     return
    # proc_id = int(sys.argv[1])
    # num_proc = int(sys.argv[2])
    # num_gpu = int(sys.argv[3])
    proc_id = 0
    num_proc = 1
    num_gpu = 1

    scene_not_FF = []
    z = 0
    for x in ['xml', 'xml1']:
        sceneRoot = osp.join(openroomsRoot, f'scenes/{x}')
        scenes_list = sorted(glob.glob(osp.join(sceneRoot, '*')))
        outputRoot = osp.join(openroomsRoot, f'data_FF_{baseline}_640')
        os.makedirs(outputRoot, exist_ok=True)

        mode = ['4', '5']
        for scene in scenes_list:
            z += 1
            if not (z % num_proc == proc_id):
                continue

            scene_name = scene.split(x)[1]
            cameraFile = osp.join(scene, 'cam.txt')
            output_dirs = [osp.join(outputRoot, 'main_' + x + scene_name),
                           osp.join(outputRoot, 'mainDiffLight_' + x + scene_name),
                           osp.join(outputRoot, 'mainDiffMat_' + x + scene_name)]
            xmlFiles = [osp.join(scene, 'main_FF_640.xml'), osp.join(scene, 'mainDiffLight_FF_640.xml'), osp.join(scene, 'mainDiffMat_FF_640.xml')]
            for dir in output_dirs:
                os.makedirs(dir, exist_ok=True)

            with open(cameraFile, 'r') as fIn:
                camera = fIn.readlines()
            num_cam = int(camera.pop(0))
            max_it = num_cam * 10
            for out_dir, xml in zip(output_dirs, xmlFiles):
                if osp.isfile(out_dir + '/poses_bounds.npy'):
                    print(f'already made')
                    continue

                print(xml)
                k = random.randrange(num_cam)
                var = {'x': 0.0, 'y': 0.0}
                camtxt, ref2c_list, _ = make_camtxt(camera, xml, k, var, baseline=(baseline + variance) / 100)

                cam = f'{scene}/tmp.txt'
                with open(cam, 'w') as f:
                    f.write('\n'.join(camtxt) + '\n')

                depths = [f'{out_dir}/imdepth_{i + 1}.dat' for i in range(5)]
                masks = [f'{out_dir}/immask_{i + 1}.png' for i in range(5)]

                flag_tmp = True
                count = 0
                for it in range(max_it):
                    for m in mode:
                        # start = time.time()
                        optixrenderer_arg = [renderer, '-f', xml, '-c', cam, '-o', out_dir + '/im', '-m', m, '--gpuIds', str(proc_id % num_gpu), '--forceOutput']
                        output = check_output(optixrenderer_arg, stderr=STDOUT, universal_newlines=True)

                    d_flag_list = []
                    w_flag_list = []
                    d_min_list = []
                    for depth_name, mask_name in zip(depths, masks):
                        mask = 0.5 * (loadImage(mask_name) + 1)[0:1, ...]
                        bool_tmp = not np.percentile(mask, 10.0) < 1
                        w_flag_list.append(bool_tmp)
                        if bool_tmp:
                            depth = loadBinary(depth_name)
                            min = np.percentile(depth[np.where(mask == 1.0)], 1.0)
                            d_flag_list.append(min > z_min)
                            d_min_list.append(min)
                        else:
                            d_flag_list.append(bool_tmp)
                            d_min_list.append(10.0)

                    if all(d_flag_list) and all(w_flag_list):
                        # convert current pose to FF and save
                        hwf, poses, camtxt = make_full_FFcam(camera, xml, k, var, baseline=baseline / 100, variance=variance / 100)
                        poses = poses.transpose([1, 2, 0])
                        save_arr = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1, 1, 9])], 1)
                        np.save(f'{out_dir}/poses_bounds.npy', save_arr)
                        dir_type = out_dir.split(sep)[-2].split('_')[0]
                        with open(f'{scene}/cam_{dir_type}_{baseline}.txt', 'w') as f:
                            f.write('\n'.join(camtxt) + '\n')
                        flag_tmp = False
                        break

                    else:
                        # center error
                        count += 1
                        if count > 8:
                            count = 0
                            var = {'x': 0.0, 'y': 0.0}
                            k = random.randrange(num_cam)
                        else:
                            if not all(d_flag_list):
                                min_idx = np.argmin(np.array(d_min_list))
                                if min_idx == 2:
                                    var['x'] += np.random.uniform(-baseline / 300, baseline / 300)
                                    var['y'] += np.random.uniform(-baseline / 300, baseline / 300)
                                elif min_idx == 0:
                                    var['x'] += np.random.uniform(baseline / 200, baseline / 100)
                                    var['y'] += np.random.uniform(baseline / 200, baseline / 100)
                                elif min_idx == 1:
                                    var['x'] -= np.random.uniform(baseline / 200, baseline / 100)
                                    var['y'] += np.random.uniform(baseline / 200, baseline / 100)
                                elif min_idx == 3:
                                    var['x'] += np.random.uniform(baseline / 200, baseline / 100)
                                    var['y'] -= np.random.uniform(baseline / 200, baseline / 100)
                                elif min_idx == 4:
                                    var['x'] -= np.random.uniform(baseline / 200, baseline / 100)
                                    var['y'] -= np.random.uniform(baseline / 200, baseline / 100)

                            if not w_flag_list[2]:
                                var['x'] += np.random.uniform(-baseline / 200, baseline / 200)
                                var['y'] += np.random.uniform(-baseline / 200, baseline / 200)
                            if not w_flag_list[0]:
                                var['x'] += np.random.uniform(baseline / 200, baseline / 100)
                                var['y'] += np.random.uniform(baseline / 200, baseline / 100)
                            if not w_flag_list[1]:
                                var['x'] -= np.random.uniform(baseline / 200, baseline / 100)
                                var['y'] += np.random.uniform(baseline / 200, baseline / 100)
                            if not w_flag_list[3]:
                                var['x'] += np.random.uniform(baseline / 200, baseline / 100)
                                var['y'] -= np.random.uniform(baseline / 200, baseline / 100)
                            if not w_flag_list[4]:
                                var['x'] -= np.random.uniform(baseline / 200, baseline / 100)
                                var['y'] -= np.random.uniform(baseline / 200, baseline / 100)

                        camtxt, ref2c_list, _ = make_camtxt(camera, xml, k, var, baseline=(baseline + variance) / 100)
                        cam = f'{scene}/tmp.txt'
                        with open(cam, 'w') as f:
                            f.write('\n'.join(camtxt) + '\n')

                if flag_tmp:
                    scene_not_FF.append(xml)
                    print(f'can not find proper FF camera {xml}')

    with open(f'{baseline}_not_rendered_{proc_id}.txt', 'w') as f:
        f.write('\n'.join(scene_not_FF) + '\n')


if __name__ == '__main__':
    main()