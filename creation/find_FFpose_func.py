import os
import traceback
import logging
import shutil
from subprocess import check_output, STDOUT, CalledProcessError
from utils_geometry import *
from transformation import *
from termcolor import colored

openroomsRoot = '/home/vig-titan2/Data/OpenRooms_findpose/data/rendering/'
renderer = '/home/vig-titan2/Downloads/OptixRenderer-master/build/bin/optixRenderer'
max_it = 10
z_min = 0.6
mask_threshold = 30.0

def make_camtxt(camera, var=None, baseline=0.3):
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

    return camtxt, ref2c


def make_full_FFcam(camera, var=None, baseline=0.3):
    baseline_low = baseline * 0.9
    baseline_high = baseline
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

    return np.array(ret_c2w_list), camtxt


def find_ff_pose(camera, xml, out_dir, k, baseline, hwf, q):
    gpu = q.get()
    try:
        flag_success = False
        camera = camera[3 * k:3 * k + 3]
        tmp_outdir = osp.join(out_dir, str(gpu))
        os.makedirs(tmp_outdir, exist_ok=True)

        var = {'x': 0.0, 'y': 0.0}
        camtxt, ref2c_list = make_camtxt(camera, var, baseline=baseline)

        cam = f'{tmp_outdir}/tmp.txt'
        with open(cam, 'w') as f:
            f.write('\n'.join(camtxt) + '\n')

        depths = [f'{tmp_outdir}/imdepth_{i + 1}.dat' for i in range(5)]
        masks = [f'{tmp_outdir}/immask_{i + 1}.png' for i in range(5)]

        mode = ['1', '4', '5']
        for it in range(max_it):
            for m in mode:
                optixrenderer_arg = [renderer, '-f', xml, '-c', cam, '-o', tmp_outdir + '/im', '-m', m, '--gpuIds', str(gpu), '--forceOutput']
                output = check_output(optixrenderer_arg, stderr=STDOUT, universal_newlines=True)

            d_flag_list = []
            w_flag_list = []
            d_min_list = []
            for depth_name, mask_name in zip(depths, masks):
                mask = 0.5 * (loadImage(mask_name) + 1)[0:1, ...]
                segall = (mask > 0.55)
                bool_tmp = not np.percentile(mask, mask_threshold) < 1
                w_flag_list.append(bool_tmp)
                if bool_tmp:
                    depth = loadBinary(depth_name)
                    depth[~segall] = np.nan
                    min = np.nanpercentile(depth, 3.0)
                    d_flag_list.append(min > z_min)
                    d_min_list.append(min)
                else:
                    d_flag_list.append(bool_tmp)
                    d_min_list.append(10.0)

            # print(d_flag_list)
            # print(k, d_min_list + w_flag_list)
            if all(d_flag_list) and all(w_flag_list):
                # convert current pose to FF and save
                poses, camtxt = make_full_FFcam(camera, var, baseline=baseline)
                poses = poses.transpose([1, 2, 0])
                save_arr = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1, 1, 9])], 1)
                np.save(f'{out_dir}/{k}_cam_mats.npy', save_arr)
                with open((xml.split('main')[0] + str(k) + '_cam_main' + xml.split('main')[1].replace('xml', 'txt')), 'w') as f:
                    f.write('\n'.join(camtxt) + '\n')
                flag_success = True
                break

            else:
                d_min_list = list(np.around(np.array(d_min_list), 2))
                text = colored(f'retry, {k}, {d_min_list}, {w_flag_list}', 'red', attrs=['reverse', 'blink'])
                print(text)
                # center error
                min_idx = np.argmin(np.array(d_min_list))
                if min_idx == 2:
                    var['x'] += np.random.uniform(-baseline / 3, baseline / 3)
                    var['y'] += np.random.uniform(-baseline / 3, baseline / 3)
                elif min_idx == 0:
                    var['x'] += np.random.uniform(baseline / 2, baseline / 2)
                    var['y'] += np.random.uniform(baseline / 2, baseline / 2)
                elif min_idx == 1:
                    var['x'] -= np.random.uniform(baseline / 2, baseline / 2)
                    var['y'] += np.random.uniform(baseline / 2, baseline / 2)
                elif min_idx == 3:
                    var['x'] += np.random.uniform(baseline / 2, baseline / 2)
                    var['y'] -= np.random.uniform(baseline / 2, baseline / 2)
                elif min_idx == 4:
                    var['x'] -= np.random.uniform(baseline / 2, baseline / 2)
                    var['y'] -= np.random.uniform(baseline / 2, baseline / 2)

                if not w_flag_list[2]:
                    var['x'] += np.random.uniform(-baseline / 2, baseline / 2)
                    var['y'] += np.random.uniform(-baseline / 2, baseline / 2)
                if not w_flag_list[0]:
                    var['x'] += np.random.uniform(baseline / 2, baseline / 2)
                    var['y'] += np.random.uniform(baseline / 2, baseline / 2)
                if not w_flag_list[1]:
                    var['x'] -= np.random.uniform(baseline / 2, baseline / 2)
                    var['y'] += np.random.uniform(baseline / 2, baseline / 2)
                if not w_flag_list[3]:
                    var['x'] += np.random.uniform(baseline / 2, baseline / 2)
                    var['y'] -= np.random.uniform(baseline / 2, baseline / 2)
                if not w_flag_list[4]:
                    var['x'] -= np.random.uniform(baseline / 2, baseline / 1)
                    var['y'] -= np.random.uniform(baseline / 2, baseline / 1)

                camtxt, ref2c_list = make_camtxt(camera, var, baseline=baseline)
                with open(cam, 'w') as f:
                    f.write('\n'.join(camtxt) + '\n')
        d_min_list = list(np.around(np.array(d_min_list), 2))
        if flag_success:
            print('success', k, d_min_list, w_flag_list)
        else:
            text = colored(f'failed, {k}, {d_min_list}, {w_flag_list}', 'blue', attrs=['reverse', 'blink'])
            print(text)

        albedos = glob.glob(tmp_outdir + '/imbaseColor_*')
        for a in albedos:
            name = a.split('/')[-1]
            shutil.move(a, f'{out_dir}/{k}_{name}')
        shutil.rmtree(tmp_outdir)

    except Exception as e:
        logging.error(traceback.format_exc())
    finally:
        q.put(gpu)
        return



def find_ff_pose_debug(camera, xml, out_dir, k, baseline, hwf):
    gpu = 0
    camera = camera[3 * k:3 * k + 3]
    tmp_outdir = osp.join(out_dir, str(gpu))
    os.makedirs(tmp_outdir, exist_ok=True)

    var = {'x': 0.0, 'y': 0.0}
    camtxt, ref2c_list = make_camtxt(camera, var, baseline=baseline)

    cam = f'{tmp_outdir}/tmp.txt'
    with open(cam, 'w') as f:
        f.write('\n'.join(camtxt) + '\n')

    depths = [f'{tmp_outdir}/imdepth_{i + 1}.dat' for i in range(5)]
    masks = [f'{tmp_outdir}/immask_{i + 1}.png' for i in range(5)]


    mode = ['1', '4', '5']
    for it in range(max_it):
        for m in mode:
            optixrenderer_arg = [renderer, '-f', xml, '-c', cam, '-o', tmp_outdir + '/im', '-m', m, '--gpuIds', str(gpu), '--forceOutput']
            output = check_output(optixrenderer_arg, stderr=STDOUT, universal_newlines=True)

        d_flag_list = []
        w_flag_list = []
        d_min_list = []
        for depth_name, mask_name in zip(depths, masks):
            mask = 0.5 * (loadImage(mask_name) + 1)[0:1, ...]
            segall = (mask > 0.55)
            bool_tmp = not np.percentile(mask, mask_threshold) < 1
            w_flag_list.append(bool_tmp)
            if bool_tmp:
                depth = loadBinary(depth_name)
                depth[~segall] = np.nan
                min = np.nanpercentile(depth, 5.0)
                d_flag_list.append(min > z_min)
                d_min_list.append(min)
            else:
                d_flag_list.append(bool_tmp)
                d_min_list.append(10.0)

        # print(d_flag_list)
        # print(k, d_min_list + w_flag_list)
        if all(d_flag_list) and all(w_flag_list):
            # convert current pose to FF and save
            poses, camtxt = make_full_FFcam(camera, var, baseline=baseline)
            poses = poses.transpose([1, 2, 0])
            save_arr = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1, 1, 9])], 1)
            np.save(f'{out_dir}/{k}_cam_mats.npy', save_arr)
            with open((xml.split('main')[0] + str(k) + '_cam_main' + xml.split('main')[1].replace('xml', 'txt')), 'w') as f:
                f.write('\n'.join(camtxt) + '\n')
            break

        else:
            d_min_list = list(np.around(np.array(d_min_list), 2))
            text = colored(f'failed, {k}, {d_min_list}, {w_flag_list}', 'red', attrs=['reverse', 'blink'])
            print(text)
            # center error
            min_idx = np.argmin(np.array(d_min_list))
            if min_idx == 2:
                var['x'] += np.random.uniform(-baseline / 3, baseline / 3)
                var['y'] += np.random.uniform(-baseline / 3, baseline / 3)
            elif min_idx == 0:
                var['x'] += np.random.uniform(baseline / 2, baseline / 2)
                var['y'] += np.random.uniform(baseline / 2, baseline / 2)
            elif min_idx == 1:
                var['x'] -= np.random.uniform(baseline / 2, baseline / 2)
                var['y'] += np.random.uniform(baseline / 2, baseline / 2)
            elif min_idx == 3:
                var['x'] += np.random.uniform(baseline / 2, baseline / 2)
                var['y'] -= np.random.uniform(baseline / 2, baseline / 2)
            elif min_idx == 4:
                var['x'] -= np.random.uniform(baseline / 2, baseline / 2)
                var['y'] -= np.random.uniform(baseline / 2, baseline / 2)

            if not w_flag_list[2]:
                var['x'] += np.random.uniform(-baseline / 2, baseline / 2)
                var['y'] += np.random.uniform(-baseline / 2, baseline / 2)
            if not w_flag_list[0]:
                var['x'] += np.random.uniform(baseline / 2, baseline / 2)
                var['y'] += np.random.uniform(baseline / 2, baseline / 2)
            if not w_flag_list[1]:
                var['x'] -= np.random.uniform(baseline / 2, baseline / 2)
                var['y'] += np.random.uniform(baseline / 2, baseline / 2)
            if not w_flag_list[3]:
                var['x'] += np.random.uniform(baseline / 2, baseline / 2)
                var['y'] -= np.random.uniform(baseline / 2, baseline / 2)
            if not w_flag_list[4]:
                var['x'] -= np.random.uniform(baseline / 2, baseline / 1)
                var['y'] -= np.random.uniform(baseline / 2, baseline / 1)

            camtxt, ref2c_list = make_camtxt(camera, var, baseline=baseline)
            with open(cam, 'w') as f:
                f.write('\n'.join(camtxt) + '\n')
    d_min_list = list(np.around(np.array(d_min_list), 2))
    print('success', k, d_min_list + w_flag_list)

    albedos = glob.glob(tmp_outdir + '/imbaseColor_*')
    for a in albedos:
        name = a.split('/')[-1]
        shutil.move(a, f'{out_dir}/{k}_{name}')
    shutil.rmtree(tmp_outdir)