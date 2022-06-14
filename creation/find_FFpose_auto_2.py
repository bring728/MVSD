import time

import cv2
from multiprocessing import Pool, Manager
from find_FFpose_func import *
from xml.etree.ElementTree import ElementTree, parse

# https://tempdev.tistory.com/entry/Python-multiprocessingPool-%EB%A9%80%ED%8B%B0%ED%94%84%EB%A1%9C%EC%84%B8%EC%8B%B1-2

processed_dir = '/home/vig-titan2/Data/OpenRooms_findpose/data/rendering/tmp'
scenes_root = '/home/vig-titan2/Data/OpenRooms_findpose/data/rendering/scenes'
num_gpu = 2
depth_to_baseline = 0.08
min_baseline = 0.08
max_baseline = 0.33

def xml2scene(scene):
    tmp = scene.split('scenes')
    a = tmp[1].split('/')
    xml_type = a[1]
    scene_name = a[2]
    scene_type = a[3].split('_')[0]
    return tmp[0] + 'data_FF_10_640/' + scene_type + '_' + xml_type + '/' + scene_name + '/'

def scene2xml(scene):
    a = scene.split('data_FF_10_640')
    b = a[1].split('/')
    scene_name = b[2]
    return a[0] + 'scenes/' + b[1].split('_')[1] + '/' + scene_name + '/' + b[1].split('_')[0] + '_FF.xml'

def main():
    m = Manager()
    q = m.Queue()
    for gpu in range(num_gpu):
        q.put(gpu)
    pool = Pool(processes=num_gpu)

    for xml_type in ['xml', 'xml1']:
    # for xml_type in ['xml',]:
        all_xml = sorted(glob.glob(osp.join(scenes_root, xml_type, '*')))
        all_scene_xml = [osp.join(a, 'main_FF.xml') for a in all_xml]
        # all_scene_xml = [osp.join(a, 'mainDiffMat_FF.xml') for a in all_xml]
        outdir_list = [xml2scene(m) for m in all_scene_xml]
        xml_list = [scene2xml(m) for m in outdir_list]
        num_scenes = len(outdir_list)
        z = 0
        for xml, out_dir in zip(xml_list, outdir_list):
            processed_dir = out_dir.replace('data_FF_10_640', 'tmp')
            name = out_dir.split('data_FF_10_640/')[1]
            if not osp.exists(xml):
                print(xml, 'does not exist!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                continue

            if osp.isfile(osp.join(processed_dir, 'pushed.txt')):
                z += 1
                continue

            # pre_pushed_file = osp.join(processed_dir.replace('DiffMat_', 'DiffLight_'), 'pushed.txt')
            pre_pushed_file1 = osp.join(processed_dir.replace('/main_', '/mainDiffLight_'), 'pushed.txt')
            pre_pushed_file2 = osp.join(processed_dir.replace('/main_', '/mainDiffMat_'), 'pushed.txt')
            if not osp.isfile(pre_pushed_file1) and not osp.isfile(pre_pushed_file2):
                z += 1
                continue

            print(f'{z} / {num_scenes}')
            os.makedirs(out_dir, exist_ok=True)

            cameraFile = xml.split('main')[0] + 'cam.txt'
            with open(cameraFile, 'r') as fIn:
                camera = fIn.readlines()
            num_cam = int(camera.pop(0))

            pre_pushed_txt = []
            if osp.isfile(pre_pushed_file1):
                with open(pre_pushed_file1, 'r') as fIn:
                    pre_pushed_txt += fIn.readlines()[0].split(' ')

            if osp.isfile(pre_pushed_file2):
                with open(pre_pushed_file2, 'r') as fIn:
                    pre_pushed_txt += fIn.readlines()[0].split(' ')
            pre_pushed_txt = set(pre_pushed_txt)
            org_dir = '/home/vig-titan2/Data/OpenRooms_org' + out_dir.split('data_FF_10_640')[1]
            if not osp.exists(org_dir):
                print(org_dir, 'is not exists, pass')
                continue

            f_tmp = open(osp.join(out_dir, 'pushed.txt'), 'w')
            f_tmp.writelines(' '.join(pre_pushed_txt))
            f_tmp.close()
            for k in pre_pushed_txt:
                k = int(k)
                depthname = (org_dir + f'imdepth_{k + 1}.dat').replace('DiffMat', '').replace('DiffLight', '')
                depth = loadBinary(depthname)
                baseline = np.mean(depth) * depth_to_baseline
                baseline = min(baseline, max_baseline)
                baseline = max(baseline, min_baseline)
                print(f'{name} // all: {num_cam}, current idx: {k}, baseline: {baseline:.3f}')

                tree = parse(xml)
                root = tree.getroot()
                sensor = root.findall('sensor')[0]
                fov = float(sensor.find('float').attrib['value'])
                fovaxis = sensor.find('string').attrib['value']
                if sensor.find('film').findall('integer')[0].attrib['name'] == 'height':
                    W = float(root.find('sensor').find('film').findall('integer')[1].attrib['value'])
                    H = float(root.find('sensor').find('film').findall('integer')[0].attrib['value'])
                else:
                    W = float(root.find('sensor').find('film').findall('integer')[0].attrib['value'])
                    H = float(root.find('sensor').find('film').findall('integer')[1].attrib['value'])

                if fovaxis == 'x':
                    focal = .5 * W / np.tan(.5 * fov * np.pi / 180.0)
                elif fovaxis == 'y':
                    focal = .5 * H / np.tan(.5 * fov * np.pi / 180.0)

                hwf = np.array([H, W, focal], dtype=float).reshape([3, 1])
                print(f'{k}, pushed')
                pool.apply_async(find_ff_pose, (camera, xml, out_dir, k, baseline, hwf, q))

            time.sleep(1)
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()
