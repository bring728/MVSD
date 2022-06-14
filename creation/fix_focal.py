import time
from multiprocessing import Pool, Manager
from find_FFpose_func import *
from xml.etree.ElementTree import ElementTree, parse
# https://tempdev.tistory.com/entry/Python-multiprocessingPool-%EB%A9%80%ED%8B%B0%ED%94%84%EB%A1%9C%EC%84%B8%EC%8B%B1-2

processed_dir = '/home/vig-titan2/Data/OpenRooms_findpose/data/rendering/tmp'
scenes_root = '/home/vig-titan2/Data/OpenRooms_findpose/data/rendering/scenes'
num_gpu = 2
depth_to_baseline = 0.09
min_baseline = 0.07
max_baseline = 0.8


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
    server_list = sorted(glob.glob('/home/vig-titan2/Data/@@FF_renew/*'))
    out_root = '/home/vig-titan2/Data/@@FF_renew/all'
    for server in server_list:
        if osp.basename(server) == 'all':
            continue
        scene_folder = glob.glob(osp.join(server, 'data_FF_10_640', '*'))
        for scenes_name in scene_folder:
            scene_list = sorted(glob.glob(osp.join(scenes_name, '*')))
            for scene in scene_list:
                out_folder = osp.join(out_root, scene.split('data_FF_10_640/')[1])
                os.makedirs(out_folder, exist_ok=True)
                xml = scene2xml(scene)
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

                npy_list = glob.glob(osp.join(scene, '*'))
                for npy in npy_list:
                    cam_mats = np.load(npy)
                    cam_mats[-1, -1, :] = focal
                    np.save(osp.join(out_folder, osp.basename(npy).replace('poses_bounds', 'cam_mats')), cam_mats)


if __name__ == '__main__':
    main()
