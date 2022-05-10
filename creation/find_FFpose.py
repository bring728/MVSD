import os
from utils import *
from multiprocessing import Pool, Queue, Manager
from find_FFpose_func import *
from xml.etree.ElementTree import ElementTree, parse
# https://tempdev.tistory.com/entry/Python-multiprocessingPool-%EB%A9%80%ED%8B%B0%ED%94%84%EB%A1%9C%EC%84%B8%EC%8B%B1-2

processed_dir = '/home/vig-titan-103/Data/OpenRooms_findpose/data/rendering/tmp'
scenes_root = '/home/vig-titan-103/Data/OpenRooms_findpose/data/rendering/scenes'
num_gpu = 2
depth_to_baseline = 0.06
min_baseline = 0.08
max_baseline = 0.28


def image_concater(im_list):
    num_img = len(im_list)
    num_img_half = round(len(im_list) / 2 + 0.1)
    if num_img == 0:
        return None
    elif num_img % 2 == 0:
        return cv2.vconcat([cv2.hconcat(im_list[:num_img_half]), cv2.hconcat(im_list[num_img_half:])])
    elif num_img % 2 == 1:
        im_list_tmp = im_list + [np.zeros_like(im_list[0]), ]
        return cv2.vconcat([cv2.hconcat(im_list_tmp[:num_img_half]), cv2.hconcat(im_list_tmp[num_img_half:])])


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


def next_k(k, k_list, num_cam, arrow='right'):
    if arrow == 'right':
        k += 1
        k %= num_cam
        while k in k_list:
            k += 1
            k %= num_cam
    elif arrow == 'left':
        k -= 1
        k += num_cam
        k %= num_cam
        while k in k_list:
            k -= 1
            k += num_cam
            k %= num_cam
    else:
        raise Exception('asdfsdf')

    return k


def main():
    m = Manager()
    q = m.Queue()
    for gpu in range(num_gpu):
        q.put(gpu)
    pool = Pool(processes=num_gpu)

    processed_scenes = []
    processed = glob.glob(processed_dir + '/*')
    for p in processed:
        processed_scenes += glob.glob(p + '/*')
    processed_scenes = [m.replace('tmp', 'data_FF_10_640') + '/' for m in processed_scenes]

    all_xml = sorted(glob.glob(scenes_root + '/xml/*') + glob.glob(scenes_root + '/xml1/*'))
    all_scene_xml = [osp.join(a, 'main_FF.xml') for a in all_xml] + [osp.join(a, 'mainDiffMat_FF.xml') for a in all_xml] + [osp.join(a, 'mainDiffLight_FF.xml') for a in all_xml]
    outdir_list = [xml2scene(m) for m in all_scene_xml]
    outdir_list = sorted(list(set(outdir_list) - set(processed_scenes)))
    xml_list = [scene2xml(m) for m in outdir_list]

    num_scenes = len(outdir_list)
    z = 0
    flag = False
    for xml, out_dir in zip(xml_list, outdir_list):
        if not osp.exists(xml):
            print(xml, 'does not exist!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            continue

        print(f'{z} / {num_scenes}')
        name = out_dir.split('data_FF_10_640/')[1]
        scene_type = name.split('/')[0]

        os.makedirs(out_dir, exist_ok=True)
        cameraFile = xml.split('main')[0] + 'cam.txt'
        org_dir = '/media/vig-titan-103/mybookduo/OpenRooms_org' + out_dir.split('data_FF_10_640')[1]
        if not osp.exists(org_dir):
            print(org_dir, 'is not exists, pass')
            continue
        with open(cameraFile, 'r') as fIn:
            camera = fIn.readlines()
        num_cam = int(camera.pop(0))

        if osp.isfile(out_dir + 'pushed.txt'):
            print(f'already made')
            z += 1
            continue

        xml_dir = os.path.dirname(xml)
        files = os.listdir(xml_dir)
        files = [f for f in files if len(f.split('_')) == 4]
        files = [f for f in files if len(f.split('_')) == 4]
        for f in files:
            if f.split('_')[2] == scene_type.split('_')[0]:
                print('try to remake.. clearing : ', os.path.join(xml_dir, f))
                os.remove(os.path.join(xml_dir, f))

        k = 0
        k_list = []
        im_list = []
        while True:
            imname = org_dir + f'im_{k + 1}.hdr'
            maskname = (org_dir + f'immask_{k + 1}.png').replace('DiffMat', '')
            depthname = (org_dir + f'imdepth_{k + 1}.dat').replace('DiffMat', '').replace('DiffLight', '')
            im = loadHdr(imname)
            seg = 0.5 * (loadImage(maskname) + 1)[0:1, :, :]
            scale = get_hdr_scale(im, seg, 'TEST')
            im = np.clip(im * scale, 0, 1.0)
            im = im ** (1.0 / 2.2)
            im = cv2.cvtColor(np.transpose(im, [1, 2, 0]), cv2.COLOR_RGB2BGR)

            segObj = (seg > 0.9)
            depth = loadBinary(depthname)
            baseline = np.mean(segObj * depth) * depth_to_baseline
            baseline = min(baseline, max_baseline)
            baseline = max(baseline, min_baseline)
            print(f'{name} // all: {num_cam}, current idx: {k}, baseline: {baseline:.3f}')

            cv2.namedWindow(str(num_cam))  # Create a named window
            cv2.moveWindow(str(num_cam), 100, 200)  # Move it to (40,30)

            im_previous = image_concater(im_list)
            if im_previous is not None:
                cv2.imshow(str(num_cam), cv2.hconcat([im, im_previous]))
            else:
                cv2.imshow(str(num_cam), im)
            key = cv2.waitKey(0)
            if key == ord('a'):
                im_list.append(cv2.resize(im, (320, 240)))
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
                    focal = .5 * W / np.tan(.5 * fov)
                elif fovaxis == 'y':
                    focal = .5 * H / np.tan(.5 * fov)

                hwf = np.array([H, W, focal], dtype=float).reshape([3, 1])
                print(f'{k}, pushed')
                pool.apply_async(find_ff_pose, (camera, xml, out_dir, k, baseline, hwf, q))
                # pool.apply_async(find_ff_pose_debug, (camera, xml, out_dir, k, baseline, hwf, q))
                k_list.append(k)
                if set(k_list) == set(range(num_cam)):
                    print('all image is pushed.')
                    break
                k = next_k(k, k_list, num_cam, 'right')

            elif key == ord('d'):
                im_list.append(cv2.resize(im, (320, 240)))
                # for noisy scene, sample count 2048
                tree = parse(xml)
                root = tree.getroot()
                root.find('sensor').find('sampler').find('integer').attrib['value'] = '2048'
                ElementTree(root).write(xml)

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
                    focal = .5 * W / np.tan(.5 * fov)
                elif fovaxis == 'y':
                    focal = .5 * H / np.tan(.5 * fov)
                hwf = np.array([H, W, focal], dtype=float).reshape([3, 1])

                print(f'{k}, pushed')
                pool.apply_async(find_ff_pose, (camera, xml, out_dir, k, baseline, hwf, q))
                k_list.append(k)
                if set(k_list) == set(range(num_cam)):
                    print('all image is pushed.')
                    break
                k = next_k(k, k_list, num_cam, 'right')

            elif key == ord('z'):
                k = next_k(k, k_list, num_cam, 'left')
            elif key == ord('c'):
                k = next_k(k, k_list, num_cam, 'right')

            elif key == ord('o'):
                cv2.imwrite(f'{num_cam}.jpg', (im * 255.0).astype(np.uint8))
            elif key == ord('q'):
                break
            elif key == ord('w'):
                flag = True
                print('terminating...')
                os.rmdir(out_dir)
                break
            else:
                k = next_k(k, k_list, num_cam, 'right')

        if len(k_list) != 0:
            k_list = [str(a) for a in k_list]
            f_tmp = open(out_dir + 'pushed.txt', 'w')
            f_tmp.writelines(' '.join(k_list))
            f_tmp.close()

        cv2.destroyWindow(str(num_cam))
        z += 1
        if flag:
            break
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
