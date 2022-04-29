# import cv2
# import glob
# from tqdm import tqdm
#
# a = open('good_scenes.txt', 'r')
# b = a.readlines()
# a.close()
#
# a = open('noisy_good_scenes.txt', 'r')
# b += a.readlines()
# a.close()
#
#
# a = [a.strip() for a in b]
# xmlFiles = a[::3]
# camFiles = a[1::3]
# outdirs = a[2::3]
#
#
#
# num=0
# for xml, cam, outdir in zip(xmlFiles, camFiles, outdirs):
#     if len(glob.glob('/home/happily' + outdir + '*')) != 73:
#         print(outdir)
#         num+=1
#


# import glob
# import os
# import random
#
# root = '/home/happily/Data/OpenRooms_org/'
# dir1 = os.listdir(root)
#
# asd = []
# for dir1_t in dir1:
#     asdd = os.listdir(root + dir1_t)
#     asdf = [f'{dir1_t}/{i}\n' for i in asdd]
#     asd += asdf
#
# random.shuffle(asd)
# test = asd[:200]
# f = open('test.txt', 'w')
# f.writelines(test)
# f.close()
# train = asd[200:]
# f = open('train.txt', 'w')
# f.writelines(train)
# f.close()
#


# import os
# import glob
#
# root = '/home/happily/Data/OpenRooms_org/'
# dir_type = os.listdir(root)
# for dir_1 in dir_type:
#     scens = os.listdir(root + dir_1)
#     for sc in scens:
#         img = glob.glob(root + dir_1 + '/' + sc + '/imDirectsgEnv_*.h5')
#         for i in img:
#             os.remove(i)
import shutil
import os

# a = open('bad_scenes.txt', 'r')
# all_scene = [b.strip().replace('/Data/OpenRoomsDataset', '/home/happily/Data/OpenRooms_FF') for b in a.readlines()]
# a.close()
# all_scene = all_scene[2::3]
#
# for scene in all_scene:
#     shutil.rmtree(scene)
#     os.makedirs(scene)

def scene2camxml(scene):
    a = scene.split('data_FF_10_640')
    b = a[1].split('/')
    scene_name = b[2]
    return a[0] + 'scenes/' + b[1].split('_')[1] + '/' + scene_name + '/cam_' + b[1].split('_')[0] + '_10.txt'


def scene2xml(scene):
    a = scene.split('data_FF_10_640')
    b = a[1].split('/')
    scene_name = b[2]
    return a[0] + 'scenes/' + b[1].split('_')[1] + '/' + scene_name + '/' + b[1].split('_')[0] + '_FF_640.xml'

def xml2scene(xml):
    split = xml.split('.')[0].split('/')
    root = xml.split('scenes')[0] + 'data_FF_10_640'
    if 'mainDiffLight' in split[-1]:
        scene_type = 'mainDiffLight'
    elif 'mainDiffMat' in split[-1]:
        scene_type = 'mainDiffMat'
    else:
        scene_type = 'main'
    scene_name = split[-2]
    xml_name = split[-3]
    return os.path.join(root, scene_type + '_' + xml_name, scene_name)

def camtxt2scene(camtxt):
    split = camtxt.split('.')[0].split('/')
    root = camtxt.split('scenes')[0] + 'data_FF_10_640'
    if 'mainDiffLight' in split[-1]:
        scene_type = 'mainDiffLight'
    elif 'mainDiffMat' in split[-1]:
        scene_type = 'mainDiffMat'
    else:
        scene_type = 'main'
    scene_name = split[-2]
    xml_name = split[-3]
    return os.path.join(root, scene_type + '_' + xml_name, scene_name)

import os

root = '/home/happily/Data/OpenRooms_FF/data/rendering/scenes/xml/'
dd = os.listdir(root)

# f = open('remain.txt', 'w')
count = 0
for d in dd:
#     scenes = os.listdir(root + d)
#     scenes = [root + d + '/' + f for f in scenes]
#     f.writelines('\n'.join(scenes))
# f.close()

    scene_full_name = root + d
    files = os.listdir(scene_full_name)
    for f in files:
        if f == 'cam.txt':
            continue
        if f.split('_')[0] == '@':
            count += 1

print(count)
        # else:
        #     if f.split('.')[-1] == 'txt':
        #         scene = camtxt2scene(scene_full_name + '/' + f)
        #         if os.path.exists(scene):
        #             f_new = f.replace('cam', '@_cam')
        #             os.rename(scene_full_name + '/' + f, scene_full_name + '/' + f_new)
        #         else:
        #             os.remove(scene_full_name + '/' + f)
        #     elif f.split('.')[-1] == 'xml':
        #         scene = xml2scene(scene_full_name + '/' + f)
        #         if os.path.exists(scene):
        #             if len(f.split('_')) == 3:
        #                 f_new = f.replace('main', '@_main').replace('_640', '')
        #                 os.rename(scene_full_name + '/' + f, scene_full_name + '/' + f_new)
        #             else:
        #                 os.remove(scene_full_name + '/' + f)
        #         else:
        #             os.remove(scene_full_name + '/' + f)
        #
        #     else:
        #         print('what file?', f)

    # scenes = os.listdir(root + d)
    # for s in scenes:
    #     scene_full_name = root + d + '/' + s
    #
        # files = os.listdir(scene_full_name)
        # for f in files:
        #     f_new = f.replace('poses_bound', '@_poses_bound')
        #     os.rename(f'{scene_full_name}/{f}', f'{scene_full_name}/{f_new}')


