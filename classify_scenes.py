import cv2
from tqdm import tqdm
txt = open('text_scenes/bad_scenes.txt', 'r')
a = txt.readlines()
txt.close()

a = a[900:]
a = [b.strip() for b in a]
xmlFiles = a[::3]
camFiles = a[1::3]
outdirs = a[2::3]
#
# c = []
# a = open('bad_scenes.txt', 'r')
# b = a.readlines()
# a.close()
# last = b[-1]
# ind = outdirs.index(last.strip())
# c.append(ind)
#
# a = open('good_scenes.txt', 'r')
# b = a.readlines()
# a.close()
# last = b[-1]
# ind = outdirs.index(last.strip())
# c.append(ind)
#
# a = open('noisy_bad_scenes.txt', 'r')
# b = a.readlines()
# a.close()
# last = b[-1]
# ind = outdirs.index(last.strip())
# c.append(ind)
#
# a = open('noisy_good_scenes.txt', 'r')
# b = a.readlines()
# a.close()
# last = b[-1]
# ind = outdirs.index(last.strip())
# c.append(ind)
#
# ind = max(c)
# xmlFiles = xmlFiles[ind+1:]
# camFiles = camFiles[ind+1:]
# outdirs = outdirs[ind+1:]
# step = ind + 1
step = 0
for xml, cam, outdir in zip(xmlFiles, camFiles, outdirs):
    print(step)
    outdir = outdir.replace('OpenRoomsDataset', 'OpenRooms_FF')
    img = cv2.imread('/home/happily/' + outdir + 'imvis_5.jpg')
    cv2.imshow('asd', img)
    c = cv2.waitKey(0)
    # if c == ord('a'):
    #     f = open('2_good.txt', 'a')
    #     f.write(xml)
    #     f.write('\n')
    #     f.write(cam)
    #     f.write('\n')
    #     f.write(outdir)
    #     f.write('\n')
    #     f.close()
    # elif c == ord('s'):
    #     f = open('2_noisy.txt', 'a')
    #     f.write(xml)
    #     f.write('\n')
    #     f.write(cam)
    #     f.write('\n')
    #     f.write(outdir)
    #     f.write('\n')
    #     f.close()
    # elif c == ord('d'):
    #     f = open('2_bad.txt', 'a')
    #     f.write(xml)
    #     f.write('\n')
    #     f.write(cam)
    #     f.write('\n')
    #     f.write(outdir)
    #     f.write('\n')
    #     f.close()
    step+=1



# import os
# from utils import *
# import cv2
# from tqdm import tqdm
#
# root_lemon = '/home/happily/Data/noisy/'
# root_org = '/home/happily/Data/OpenRooms_FF/data/rendering/data_FF_10_640/'
# dir_list = os.listdir(root_lemon)
# for dir in dir_list:
#     scene_list = os.listdir(root_lemon + dir)
#     for scene in scene_list:
#         im_lemon_name = root_lemon + dir + '/' + scene + '/im_1.rgbe'
#         seg_name = root_org + dir + '/' + scene + '/immask_1.png'
#         im_name = root_org + dir + '/' + scene + '/im_1.rgbe'
#
#         im = loadHdr(im_name)
#         seg = 0.5 * (loadImage(seg_name) + 1)[0:1, :, :]
#         scale = get_hdr_scale(im, seg, 'TEST')
#         im_org = img_rgb2bgr(np.clip(im * scale, 0, 1.0)).transpose((1,2,0))
#
#         im_org = (cv2.imread(im_name.replace('im','imvis').replace('rgbe','jpg')) / 255.0).astype(np.float32)
#
#         im = loadHdr(im_lemon_name)
#         scale = get_hdr_scale(im, seg, 'TEST')
#         im_lemon = img_rgb2bgr(np.clip(im * scale, 0, 1.0)).transpose((1,2,0)) ** (1.0 / 2.2)
#         img = cv2.hconcat([im_org, im_lemon])
#         cv2.imshow('asd', img)
#         cv2.waitKey(0)
