import os
from utils import *
from utils_geometry import *
import shutil
from termcolor import colored

# https://tempdev.tistory.com/entry/Python-multiprocessingPool-%EB%A9%80%ED%8B%B0%ED%94%84%EB%A1%9C%EC%84%B8%EC%8B%B1-2

output_root = '/home/vig-titan2/Data/OpenRooms_findpose/data/rendering/data_FF_10_640'
tmp_dir = '/home/vig-titan2/Data/OpenRooms_findpose/data/rendering/tmp'
ff_renew_root = '/home/vig-titan2/Data/@@FF_renew/'
renew_dir = ff_renew_root + 'rendering/'

num_gpu = 2
depth_to_baseline = 0.06
debug = False


def npy2xml(npy):
    a = npy.split('data_FF_10_640')[1]
    b = a.split('/')
    scene_name = b[2]
    c = b[1].split('_')
    dir_type = c[0]
    xml_type = c[1]
    idx = b[3].split('_')[0]
    cam_txt = npy.split('data_FF_10_640')[0] + 'scenes/' + xml_type + '/' + scene_name + '/' + idx + '_cam_' + dir_type + '_FF.txt'
    xml = npy.split('data_FF_10_640')[0] + 'scenes/' + xml_type + '/' + scene_name + '/' + dir_type + '_FF.xml'
    return cam_txt, xml


def main():
    dirs = sorted(glob.glob(output_root + '/*'))
    for dir in dirs:
        scenes = glob.glob(dir + '/*')
        for scene in scenes:
            files = sorted(glob.glob(scene + '/*'))
            if scene + '/pushed.txt' in files:
                f_tmp = open(scene + '/pushed.txt', 'r')
                f_tmp.seek(0)
                pushed_list = f_tmp.read().split(' ')
                f_tmp.close()
                for pushed in pushed_list:
                    files_tmp = [f for f in files if f.split('/')[-1].split('_')[0] == pushed]
                    if len(files_tmp) == 6:
                        print(f'{scene}_{[pushed]} is success, check the result')
                        if debug:
                            for file_albedo in files_tmp:
                                if file_albedo.endswith('npy'):
                                    continue
                                a = cv2.imread(file_albedo)
                                cv2.namedWindow(scene)  # Create a named window
                                cv2.moveWindow(scene, 1900, 500)  # Move it to (40,30)
                                cv2.imshow(scene, a)
                                c = cv2.waitKey(0)
                                if c == ord('q'):
                                    cv2.destroyWindow(scene)
                                    print(scene, 'find wrong ff pose')
                                    return
                            cv2.destroyWindow(scene)
                        dir_name = os.path.dirname(files_tmp[0]).split('data_FF_10_640')[1]
                        os.makedirs(renew_dir + 'data_FF_10_640/' + dir_name, exist_ok=True)
                        for file_albedo in files_tmp:
                            if file_albedo.endswith('npy'):
                                shutil.move(file_albedo, renew_dir + 'data_FF_10_640' + dir_name + '/' + os.path.basename(file_albedo))
                                cam, xml = npy2xml(file_albedo)
                                cam_xml_new_dir = os.path.dirname(cam).split('/rendering/')[1]
                                os.makedirs(renew_dir + cam_xml_new_dir, exist_ok=True)
                                shutil.move(cam, renew_dir + cam_xml_new_dir + '/' + os.path.basename(cam))
                                if not os.path.exists(renew_dir + cam_xml_new_dir + '/' + os.path.basename(xml)):
                                    shutil.copyfile(xml, renew_dir + cam_xml_new_dir + '/' + os.path.basename(xml))
                    else:
                        text = colored(f'{scene}_{[pushed]} is failed', 'red', attrs=['reverse', 'blink'])
                        print(text)

                # print(scene, ', finding pose is done.')
                # f_tmp = open(done_txt, 'a')
                # f_tmp.write(scene.split('titan2')[1])
                # f_tmp.write('\n')
                # f_tmp.close()

                # if len(files) % 6 != 1:
                #     print(scene, 'file num error!')
            else:
                print(scene, 'can not find any ff pose.')
                # f_tmp = open(worst_txt, 'a')
                # f_tmp.write(scene.split('titan2/')[1])
                # f_tmp.write('\n')
                # f_tmp.close()

            # os.makedirs(tmp_dir + scene.split('data_FF_10_640')[1], exist_ok=True)
            shutil.move(scene, tmp_dir + scene.split('data_FF_10_640')[1])


if __name__ == '__main__':
    main()
