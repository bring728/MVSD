import os
import glob
import os.path as osp

datapath = '/home/happily/Data/OpenRooms_FF/'
if __name__ == '__main__':
    all_scene = []
    mvs_list = []
    midas_list = []
    env_direct_list = []
    scene_type_list = glob.glob(osp.join(datapath, '*'))
    for scene_type in scene_type_list:
        scenes = sorted(glob.glob(osp.join(scene_type, '*')))
        for scene in scenes:
            ff_list = os.listdir(scene)
            ff_list = sorted(list(set([f.split('_')[0] for f in ff_list])))
            for k in ff_list:
                if len(glob.glob(osp.join(scene, f'{k}_*.rgbe'))) != 9 or len(glob.glob(osp.join(scene, f'{k}_*.hdr'))) != 9 or len(
                        glob.glob(osp.join(scene, f'{k}_imdepth_*.dat'))) != 9 or len(glob.glob(osp.join(scene, f'{k}_*.png'))) != 36:
                    # env_direct_list.append((osp.join(scene, k) + '_').split('OpenRooms_FF/')[1])
                    raise Exception('file num error', scene, k)
                # for pfm in glob.glob(osp.join(scene, f'{k}_conf*.dat')):
                #     os.remove(pfm)
                for pfm in glob.glob(osp.join(scene, f'{k}_depthvis*')):
                    os.remove(pfm)
                # for pfm in glob.glob(osp.join(scene, f'{k}_imvis*.jpg')):
                #     os.remove(pfm)
                # for pfm in glob.glob(osp.join(scene, f'{k}_depthest*.dat')):
                #     os.remove(pfm)
                # for pfm in glob.glob(osp.join(scene, f'{k}_depthnormmax*.dat')):
                #     os.rename(pfm, pfm.replace('max', ''))
                for pfm in glob.glob(osp.join(scene, f'{k}_depthnorm*.dat')):
                    os.remove(pfm)
                for pfm in glob.glob(osp.join(scene, f'{k}_poses_bounds.npy')):
                    os.remove(pfm)
