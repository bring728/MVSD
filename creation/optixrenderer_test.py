import os
import shutil
import traceback
import logging
import os.path as osp
from multiprocessing import Pool, Manager
import time
from subprocess import check_output, STDOUT, CalledProcessError
from utils import *

num_p = 2
renderer = '/home/vig-titan2/Downloads/OptixRenderer-master/build/bin/optixRenderer'


def work(xml, camtxt, outdir):
    optixrenderer_arg = [renderer, '-f', xml, '-c', camtxt, '-o', outdir, '-m', '0', '--gpuIds', str(0), '--forceOutput']
    output = check_output(optixrenderer_arg, stderr=STDOUT, universal_newlines=True)


logger = logging.getLogger()
if __name__ == "__main__":

    mask = 0.5 * (loadImage('/home/vig-titan2/Data/OpenRooms_findpose/data/rendering/data_FF_10_640/mainDiffLight_xml1/scene0110_02/0/immask_1.png') + 1)[0:1, ...]

    segall = (mask > 0.45)
    bool_tmp = not np.percentile(mask, 30.0) < 1
    if bool_tmp:
        depth = loadBinary('/home/vig-titan2/Data/OpenRooms_findpose/data/rendering/data_FF_10_640/mainDiffLight_xml1/scene0110_02/0/imdepth_1.dat')
        depth[~segall] = np.nan
        min = np.percentile(depth * segall, 5.0)


    # pool = Pool(processes=1)
    pool = Pool(processes=2)

    start = time.time()
    xml = '/home/vig-titan2/Data/OpenRooms_findpose/data/rendering/scenes/xml_tmp/scene0009_00/main_FF.xml'
    cam = '/home/vig-titan2/Data/OpenRooms_findpose/data/rendering/scenes/xml_tmp/scene0009_00/cam.txt'
    outdir = '/home/vig-titan2/Data/OpenRooms_findpose/data/rendering/scenes/xml_tmp/scene0009_00/im'
    pool.apply_async(work, (xml, cam, outdir))

    xml = '/home/vig-titan2/Data/OpenRooms_findpose/data/rendering/scenes/xml_tmp/scene0001_00/main_FF.xml'
    cam = '/home/vig-titan2/Data/OpenRooms_findpose/data/rendering/scenes/xml_tmp/scene0001_00/cam.txt'
    outdir = '/home/vig-titan2/Data/OpenRooms_findpose/data/rendering/scenes/xml_tmp/scene0001_00/im'
    pool.apply_async(work, (xml, cam, outdir))

    pool.close()
    pool.join()

    print(time.time() - start)

