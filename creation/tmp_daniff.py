from utils import *
from subprocess import check_output, STDOUT, CalledProcessError
import os
import glob
import os.path as osp

root = '/Data/OpenRoomsDataset/data/rendering/data_FF_10_640'
renderer_directlight = '/Lib/OptixRendererDirectLight/src/bin/optixRenderer'

f = open('creation/train.txt', 'r')
outdir = f.readlines()
f.close()

outdir_list = [osp.join(root, a.strip()) for a in outdir]
xml_list = [outdir2xml(a) for a in outdir_list]

for outdir, xml in zip(outdir_list, xml_list):
    k = osp.basename(outdir).split('_')[0]
    os.makedirs(osp.dirname(outdir), exist_ok=True)
    cam_file_name = xml2camtxt(xml, k)
    optixrenderer_arg = [renderer_directlight, '-f', xml, '-c', cam_file_name, '-o', osp.join(outdir, f'im'), '-m', '7', '--imWidth', '40',
                         '--imHeight', '30', '--maxPathLength', '2 ', '--gpuIds', str(gpu_id), '--forceOutput']
    output = check_output(optixrenderer_arg, stderr=STDOUT, universal_newlines=True)
