#!/bin/bas#!/bin/bash

#python3 ~/PycharmProjects/MPR_openrooms/train_MPI_DDP.py '0,1,2,3,4,5,6,7' 'initbrdf_1.yml' 3456 &
#CUDA_VISIBLE_DEVICES=2,3 python3 ~/PycharmProjects/MPR_openrooms/train_MPI_DDP.py 2 'MPI_2.yml' 3457 &
#wait


python3 ~/PycharmProjects/MVSD/train_DDP_wrapper.py 2 'stage1-1_2.yml'
wait

python3 ~/PycharmProjects/MVSD/train_DDP_wrapper.py 2 'stage1-2_0.yml'
wait

#output_txt="10_rendered.txt"
#touch $output_txt
#IFS=$"\n"
#for ((id=0;id<$numproc;id++));
#do
#  cat 10_rendered$id.txt >> $output_txt
#done