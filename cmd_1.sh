#!/bin/bas#!/bin/bash

#python3 ~/PycharmProjects/MPR_openrooms/train_MPI_DDP.py '0,1,2,3,4,5,6,7' 'initbrdf_1.yml' 3456 &
#CUDA_VISIBLE_DEVICES=2,3 python3 ~/PycharmProjects/MPR_openrooms/train_MPI_DDP.py 2 'MPI_2.yml' 3457 &
#wait

python3 ~/PycharmProjects/MVSD/eval_stage_1.py 0 'stage1_0.yml' &
python3 ~/PycharmProjects/MVSD/eval_stage_1.py 1 'stage1_1.yml' &
python3 ~/PycharmProjects/MVSD/eval_stage_1.py 2 'stage1_2.yml' &
python3 ~/PycharmProjects/MVSD/eval_stage_1.py 3 'stage1_3.yml' &
python3 ~/PycharmProjects/MVSD/eval_stage_1.py 4 'stage1_4.yml' &
python3 ~/PycharmProjects/MVSD/eval_stage_1.py 5 'stage1_5.yml' &
python3 ~/PycharmProjects/MVSD/eval_stage_1.py 6 'stage1_6.yml' &
python3 ~/PycharmProjects/MVSD/eval_stage_1.py 7 'stage1_7.yml' &
wait

#output_txt="10_rendered.txt"
#touch $output_txt
#IFS=$"\n"
#for ((id=0;id<$numproc;id++));
#do
#  cat 10_rendered$id.txt >> $output_txt
#done