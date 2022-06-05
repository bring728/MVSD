#!/bin/bas#!/bin/bash

python3 ~/PycharmProjects/MVSD/train_DDP_wrapper.py 8 'stage1-1_0.yml'
wait

python3 ~/PycharmProjects/MVSD/train_DDP_wrapper.py 8 'stage1-1_1.yml'
wait

python3 ~/PycharmProjects/MVSD/train_DDP_wrapper.py 8 'stage1-1_2.yml'
wait



#output_txt="10_rendered.txt"
#touch $output_txt
#IFS=$"\n"
#for ((id=0;id<$numproc;id++));
#do
#  cat 10_rendered$id.txt >> $output_txt
#done