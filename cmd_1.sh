#!/bin/bas#!/bin/bash

python3 ~/PycharmProjects/MPR_openrooms/train_MPI_DDP.py '0,1,2,3,4,5,6,7' 'initbrdf_1.yml' 3456 &
#CUDA_VISIBLE_DEVICES=2,3 python3 ~/PycharmProjects/MPR_openrooms/train_MPI_DDP.py 2 'MPI_2.yml' 3457 &
#wait

#python3 ~/PycharmProjects/MPR_openrooms/train_MPI.py 0 'MPI_0.yml' &
#python3 ~/PycharmProjects/MPR_openrooms/train_MPI.py 1 'MPI_1.yml' &
#python3 ~/PycharmProjects/MPR_openrooms/train_MPI.py 2 'MPI_2.yml' &
#python3 ~/PycharmProjects/MPR_openrooms/train_MPI.py 3 'MPI_3.yml' &
#wait

#python3 ~/PycharmProjects/MPR_openrooms/test_MPI.py 0 'MPI_0.yml' &
#python3 ~/PycharmProjects/MPR_openrooms/test_MPI.py 1 'MPI_1.yml' &
#python3 ~/PycharmProjects/MPR_openrooms/test_MPI.py 2 'MPI_2.yml' &
#python3 ~/PycharmProjects/MPR_openrooms/test_MPI.py 3 'MPI_3.yml' &
#wait

#python3 ~/PycharmProjects/MPR_openrooms/test_MPI_novelview.py 0 'MPI_2.yml' 234 &
#python3 ~/PycharmProjects/MPR_openrooms/test_MPI_novelview.py 1 'MPI_2.yml' 256 &
#python3 ~/PycharmProjects/MPR_openrooms/test_MPI_novelview.py 2 'MPI_2.yml' 278 &
#python3 ~/PycharmProjects/MPR_openrooms/test_MPI_novelview.py 3 'MPI_2.yml' 279 &
#python3 ~/PycharmProjects/MPR_openrooms/test_MPI_novelview.py 3 'MPI_3.yml' &
#wait

#output_txt="10_rendered.txt"
#touch $output_txt
#IFS=$"\n"
#for ((id=0;id<$numproc;id++));
#do
#  cat 10_rendered$id.txt >> $output_txt
#doneh

#CUDA_VISIBLE_DEVICES=0,1 python3 ~/PycharmProjects/MPR_openrooms/train_MPI_DDP.py 2 'MPI_1.yml' 3456 &
#CUDA_VISIBLE_DEVICES=2,3 python3 ~/PycharmProjects/MPR_openrooms/train_MPI_DDP.py 2 'MPI_2.yml' 3457 &
#wait

#python3 ~/PycharmProjects/MPR_openrooms/train_MPI.py 0 'MPI_0.yml' &
#python3 ~/PycharmProjects/MPR_openrooms/train_MPI.py 1 'MPI_1.yml' &
#python3 ~/PycharmProjects/MPR_openrooms/train_MPI.py 2 'MPI_2.yml' &
#python3 ~/PycharmProjects/MPR_openrooms/train_MPI.py 3 'MPI_3.yml' &
#wait

#python3 ~/PycharmProjects/MPR_openrooms/test_MPI.py 0 'MPI_0.yml' &
#python3 ~/PycharmProjects/MPR_openrooms/test_MPI.py 1 'MPI_1.yml' &
#python3 ~/PycharmProjects/MPR_openrooms/test_MPI.py 2 'MPI_2.yml' &
#python3 ~/PycharmProjects/MPR_openrooms/test_MPI.py 3 'MPI_3.yml' &
#wait

python3 ~/PycharmProjects/MPR_openrooms/test_MPI_novelview.py 0 'MPI_2.yml' 234 &
python3 ~/PycharmProjects/MPR_openrooms/test_MPI_novelview.py 1 'MPI_2.yml' 256 &
python3 ~/PycharmProjects/MPR_openrooms/test_MPI_novelview.py 2 'MPI_2.yml' 278 &
python3 ~/PycharmProjects/MPR_openrooms/test_MPI_novelview.py 3 'MPI_2.yml' 279 &
#python3 ~/PycharmProjects/MPR_openrooms/test_MPI_novelview.py 3 'MPI_3.yml' &
wait

#output_txt="10_rendered.txt"
#touch $output_txt
#IFS=$"\n"
#for ((id=0;id<$numproc;id++));
#do
#  cat 10_rendered$id.txt >> $output_txt
#done