import os
import shutil
import traceback
import logging
import os.path as osp
from multiprocessing import Pool, Manager
import time
from subprocess import check_output, STDOUT, CalledProcessError
import glob
import random

import logging, datetime
import logging.handlers
from time import sleep

if __name__ == "__main__":
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.propagate = True
    formatter = logging.Formatter("%(asctime)s;[%(levelname)s];%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    ## 그냥 처리
    # fileHandler = logging.FileHandler('./log.txt' ,mode = "w")
    streamHandler = logging.StreamHandler()
    log_max_size = 1024
    log_file_count = 5
    ## 용량별 처리
    ### log.txt에는 용량만큼 쌓고
    ### backupCount 수만큼 쌓는 것을 저장함.
    fileHandler = logging.handlers.RotatingFileHandler(filename='./log.txt',
                                                       maxBytes=log_max_size,
                                                       backupCount=log_file_count,
                                                       mode="w",
                                                      )
    ## 시간별 처리
    ### log.txt에는 when 시간 동안 쌓이고
    ### backupCount에서 그 형식의 이름으로 저장
    # fileHandler = logging.handlers.TimedRotatingFileHandler(
    #     filename='./log.txt',
    #     when="M",  # W0
    #     backupCount=4,
    #     atTime=datetime.time(0, 0, 0)
    # )
    fileHandler.setLevel(logging.DEBUG)
    streamHandler.setLevel(logging.INFO)

    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)

    idx = 0
    while True:
        log.debug('debug {}'.format(idx))
        log.info('info {}'.format(idx))
        log.warning('warning {}'.format(idx))
        log.error('error {}'.format(idx))
        log.critical('critical {}'.format(idx))
        idx += 1
        sleep(0.5)
        if idx == 1000:
            break
