import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, train=True):
        super(Benchmark, self).__init__(args, train, benchmark=True)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]

        # print(self.scale)
        if self.scale == [2]:
            type = 'x2'
        elif self.scale == [4]:
            type = 'x4'
        else:
            type = 'x8'

        for entry in os.scandir(self.dir_hr+'/'+type):

            # filename = os.path.splitext(entry.name)[0]
            entry = str(entry)
            entry = entry.split('<DirEntry \'')[1]
            filename = entry.split(type)[0]

            hrname = type+'/'+filename+type


            list_hr.append(os.path.join(self.dir_hr, hrname + self.ext))
            # print(os.path.join(self.dir_hr, hrname + self.ext))

            filename = filename.replace('HR', 'LRBI')
            for si, s in enumerate(self.scale):

                list_lr[si].append(os.path.join(
                    self.dir_lr,
                    'x{}/{}x{}{}'.format(s, filename, s, self.ext)
                ))
        #
        # for entry in os.scandir(self.dir_hr+type+'/'):
        #     filename = os.path.splitext(entry.name)[0]
        #     list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
        #     for si, s in enumerate(self.scale):
        #         list_lr[si].append(os.path.join(
        #             self.dir_lr,
        #             'X{}/{}x{}{}'.format(s, filename, s, self.ext)
        #         ))

        list_hr.sort()
        for l in list_lr:
            l.sort()

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        # print(dir_data)
        dir_data = '/export/liuzhe/program2/RCAN_test/RCAN_TestCode/HR/Set5'

        self.apath = os.path.join(dir_data, 'benchmark', self.args.data_test)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        # print(self.dir_hr)
        self.dir_hr = dir_data
        self.dir_lr = '/export/liuzhe/program2/RCAN_test/RCAN_TestCode/LR/LRBI/Set5/'
        self.ext = '.png'

