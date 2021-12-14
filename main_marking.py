# system packages
import os
# 3-party packages
import numpy as np
from matplotlib import pyplot as plt
# custom packages
from libs.signals import interpolation_1d
from libs.signal_search import bio_signal_marker
from libs.read_data import read_edf_file


if __name__ == '__main__':
    ## data frame load
    data_frame = read_edf_file('BOGN00003',
                               data_path='G:\data\public_data\Stanford Technology Analytics and Genomics in Sleep\edf',
                               label_path='G:\data\public_data\Stanford Technology Analytics and Genomics in Sleep\label')
    ## load abdo
    abdo = data_frame['signal']['ABDM']
    abdo_fs = data_frame['info']['sampling_rate_sig']['ABDM']
    ## interpolate noise label
    annotation = np.zeros([len(abdo), ])
    ## data visuallization
    ts = range(len(abdo))
    data_visuallization_object = bio_signal_marker(ts, abdo, abdo, annotation,
                                                   abdo_fs, wheel_sec=100, screen_sec=300)
    data_visuallization_object.run()
