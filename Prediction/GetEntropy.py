# -*- coding: utf-8 -*-
'''
Created on 2014年8月3日

@author: ZhuJiahui506
'''

import os
import numpy as np
from TextToolkit import quick_write_list_to_text
import time


def get_topic_entropy(read_directory, write_filename):
    
    file_number = sum([len(files) for root, dirs, files in os.walk(read_directory)])
    
    all_e = []
    
    for i in range(file_number):
        
        PHAI = np.loadtxt(read_directory + '/' + str(i + 1) + '.txt')
        
        #出现单个
        if len(PHAI) >= 300:
            PHAI = np.array([PHAI])
        
        this_e_list = []
        for j in range(len(PHAI)):
            temp_e = 0.0
            for k in range(len(PHAI[j])):
                if PHAI[j][k] > 0.00001:
                    temp_e += (-1.0 * PHAI[j][k] * np.log2(PHAI[j][k]))
            
            this_e_list.append(temp_e)
        
        all_e.append(str(np.average(this_e_list)))
    
    quick_write_list_to_text(all_e, write_filename)

        

if __name__ == '__main__':
    start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    write_directory1 = root_directory + u'dataset/prediction'
    write_filename = write_directory1 + u'/topic_entropy.txt'
    
    read_directory = root_directory + u'dataset/DCTM/mctrwctm_ct_word'

    if (not(os.path.exists(write_directory1))):
        os.mkdir(write_directory1)

    get_topic_entropy(read_directory, write_filename)
    
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'

    