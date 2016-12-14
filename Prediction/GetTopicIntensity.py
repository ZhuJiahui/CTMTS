# -*- coding: utf-8 -*-
'''
Created on 2014年7月27日

@author: ZhuJiahui506
'''

import os
import numpy as np
from TextToolkit import quick_write_list_to_text
import time


def get_topic_intensity(read_directory,read_directory2, write_directory):
    
    file_number = sum([len(files) for root, dirs, files in os.walk(read_directory)])
    
    for i in range(file_number):
        
        THETA = np.loadtxt(read_directory + '/' + str(i + 1) + '.txt')
        PHAI = np.loadtxt(read_directory2 + '/' + str(i + 1) + '.txt')
        
        #出现单个
        if len(PHAI) >= 300:
            THETA = np.array([THETA])
            THETA = THETA.T
        
        each_cluster_number = []
        for j in range(len(THETA[0])):
            each_cluster_number.append(0)
            
        for j in range(len(THETA)):
            max_index = np.argmax(THETA[j])
            
            each_cluster_number[max_index] += 1
        
        
        topic_intensity = [(float(x) / len(THETA)) for x in each_cluster_number]
        topic_intensity_tostring = [str(x) for x in topic_intensity]
        
        quick_write_list_to_text(topic_intensity_tostring, write_directory + '/' + str(i + 1) + '.txt')
        
        print "Segment %d Completed." % (i + 1)
        

if __name__ == '__main__':
    start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    write_directory1 = root_directory + u'dataset/prediction'
    write_directory = root_directory + u'dataset/prediction/topic_intensity'
    
    read_directory = root_directory + u'dataset/DCTM/mctrwctm_doc_ct'
    read_directory2 = root_directory + u'dataset/DCTM/mctrwctm_ct_word'

    if (not(os.path.exists(write_directory1))):
        os.mkdir(write_directory1)
        os.mkdir(write_directory)
    
    if (not(os.path.exists(write_directory))):
        os.mkdir(write_directory)

    get_topic_intensity(read_directory, read_directory2, write_directory)
    
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'

    