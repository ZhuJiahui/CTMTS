# -*- coding: utf-8 -*-
'''
Created on 2014年7月28日

@author: ZhuJiahui506
'''
import os
import time
import numpy as np
from TextToolkit import quick_write_list_to_text, get_text_to_single_list

def get_scalability(read_directory1, read_filename1, read_filename2, read_filename3, write_filename):
    
    file_number = sum([len(files) for root, dirs, files in os.walk(read_directory1)])

    time_result = []
    
    time_list1 = []
    time_list2 = []
    time_list3 = []
    get_text_to_single_list(time_list1, read_filename1)
    get_text_to_single_list(time_list2, read_filename2)
    get_text_to_single_list(time_list3, read_filename3)
    
    temp_weibo = 0
    temp_time1 = 0.0
    temp_time2 = 0.0
    temp_time3 = 0.0
    
    count  = 1
    line_count = 0
    for i in range(424, 449):
        
        THETA = np.loadtxt(read_directory1 + '/' + str(i + 1) + '.txt')
        
        temp_weibo += len(THETA)

        temp_time1 += float(time_list1[line_count])
        temp_time2 += float(time_list2[line_count])
        temp_time3 += float(time_list3[line_count])
        
        line_count += 1
        
        if temp_weibo >= (100000 * count):
            time_result.append(str(temp_time1) + " " + str(temp_time2) + " " + str(temp_time3))
            count += 1
        
        if count > 5:
            print i
            break
    
    print time_result
    quick_write_list_to_text(time_result, write_filename)
        
        
if __name__ == '__main__':
    
    start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    read_directory1 = root_directory + u'dataset2/DCTM/mctrwctm_doc_ct20'
    read_filename1 = root_directory + u'dataset2/DCTM/mctrwctm_ct_time20.txt'
    read_filename2 = root_directory + u'dataset2/DCTM/mctrwdctm_time20.txt'
    read_filename3 = root_directory + u'dataset2/DCTM/spctm_ct_time20.txt'
    
    write_filename = root_directory + u'dataset2/DCTM/scalability20.txt'
    
    get_scalability(read_directory1, read_filename1, read_filename2, read_filename3, write_filename)
    
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
