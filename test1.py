# -*- coding: utf-8 -*-
'''
Created on 2014年7月22日

@author: ZhuJiahui506
'''
import os
import sys
from TextToolkit import quick_write_list_to_text

if __name__ == '__main__':
    #listfile = os.listdir("dataset/original_data")
    result = []
    f = open(u"dataset/original_data/E6-德国.txt", 'rb')
    line = f.readline()
    count = 1
    while line:
        result.append(line.strip())
        
        
        if len(line.strip().split('\t')) != 7:
            print count
            break
        
        line = f.readline()
        count += 1
    f.close()
    
    print "Complete"
    
    quick_write_list_to_text(result, '0001.txt')
    