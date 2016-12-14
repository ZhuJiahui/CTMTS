# -*- coding: utf-8 -*-
'''
Created on 2014年7月28日

@author: ZhuJiahui506
'''

import os
import time
import numpy as np
from TextToolkit import quick_write_list_to_text

def spctm_perplexity(read_directory1, read_directory2, read_directory3, write_filename):
    
    file_number = sum([len(files) for root, dirs, files in os.walk(read_directory1)])
    
    perp_list = []
    for i in range(424, 449):
        
        THETA = np.loadtxt(read_directory1 + '/' + str(i + 1) + '.txt')
        PHAI = np.loadtxt(read_directory2 + '/' + str(i + 1) + '.txt')
        vsm = np.loadtxt(read_directory3 + '/' + str(i + 1) + '.txt')

        #出现单个
        if len(PHAI) >= 300:
            THETA = np.array([THETA])
            THETA = THETA.T
            PHAI = np.array([PHAI])
        
        #print THETA.shape[0]
        #print THETA.shape[1]
        #print PHAI.shape[0]
        #print PHAI.shape[1]
        p_doc_word = np.dot(THETA, PHAI)
        #print p_doc_word
        #break

        
        this_perp = 0.0
        #this_nd = 0.0
        for j in range(len(p_doc_word)):
            temp_p_w = 1.0
            #temp_nd = 0
            for k in range(len(p_doc_word[0])):
                if vsm[j][k] > 0.00001 and p_doc_word[j][k] > 0.0005:
                    #temp_p_w += vsm[j][k] * np.log(p_doc_word[j][k])
                    temp_p_w = temp_p_w * (p_doc_word[j][k] ** vsm[j][k])
                
                #if vsm[j][k] > 0.00001:
                    #temp_nd += 1

                
            this_perp += np.log(temp_p_w)
            #this_nd += np.sum(vsm[j])

        perp = np.exp(-1.00 * this_perp / np.sum(vsm))
        perp_list.append(perp)
        print perp
    print np.average(perp_list)
    
    perp_list_tostring = [str(x) for x in perp_list]
    
    quick_write_list_to_text(perp_list_tostring, write_filename)

if __name__ == '__main__':
    start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    read_directory1 = root_directory + u'dataset2/DCTM/spctm_doc_ct60'
    read_directory2 = root_directory + u'dataset2/DCTM/spctm_ct_word60'
    read_directory3 = root_directory + u'dataset2/text_model/vsm'
    
    write_filename = root_directory + u'dataset2/DCTM/spctm_perplexity60.txt'
    
    spctm_perplexity(read_directory1, read_directory2, read_directory3, write_filename)
    
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
