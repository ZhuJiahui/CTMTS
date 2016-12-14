# -*- coding: utf-8 -*-
'''
Created on 2015年5月5日

@author: ZhuJiahui506
'''

import os
import time
import numpy as np
from TextToolkit import quick_write_list_to_text


def tw_to_ltm_curpus(read_directory1, read_directory2, write_directory):
    '''
    
    :param read_directory1:
    :param read_directory2:
    :param write_directory:
    '''

    file_number = np.sum([len(files) for root, dirs, files in os.walk(read_directory1)])
    
    for i in range(file_number):
        
        all_weibo_fenci = []
        word_space_dict = {}
        word_count = 0
        
        merged_word = []
        
        f = open(read_directory2 + '/' + str(i + 1) + '.txt')
        line = f.readline()
        while line:
            this_word = line.strip().split('---')[0]
            all_weibo_fenci.append(this_word)
            word_space_dict[this_word] = str(word_count)
            
            merged_word.append(str(word_count) + ":" + "+".join(this_word.split()))
            
            word_count += 1
            line = f.readline()  
        f.close()
        
        corpus = []
        
        if len(all_weibo_fenci) >= 500:
            f = open(read_directory1 + '/' + str(i + 1) + '.txt')
            line = f.readline()
            while line:
                this_line = line.strip().split("---")
                cor_line = []
                for each in this_line:
                    word_entity = each.split(',')[0]
                    try:
                        cor_line.append(word_space_dict[word_entity])
                    except:
                        pass
                if len(cor_line) >= 1:
                    corpus.append(" ".join(cor_line))

                line = f.readline()  
            f.close()
        else:
            pass
        
        twrite_directory = write_directory + '/' + str(i + 1)
        if (not(os.path.exists(twrite_directory))):
            os.mkdir(twrite_directory)
        
        quick_write_list_to_text(corpus, twrite_directory + '/' + str(i + 1) + '.docs')
        quick_write_list_to_text(merged_word, twrite_directory + '/' + str(i + 1) + '.vocab')
        
        print "Segment %d Completed." % (i + 1)

if __name__ == "__main__":

    start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'

    read_directory1 = root_directory + u'dataset2/text_model/content1'
    read_directory2 = root_directory + u'dataset2/text_model/select_words'
    write_directory = root_directory + u'dataset2/text_model/gklda_corpus'
    
    if (not(os.path.exists(write_directory))):
        os.mkdir(write_directory)
    
    tw_to_ltm_curpus(read_directory1, read_directory2, write_directory)
    
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'