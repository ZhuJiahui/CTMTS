# -*- coding: utf-8 -*-
'''
Created on 2014年7月27日

@author: ZhuJiahui506
'''


import os
import numpy as np
from TextToolkit import quick_write_list_to_text
import time
from operator import itemgetter


'''
Step 11
Text Clustering.
'''
def get_ctm_real_topics(read_directory1, read_directory2, write_directory):
    
    gamma = 0.01
    file_number = sum([len(files) for root, dirs, files in os.walk(read_directory1)])
    
    for i in range(424, 449):
        
        PHAI = np.loadtxt(read_directory1 + '/' + str(i + 1) + '.txt')
        
        # 本片数据的词汇列表
        this_word_list = []
        f1 = open(read_directory2 + '/' + str(i + 1) + '.txt', 'rb')
        line = f1.readline()
        while line:
            this_word_list.append(line.split('---')[0])
            line = f1.readline()
        
        f1.close()
        
        #出现单个
        if len(PHAI) >= 200:
            PHAI = np.array([PHAI])
        
        real_topics = []
        for j in range(len(PHAI)):
            this_topic = []
            this_topic_weight = []
            for k in range(len(PHAI[j])):
                if PHAI[j][k] > gamma:
                    this_topic.append(this_word_list[k])
                    this_topic_weight.append(PHAI[j][k])
            
            tt = zip(this_topic, this_topic_weight)
            tt = sorted(tt, key = itemgetter(1), reverse=True)
            this_topic = []
            for each in tt:
                this_topic.append(str(each[1]) + '*' + str(each[0]))
            
            real_topics.append(" ".join(this_topic))

        quick_write_list_to_text(real_topics, write_directory + '/' + str(i + 1) + '.txt')
        
        print "Segment %d Completed." % (i + 1)


if __name__ == '__main__':
    start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    write_directory = root_directory + u'dataset2/DCTM/mctrwdctm_real_topics'
    
    #read_directory1 = root_directory + u'dataset2/DCTM/topic_word20'
    read_directory1 = root_directory + u'dataset2/DCTM/mctrwdctm_ct_word20'
    read_directory2 = root_directory + u'dataset2/text_model/select_words'

    if (not(os.path.exists(write_directory))):
        os.mkdir(write_directory)

    get_ctm_real_topics(read_directory1, read_directory2, write_directory)
    
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
