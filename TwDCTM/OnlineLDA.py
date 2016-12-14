# -*- coding: utf-8 -*-
'''
Created on 2014年7月23日

@author: ZhuJiahui506
'''

import os
import time
from gensim import corpora, models
import numpy as np
from TextToolkit import get_text_to_complex_list, quick_write_list_to_text


'''
Step 9
Online LDA.
'''

def online_lda(read_directory1, read_directory2, write_directory1, write_directory2, write_directory3):
    
    file_number = sum([len(files) for root, dirs, files in os.walk(read_directory1)])
    #latent_topic_number = 50
    latent_topic_number = 60
    
    for i in range(424, 449):
        each_weibo_fenci = []
        all_weibo_word = []
        
        f = open(read_directory1 + '/' + str(i + 1) + '.txt', 'rb')
        line = f.readline()
        while line:
            each_weibo_fenci.append(line.strip().split('---'))
            line = f.readline()
        f.close()
        #get_text_to_complex_list(each_weibo_fenci, read_directory1 + '/' + str(i + 1) + '.txt', 0)
        
        f = open(read_directory2 + '/' + str(i + 1) + '.txt')
        line = f.readline()
        while line:
            all_weibo_word.append([line.strip().split('---')[0]])
            line = f.readline()  
        f.close()
        
        dictionary = corpora.Dictionary(all_weibo_word)
        tf_corpus = [dictionary.doc2bow(text) for text in each_weibo_fenci]
        
        tf_corpus_to_string = []
        for each in tf_corpus:
            ss = [str(x) for x in each]
            tf_corpus_to_string.append("+".join(ss))
        
        lda = models.ldamodel.LdaModel(tf_corpus, num_topics=latent_topic_number)
        
        #获取文档-潜在主题分布矩阵
        THETA = []
        for j in range(len(tf_corpus)):
            this_line = np.zeros(latent_topic_number)
            for each1 in lda[tf_corpus[j]]:
                #each1是一个元组(topic_id, probability)
                this_line[each1[0]] = each1[1]
            
            THETA.append(" ".join([str(x) for x in this_line]))
        
        #获取潜在主题-词汇分布矩阵
        PHAI = []
        raw_topics = lda.show_topics(num_topics=latent_topic_number, formatted=False)
        for j in range(latent_topic_number):
            this_line = np.zeros(len(all_weibo_word))
            for each2 in raw_topics[j]:
                #each1是一个元组(probability, (str)topic_id)
                this_line[int(each2[1])] = each2[0]
            
            PHAI.append(" ".join([str(x) for x in this_line]))
            
        
        quick_write_list_to_text(tf_corpus_to_string, write_directory1 + '/' + str(i + 1) + '.txt')
        quick_write_list_to_text(THETA, write_directory2 + '/' + str(i + 1) + '.txt')
        quick_write_list_to_text(PHAI, write_directory3 + '/' + str(i + 1) + '.txt')
        
        print "Segment %d Completed." % (i + 1)
                
        
        
if __name__ == '__main__':
    
    start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    read_directory1 = root_directory + u'dataset2/text_model/content2'
    read_directory2 = root_directory + u'dataset2/text_model/select_words'
    write_directory = root_directory + u'dataset2/DCTM'
    #write_directory1 = root_directory + u'dataset/DCTM/tf_corpus'
    #write_directory2 = root_directory + u'dataset/DCTM/doc_topic'
    #write_directory3 = root_directory + u'dataset/DCTM/topic_word'
    write_directory1 = root_directory + u'dataset2/DCTM/tf_corpus60'
    write_directory2 = root_directory + u'dataset2/DCTM/doc_topic60'
    write_directory3 = root_directory + u'dataset2/DCTM/topic_word60'
    
    if (not(os.path.exists(write_directory))):
        os.mkdir(write_directory)
        os.mkdir(write_directory1)
        os.mkdir(write_directory2)
        os.mkdir(write_directory3)
    if (not(os.path.exists(write_directory1))):
        os.mkdir(write_directory1)
    if (not(os.path.exists(write_directory2))):
        os.mkdir(write_directory2)
    if (not(os.path.exists(write_directory3))):
        os.mkdir(write_directory3)
    
    online_lda(read_directory1, read_directory2, write_directory1, write_directory2, write_directory3)
    
    print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
    
