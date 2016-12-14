# -*- coding: utf-8 -*-
'''
Created on 2014年7月27日

@author: ZhuJiahui506
'''

import os
import numpy as np
from TextToolkit import quick_write_list_to_text, write_matrix_to_text
import time
from KLD import SKLD
from TwDCTM.GetClusterNumber import get_cluster_number
from TwDCTM.SpectralCluster import spectral_cluster2
from TwDCTM.GetIR import get_IR

'''
Step 10
MCTRW-CTM
'''
def MCTRW_CTM(read_directory1, read_directory2, write_directory1, write_directory2, write_directory3, write_filename1, write_filename2):
    
    #gamma = 0.0666
    gamma = 0.001
    run_time = []
    cn_list = []
    file_number = sum([len(files) for root, dirs, files in os.walk(read_directory1)])##
    
    for i in range(424, 449):
        
        start = time.clock()
        
        THETA = np.loadtxt(read_directory1 + '/' + str(i + 1) + '.txt')
        PHAI = np.loadtxt(read_directory2 + '/' + str(i + 1) + '.txt')
        
        '''
        构建多视图
        '''
        #视图1，根据词汇分布计算潜在主题之间的相似度
        W1 = np.zeros((len(PHAI), len(PHAI)))
        for j in range(len(PHAI)):
            for k in range(j, len(PHAI)):
                W1[j, k] = 1.0 / (SKLD(PHAI[j], PHAI[k]) + 1.0)
                W1[k, j] = W1[j, k]
        
        #视图2，根据相关微博文本集合计算潜在主题之间的相似度
        W2 = np.zeros((len(PHAI), len(PHAI)))
        
        related_weibo_list = []
        for j in range(len(PHAI)):
            related_weibo_list.append([])
            
        for j in range(len(THETA)):
            for k in range(len(THETA[0])):
                if THETA[j, k] >= gamma:
                    related_weibo_list[k].append(j)
        
        for j in range(len(PHAI)):
            for k in range(j, len(PHAI)):
                numerator = len(set(related_weibo_list[j]) & set(related_weibo_list[k]))
                denominator = len(set(related_weibo_list[j]) | set(related_weibo_list[k]))
                if j == k:
                    W2[j, k] = 1.0
                    W2[k, j] = 1.0
                elif denominator == 0.0:
                    W2[j, k] = 0.01
                    W2[k, j] = 0.01
                else:
                    W2[j, k] = np.true_divide(numerator, denominator) + 0.01
                    W2[k, j] = W2[j, k]
        #print W1
        #print W2
        '''
        估计聚类数目
        '''
        cluster_number = get_cluster_number(W1)
        cn_list.append(str(cluster_number))
        print cluster_number

        '''
        获取内在关联一致性矩阵
        '''
        IRM = get_IR(W1, W2)
        
        
        '''
        简单谱聚类
        '''
        cluster_tag = spectral_cluster2(IRM, cluster_number)
        
        
        '''
        聚类分析
        '''
        center_topic = np.zeros((cluster_number, len(PHAI[0])))  #初始化
        each_cluster_number = np.zeros(cluster_number, int)  #每个聚簇中的潜在主题个数
        
        weibo_topic_similarity = np.zeros((cluster_number, len(THETA)))
        THETA = THETA.transpose()
        
        for j in range(len(cluster_tag)):
            center_topic[cluster_tag[j]] += PHAI[j]
            each_cluster_number[cluster_tag[j]] += 1
            
            weibo_topic_similarity[cluster_tag[j]] += THETA[j]
        
        for j in range(cluster_number):
            center_topic[j] = center_topic[j] / each_cluster_number[j]
        
        weibo_topic_similarity = weibo_topic_similarity.transpose()
        
        ecn_to_string = [str(x) for x in each_cluster_number]
        #time.sleep(5)
        run_time.append(str(time.clock() - start))
        print "This time:", str(time.clock() - start)
        
        write_matrix_to_text(weibo_topic_similarity, write_directory1 + '/' + str(i + 1) + '.txt')
        write_matrix_to_text(center_topic, write_directory2 + '/' + str(i + 1) + '.txt')
        quick_write_list_to_text(ecn_to_string, write_directory3 + '/' + str(i + 1) + '.txt')
        
        print "Segment %d Completed." % (i + 1)
    quick_write_list_to_text(run_time, write_filename1)
    quick_write_list_to_text(cn_list, write_filename2)

if __name__ == '__main__':
    #start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    read_directory1 = root_directory + u'dataset2/DCTM/doc_topic60'
    read_directory2 = root_directory + u'dataset2/DCTM/topic_word60'
    write_directory1 = root_directory + u'dataset2/DCTM/mctrwctm_doc_ct60'
    write_directory2 = root_directory + u'dataset2/DCTM/mctrwctm_ct_word60'
    write_directory3 = root_directory + u'dataset2/DCTM/mctrwctm_ct_number60'
    write_filename1 = root_directory + u'dataset2/DCTM/mctrwctm_ct_time60.txt'
    write_filename2 = root_directory + u'dataset2/DCTM/mctrwctm_cn60.txt'

    if (not(os.path.exists(write_directory1))):
        os.mkdir(write_directory1)
    if (not(os.path.exists(write_directory2))):
        os.mkdir(write_directory2)
    if (not(os.path.exists(write_directory3))):
        os.mkdir(write_directory3)

    
    MCTRW_CTM(read_directory1, read_directory2, write_directory1, write_directory2, write_directory3, write_filename1, write_filename2)
    
    #print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
