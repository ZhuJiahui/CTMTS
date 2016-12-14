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
from TwDCTM.GetIR import get_IR
from TwDCTM.SpectralCluster import spectral_cluster2, spectral_cluster


'''
Step 10
DCTM
'''

def merge_space(word_list1, word_list2, p1, p2):
    new_word_list = word_list2
    for word in set(word_list1).difference(new_word_list):
        new_word_list.append(word)
    
    new_p2 = np.zeros((len(p2), len(new_word_list)))
    new_p2[:, 0 : len(p2[0])] = p2
    
    new_p1 = np.zeros((len(p1), len(new_word_list)))

    for j in range(len(p1)):
        p1_dict = {}
        for k in range(len(p1[j])):
            if p1[j][k] > 0.0001:
                p1_dict[word_list1[k]] = p1[j][k]
                
        p1_dict2 = {}
        for each1 in new_word_list:
            if each1 in p1_dict.keys():
                p1_dict2[each1] = p1_dict[each1]
            else:
                p1_dict2[each1] = 0
        
        for k in range(len(new_word_list)):
            new_p1[j, k] = p1_dict2[new_word_list[k]]
            
    return new_p1, new_p2, new_word_list
            

def MCTRW_DCTM(read_directory1, read_directory2, read_directory3, write_directory1, write_directory2, write_directory3, write_filename):
    
    #gamma = 0.0666
    gamma = 0.001
    s_lambda = 0.6
    
    tci = 0.0  #主题上下文相关系数
    
    run_time = []
    
    file_number = sum([len(files) for root, dirs, files in os.walk(read_directory1)])
    
    #随机初始化
    last_central_topic = []
    last_each_cluster_number = []
    last_word_list = []
    
    for i in range(424, 449):
        
        '''
        预备工作
        '''

        start = time.clock()
        
        THETA = np.loadtxt(read_directory1 + '/' + str(i + 1) + '.txt')
        PHAI = np.loadtxt(read_directory2 + '/' + str(i + 1) + '.txt')
        
        this_lt_number = len(PHAI)
        
        # 视图1 根据词汇分布计算潜在主题之间的相似度
        W1 = np.zeros((len(PHAI), len(PHAI)))
        for j in range(len(PHAI)):
            for k in range(j, len(PHAI)):
                W1[j, k] = 1.0 / (SKLD(PHAI[j], PHAI[k]) + 1.0)
                W1[k, j] = W1[j, k]
        
        # 估计聚类数目
        cluster_number = get_cluster_number(W1)
        print cluster_number
        
        # 本片数据的词汇列表
        this_word_list = []
        f1 = open(read_directory3 + '/' + str(i + 1) + '.txt', 'rb')
        line = f1.readline()
        while line:
            this_word_list.append(line.split('---')[0])
            line = f1.readline()
        
        f1.close()
        
        #start = time.clock()

        #分情况1
        if i <= (424 + 3) or tci < 0.8: 
            print 'case1'
        
            # 视图2，根据相关微博文本集合计算潜在主题之间的相似度
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
            
            '''
            获取内在关联一致性矩阵并聚类
            '''
            IRM = get_IR(W1, W2)
            cluster_tag = spectral_cluster2(IRM, cluster_number)
            
            # 聚类分析
            center_topic = np.zeros((cluster_number, len(PHAI[0])))
            each_cluster_number = np.zeros(cluster_number, int)
        
            weibo_topic_similarity = np.zeros((cluster_number, len(THETA)))
            THETA = THETA.transpose()
            
            for j in range(len(cluster_tag)):
                center_topic[cluster_tag[j]] += PHAI[j]
                each_cluster_number[cluster_tag[j]] += 1
            
                weibo_topic_similarity[cluster_tag[j]] += THETA[j]

            for j in range(cluster_number):
                center_topic[j] = center_topic[j] / each_cluster_number[j]
        
            weibo_topic_similarity = weibo_topic_similarity.transpose()
            
            tci = 1.0
            
        #分情况2
        else:
            # 回溯一个数据片
            # 初始化中心主题
            
            print 'case2'
            init_central_topic = np.zeros((cluster_number, len(PHAI[0])))
            
            ##########
            if len(last_central_topic[0]) >= cluster_number:
                print 'case21'
                idx = last_each_cluster_number[0].argsort()
                idx = idx[::-1]
                
                #按强度选
                init_central_topic = last_central_topic[0][idx][0 : cluster_number, :]
                
                # 合并向量空间
                merge_init_ct, merge_this_lt, new_word_list = merge_space(last_word_list[0], this_word_list, init_central_topic, PHAI)
            
            
                #计算当前潜在主题与前一片的中心主题之间的相似度
                lt_ct_similarity = np.zeros((this_lt_number, cluster_number));
                for j in range(this_lt_number):
                    for k in range(cluster_number):
                        lt_ct_similarity[j, k] = 1.0 / (SKLD(merge_this_lt[j], merge_init_ct[k]) + 1.0)
                
                #print lt_ct_similarity
                cluster_tag = []
                new_part_lt = []  #原空间(500维)下的本数据片的新出现的潜在主题
                last_part_lt = []  #原空间下的与上一数据片中的中心主题比较相似的潜在主题,二维
                
                for j in range(cluster_number):
                    last_part_lt.append([])
                
                for j in range(this_lt_number):
                    if np.max(lt_ct_similarity[j]) < s_lambda:
                        #新的潜在主题
                        new_part_lt.append(PHAI[j])
                        cluster_tag.append(-1)  #新类编号
                    else:
                        #是旧的主题的延续
                        max_index = np.argmax(lt_ct_similarity[j])
                        last_part_lt[max_index].append(PHAI[j])
                        cluster_tag.append(max_index)
                          
                empty_count = 0
                this_last_ct = []
                this_last_ct_count = []
                
                for j in range(cluster_number):
                
                    if len(last_part_lt[j]) == 0:
                        empty_count += 1
                    else:
                        temp_this_ct = np.zeros(len(PHAI[0]))
                        
                        for k in range(len(last_part_lt[j])):
                            temp_this_ct += last_part_lt[j][k]
                        
                        #已确定部分完成聚合
                        this_last_ct.append(temp_this_ct / len(last_part_lt[j]))
                        this_last_ct_count.append(len(last_part_lt[j]))
            
            
                center_topic = np.zeros((cluster_number, len(PHAI[0])))
                each_cluster_number = np.zeros(cluster_number, int)
            
                print "empty_number" , empty_count
                
                remain_ct_num = empty_count
                
                if remain_ct_num == 0 and len(new_part_lt) == 0:
                    #直接将上一片的主题作为本片的主题
                    #每片求均值
                    for j in range(len(this_last_ct)):
                        center_topic[j] = this_last_ct[j]
                        each_cluster_number[j] = this_last_ct_count[j]
                    
                    tci = 1.0
                #此种情况一般不会发生，若发生，表明s_lamdba设置过小
                elif remain_ct_num > 0 and len(new_part_lt) == 0:
                    print 'Exception21'
                    #直接将上一片的主题作为本片的主题
                    #每片求均值
                    for j in range(len(this_last_ct)):
                        center_topic[j] = this_last_ct[j]
                        each_cluster_number[j] = this_last_ct_count[j]
                
                    center_topic = center_topic[0 : len(this_last_ct), :]
                    each_cluster_number = each_cluster_number[0 : len(this_last_ct)]
                    cluster_tag = cluster_tag[0 : len(this_last_ct)]
                    tci = 0.0
                elif remain_ct_num == 0 and len(new_part_lt) > 0:    
                #替换1个中心主题
                
                    new_part_ct = np.zeros((1, len(PHAI[0])))
                    for j in range(len(new_part_lt)):
                        new_part_ct += new_part_lt[j]
                
                    new_part_ct = new_part_ct / len(new_part_lt)
                
                    min_index = np.argmin(this_last_ct_count)
                
                    #找出被删去的主题与哪一个最为相近，合并之
                    merge_si = np.zeros(len(this_last_ct), float)
                    for j in range(len(this_last_ct)):
                        if j == min_index:
                            merge_si[j] = -1
                        else:
                            merge_si[j] = 1.0 / (SKLD(this_last_ct[min_index], this_last_ct[j]) + 1.0)
                
                    merge_des = np.argmax(merge_si)
                
                    this_last_ct[min_index] = new_part_ct
                    this_last_ct_count[min_index] = len(new_part_lt)
                    
                    ###???
                    #this_last_ct[merge_des] = (this_last_ct[merge_des] + this_last_ct[min_index]) / 2.0
                    #聚类元素个数相加
                    this_last_ct_count[merge_des] = this_last_ct_count[merge_des] + this_last_ct_count[min_index]
                
                    for j in range(len(this_last_ct)):
                        center_topic[j] = this_last_ct[j]
                        each_cluster_number[j] = this_last_ct_count[j]
                
                    for j in range(len(cluster_tag)):
                        #-1变为min_index
                        #min_index变为merge_des
                        if cluster_tag[j] == -1:
                            cluster_tag[j] = min_index
                        elif cluster_tag[j] == min_index:
                            cluster_tag[j] = merge_des
                    
                    tci = np.true_divide((cluster_number - 1), cluster_number)
                    
                else:
                    #更新前面部分
                    for j in range(len(this_last_ct)):
                        center_topic[j] = this_last_ct[j]
                        each_cluster_number[j] = this_last_ct_count[j]
                
                    #新增1个主题
                    if remain_ct_num == 1:
                        new_part_ct = np.zeros((1, len(PHAI[0])))
                        for j in range(len(new_part_lt)):
                            new_part_ct += new_part_lt[j]
                
                        new_part_ct = new_part_ct / len(new_part_lt)

                        center_topic[-1] = new_part_ct
                        each_cluster_number[-1] = len(new_part_lt)
                        for j in range(len(cluster_tag)):
                            if cluster_tag[j] == -1:
                                cluster_tag[j] = cluster_number - 1
                
                    #这里可能会有异常                
                    #elif len(new_part_lt) == 1:
                    
                    #新增若干个主题   
                    else:
                        #谱聚类
                        #print new_part_lt
                        sp_label = spectral_cluster(new_part_lt, remain_ct_num)
                        new_part_ct = np.zeros((remain_ct_num, len(PHAI[0])))
                        new_part_ct_number = np.zeros(remain_ct_num, int)
                        for j in range(len(sp_label)):
                            new_part_ct[sp_label[j]] += new_part_lt[j]
                            new_part_ct_number[sp_label[j]] += 1
                    
                        for j in range(remain_ct_num):
                            new_part_ct[j] = new_part_ct[j] / new_part_ct_number[j]
                            center_topic[len(this_last_ct) + j] = new_part_ct[j]
                            each_cluster_number[len(this_last_ct) + j] = new_part_ct_number[j]
                    
                        new_count = 0
                        for j in range(len(cluster_tag)):
                            if cluster_tag[j] == -1:
                                cluster_tag[j] = cluster_number - remain_ct_num + sp_label[new_count]
                                new_count += 1
                    tci = np.true_divide((cluster_number - remain_ct_num), cluster_number)
            
            ##########    
            else:
                #全选
                
                print 'case22'
                init_central_topic[0 : len(last_central_topic[0]), :] = last_central_topic[0]
                
                # 合并向量空间
                merge_init_ct, merge_this_lt, new_word_list = merge_space(last_word_list[0], this_word_list, init_central_topic, PHAI)
            
            
                #计算当前潜在主题与前一片的中心主题之间的相似度
                lt_ct_similarity = np.zeros((this_lt_number, len(init_central_topic)));
                for j in range(this_lt_number):
                    for k in range(len(init_central_topic)):
                        lt_ct_similarity[j, k] = 1.0 / (SKLD(merge_this_lt[j], merge_init_ct[k]) + 1.0)
                
                #print lt_ct_similarity
                cluster_tag = []
                new_part_lt = []  #原空间(500维)下的本数据片的新出现的潜在主题
                last_part_lt = []  #原空间下的与上一数据片中的中心主题比较相似的潜在主题,二维
                
                for j in range(len(init_central_topic)):
                    last_part_lt.append([])
                
                for j in range(this_lt_number):
                    if np.max(lt_ct_similarity[j]) < s_lambda:
                        #新的潜在主题
                        new_part_lt.append(PHAI[j])
                        cluster_tag.append(-1)  #新类编号
                    else:
                        #是旧的主题的延续
                        max_index = np.argmax(lt_ct_similarity[j])
                        last_part_lt[max_index].append(PHAI[j])
                        cluster_tag.append(max_index)
                          
                empty_count = 0
                this_last_ct = []
                this_last_ct_count = []
                
                for j in range(cluster_number):
                
                    if len(last_part_lt[j]) == 0:
                        empty_count += 1
                    else:
                        temp_this_ct = np.zeros(len(PHAI[0]))
                        
                        for k in range(len(last_part_lt[j])):
                            temp_this_ct += last_part_lt[j][k]
                        
                        #已确定部分完成聚合
                        this_last_ct.append(temp_this_ct / len(last_part_lt[j]))
                        this_last_ct_count.append(len(last_part_lt[j]))
            
            
                center_topic = np.zeros((cluster_number, len(PHAI[0])))
                each_cluster_number = np.zeros(cluster_number, int)
            
                print "empty_number" , empty_count
                
                remain_ct_num = cluster_number - (len(init_central_topic) - empty_count)
                
                if remain_ct_num == 0 and len(new_part_lt) == 0:
                    #此种情况不会发生
                    #直接将上一片的主题作为本片的主题
                    #每片求均值
                    for j in range(len(this_last_ct)):
                        center_topic[j] = this_last_ct[j]
                        each_cluster_number[j] = this_last_ct_count[j]
                    
                    tci = 1.0
                #此种情况一般不会发生，若发生，表明s_lamdba设置过小
                elif remain_ct_num > 0 and len(new_part_lt) == 0:
                    #直接将上一片的主题作为本片的主题
                    #每片求均值
                    
                    print 'Exception22'
                    for j in range(len(this_last_ct)):
                        center_topic[j] = this_last_ct[j]
                        each_cluster_number[j] = this_last_ct_count[j]
                
                    center_topic = center_topic[0 : len(this_last_ct), :]
                    each_cluster_number = each_cluster_number[0 : len(this_last_ct)]
                    cluster_tag = cluster_tag[0 : len(this_last_ct)]
                    tci = 0.0
                elif remain_ct_num == 0 and len(new_part_lt) > 0:
                #此种情况不会发生    
                #替换1个中心主题
                    print 'Exception23'
                    new_part_ct = np.zeros((1, len(PHAI[0])))
                    for j in range(len(new_part_lt)):
                        new_part_ct += new_part_lt[j]
                
                    new_part_ct = new_part_ct / len(new_part_lt)
                
                    min_index = np.argmin(this_last_ct_count)
                
                    #找出被删去的主题与哪一个最为相近，合并之
                    merge_si = np.zeros(len(this_last_ct), float)
                    for j in range(len(this_last_ct)):
                        if j == min_index:
                            merge_si[j] = -1
                        else:
                            merge_si[j] = 1.0 / (SKLD(this_last_ct[min_index], this_last_ct[j]) + 1.0)
                
                    merge_des = np.argmax(merge_si)
                
                    this_last_ct[min_index] = new_part_ct
                    this_last_ct_count[min_index] = len(new_part_lt)
                    
                    ###???
                    #this_last_ct[merge_des] = (this_last_ct[merge_des] + this_last_ct[min_index]) / 2.0
                    #聚类元素个数相加
                    this_last_ct_count[merge_des] = this_last_ct_count[merge_des] + this_last_ct_count[min_index]
                
                    for j in range(len(this_last_ct)):
                        center_topic[j] = this_last_ct[j]
                        each_cluster_number[j] = this_last_ct_count[j]
                
                    for j in range(len(cluster_tag)):
                        #-1变为min_index
                        #min_index变为merge_des
                        if cluster_tag[j] == -1:
                            cluster_tag[j] = min_index
                        elif cluster_tag[j] == min_index:
                            cluster_tag[j] = merge_des
                    
                    tci = np.true_divide((cluster_number - 1), cluster_number)
                else:
                    #更新前面部分
                    for j in range(len(this_last_ct)):
                        center_topic[j] = this_last_ct[j]
                        each_cluster_number[j] = this_last_ct_count[j]
                
                    #新增1个主题
                    if remain_ct_num == 1:
                        new_part_ct = np.zeros((1, len(PHAI[0])))
                        for j in range(len(new_part_lt)):
                            new_part_ct += new_part_lt[j]
                
                        new_part_ct = new_part_ct / len(new_part_lt)

                        center_topic[-1] = new_part_ct
                        each_cluster_number[-1] = len(new_part_lt)
                        for j in range(len(cluster_tag)):
                            if cluster_tag[j] == -1:
                                cluster_tag[j] = cluster_number - 1
                
                    #这里可能会有异常                
                    #elif len(new_part_lt) == 1:
                    
                    #新增若干个主题   
                    else:
                        #谱聚类
                        #print new_part_lt
                        sp_label = spectral_cluster(new_part_lt, remain_ct_num)
                        new_part_ct = np.zeros((remain_ct_num, len(PHAI[0])))
                        new_part_ct_number = np.zeros(remain_ct_num, int)
                        for j in range(len(sp_label)):
                            new_part_ct[sp_label[j]] += new_part_lt[j]
                            new_part_ct_number[sp_label[j]] += 1
                    
                        for j in range(remain_ct_num):
                            new_part_ct[j] = new_part_ct[j] / new_part_ct_number[j]
                            center_topic[len(this_last_ct) + j] = new_part_ct[j]
                            each_cluster_number[len(this_last_ct) + j] = new_part_ct_number[j]
                    
                        new_count = 0
                        for j in range(len(cluster_tag)):
                            if cluster_tag[j] == -1:
                                cluster_tag[j] = cluster_number - remain_ct_num + sp_label[new_count]
                                new_count += 1
                     
                    tci = np.true_divide((cluster_number - remain_ct_num), cluster_number)
            #计算文档-主题相似度
            weibo_topic_similarity = np.zeros((cluster_number, len(THETA)))
            THETA = THETA.transpose()
            
            for j in range(len(cluster_tag)):
                weibo_topic_similarity[cluster_tag[j]] += THETA[j]
        
            weibo_topic_similarity = weibo_topic_similarity.transpose()


        '''
        公共部分
        '''
        
        run_time.append(str(time.clock() - start))
        print "This time:", str(time.clock() - start)
        
        # 本片数据作为缓冲区
        ecn_to_string = [str(x) for x in each_cluster_number]
        last_central_topic = []
        last_each_cluster_number = []
        last_word_list = []
        
        last_central_topic.append(center_topic)
        last_each_cluster_number.append(each_cluster_number)
        last_word_list.append(this_word_list)

        write_matrix_to_text(weibo_topic_similarity, write_directory1 + '/' + str(i + 1) + '.txt')
        write_matrix_to_text(center_topic, write_directory2 + '/' + str(i + 1) + '.txt')
        quick_write_list_to_text(ecn_to_string, write_directory3 + '/' + str(i + 1) + '.txt')
        
        print "Segment %d Completed." % (i + 1)
    
    quick_write_list_to_text(run_time, write_filename)


if __name__ == '__main__':
    #start = time.clock()
    now_directory = os.getcwd()
    root_directory = os.path.dirname(now_directory) + '/'
    
    read_directory1 = root_directory + u'dataset2/DCTM/doc_topic60'
    read_directory2 = root_directory + u'dataset2/DCTM/topic_word60'
    read_directory3 = root_directory + u'dataset2/text_model/select_words'
    write_directory1 = root_directory + u'dataset2/DCTM/mctrwdctm_doc_ct60'
    write_directory2 = root_directory + u'dataset2/DCTM/mctrwdctm_ct_word60'
    write_directory3 = root_directory + u'dataset2/DCTM/mctrwdctm_ct_number60'
    write_filename = root_directory + u'dataset2/DCTM/mctrwdctm_time60.txt'


    if (not(os.path.exists(write_directory1))):
        os.mkdir(write_directory1)
    if (not(os.path.exists(write_directory2))):
        os.mkdir(write_directory2)
    if (not(os.path.exists(write_directory3))):
        os.mkdir(write_directory3)

    
    MCTRW_DCTM(read_directory1, read_directory2, read_directory3, write_directory1, write_directory2, write_directory3, write_filename)
    
    #print 'Total time %f seconds' % (time.clock() - start)
    print 'Complete !!!'
