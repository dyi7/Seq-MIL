# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:10:29 2020

@author: DELL
"""
import pickle as pkl
import gzip
import numpy as np
import random
import math
import pandas as pd
from datetime import datetime
from datetime import timedelta
from scipy import stats

def delay(j, day):
    return (datetime.strptime(j, '%Y-%m-%d') - timedelta(days=day)).strftime('%Y-%m-%d')


class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, instance, group_score,#min_instances=10,
                 dict,types='title', #加emb
                 batch_size=32,
                 n_words=-1,
                 max_size = 200,
                 cut_word=False, cut_news=False,
                 shuffle=True, shuffle_sentence=False,  quiet=False):  # delay means how many days over the past

        self.instance = pd.read_csv(instance).set_index('date')
        self.instance = self.instance[types].groupby(self.instance.index).apply(list).apply(pd.Series).fillna(
            '')  # group together
        self.group_score = pd.read_csv(group_score).set_index('Date')
        #self.min_instances = min_instances
        self.max_size = max_size  # max number of groups
        self.current_group_index = 0
        self.current_instance_index = 0
        
        with open(dict, 'rb') as f:
            self.dict = pkl.load(f)
        self.down = 0
        self.up = 0
        self.data_dictionary = {}
        #self.instance, self.group_labels, self.group_lengths = instances, group_labels, group_lengths
        
        self.batch_size = batch_size
        self.n_words = n_words
        self.shuffle = shuffle
        self.shuffle_sentence = shuffle_sentence
        self.types = types
        self.end_of_data = False
        self.cut_word = cut_word if cut_word else float('inf')  # cut the word
        self.cut_news = cut_news if cut_news else None  # cut the sentence
        self.instance_buffer = []
        self.group_score_buffer = []
        self.ori_instance_buffer = []
        
        self.k = batch_size * 20
        self.index = 0
        
        """if not quiet:
            print('Total instances: ', len(self.instance), ' in ', len(self.group_score), ' groups')
            print('Up instances: ', self.up, ' Down instances: ', self.down)"""
        

    def __iter__(self):
        return self

    def reset(self):
        self.index = 0

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        
        temp = []
        original_temp = []
        original=[]
        original_gath= []
        instances = []
        instan_gath = []
        group = []
        group_scores = []
        group_lengths = []  # needed to find limits of groups
        al = []
        d = 1
        assert len(self.instance_buffer) == len(self.group_score_buffer), 'Buffer size mismatch!'

        if len(self.instance_buffer) == 0:
            for j, i in enumerate(self.group_score.index.values[self.index:self.index + self.k]):  # j for count i for value
                try:
                    ss = list(filter(lambda x: self.cut_word > len(x.split()) > 0,
                                     self.instance.loc[delay(i, 1)].values[:self.cut_news]))
                    ll = self.group_score.loc[i].values
                except KeyError as e:  # out of length
                    print(i + ' ' + str(e))
                    continue

                self.instance_buffer.append(ss)
                self.group_score_buffer.append(int(ll))
            if 'j' in locals():
                self.index += j + 1
            ##TODO delete useless

            if self.shuffle:
                # sort by target buffer
                tlen = np.array([len(t) for t in self.instance_buffer])
                tidx = tlen.argsort()
                # argsort the index from low to high
                # shuffle mini-batch
                tindex = []
                ##Todo shuffle
                small_index = list(range(int(math.ceil(len(tidx) * 1. / self.batch_size))))
                random.shuffle(small_index)
                for i in small_index:
                    if (i + 1) * self.batch_size > len(tidx):
                        tindex.extend(tidx[i * self.batch_size:])
                    else:
                        tindex.extend(tidx[i * self.batch_size:(i + 1) * self.batch_size])
                tidx = tindex

                _sbuf = [self.instance_buffer[i] for i in tidx]
                _lbuf = [self.group_score_buffer[i] for i in tidx]

                self.instance_buffer = _sbuf
                self.group_score_buffer = _lbuf
                ##TODO delete useless
                del _sbuf, _lbuf
            for i in self.instance_buffer:
                original_temp.append([[j] for j in i])
                temp.append([j.strip().split() for j in i])  # split words and save to array
                
            self.instance_buffer = temp
            self.ori_instance_buffer = original_temp
            ##TODO delete useless
            del temp,original_temp
            
        if len(self.instance_buffer) == 0 or len(self.group_score_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:
            while True:
                # read from source file and map to word index
                instance_temp = []
                k = []
                try:
                    j = self.instance_buffer.pop(0)  # 1 day before N个instance, #每个“[[..]]”, 【[]],[],[]】
                    h = self.ori_instance_buffer.pop(0) #【['abbott defends parental... .'],['goldman sachs hires ... .']...['...']】
                except IndexError:
                    break
                ##TODO do shuffle 
                if self.shuffle_sentence:
                    np.random.shuffle(j)
                    np.random.shuffle(h)
                for i in j:  # deal with 1 day before, each instance,一个[]
                    ss = [self.dict[w] if w in self.dict else 1 for w in i]  # 1 means _UNK_  i:each instance,w:each word
                    if self.n_words > 0:
                        ss = [w if w < self.n_words else 1 for w in ss]  # 1 means _UNK_
                    instance_temp.append(ss)  #ss(dict的数字！) = features(已*300) = one word, instance_temp = emb = #【[],[],[]..[]】
                    #print(instance_temp) 
                    #instance_temp1 = np.pad(instance_temp,((self.max_size-len(instance_temp),0),(0,0)),'constant',constant_values = (0,0))
                    #instances.append(instance_temp)  #instance_temp is one sentence  instances.append(instance_temp)
                
                for m in h:
                    ss_o = [i for i in m]
                    k += ss_o
                
                # read label
                instances.append(instance_temp) #instance_temp[[]],32个[[]]
                instan_gath += instance_temp
                original.append(k) #32，同instances【[]】
                original_gath+=(k) #同instan_gath
                al.append([original_gath])
                group.append(instance_temp)
                # XXXX instances += instance_temp
                # XXXX group += instance_temp
                #print('*'*80)  #32个循环
                #print(np.shape(instan_gath))
                
                
                score = self.group_score_buffer.pop(0)
                group_scores.append(score)   #label
                
                self.data_dictionary[d] = (group, score)
                d += 1
                ##TODO delete useless
                ##del instance_temp, source_d1_temp, source_d2_temp

                if d >= self.max_size or len(group_scores) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True
        ####################################End of loop
        
        ################################## Can only be placed here!!!   
        group_lengths = len(original_gath)
        """for i in range(len(instances)):
            group_lengths.append(len(instances[i]))  #32-dim
        #maxlen__x = max(group_lengths)"""

        if score == 0:  # for stats
            self.down += 1
        else:
            self.up += 1

        if len(group_scores) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        
        
        #print(instan_gath)
        # print(np.shape(instan_gath))  #= np.sum(group_lengths) (365,)
        #X = np.array(instances, dtype='float16') 
        #X = np.array(original)  #仍然不可以
        X = np.array(original_gath)
        group_scores = np.array(group_scores, dtype='uint8')
        group_lengths = np.array(group_lengths, dtype='uint16')
        #del instances  # memory save
        return X, instan_gath, group_scores, group_lengths #X list,list, array,array
        #return instances, instan_gath, group_scores, group_lengths
        #return source, label
        
        #返回3样

        
def main():
    train = TextIterator('news_set/train.csv',
                         'price_set/train_label.csv',
                         max_size = 200, # max number of groups
                         dict='news_set/vocab_cased_title.pickle',
                         types='title',
                         n_words=43920,
                         batch_size=32, cut_word=False, cut_news=False,
                         shuffle=True,  quiet=False) 
    validate = TextIterator('news_set/validate.csv',
                            'price_set/validate_label.csv',
                            max_size = 200,# max number of groups
                            dict='news_set/vocab_cased_title.pickle',
                            types='title',
                            n_words=43920,
                            batch_size=32, cut_word=False, cut_news=False,
                            shuffle=True,  quiet=False)
    test = TextIterator('news_set/test.csv',
                        'price_set/test_label.csv',
                        max_size = 200, # max number of groups
                        dict='news_set/vocab_cased_title.pickle',
                        types='title',
                        n_words=43920,
                        batch_size=32, cut_word=False, cut_news=False,
                        shuffle=True,  quiet=False)
    # cut news: max news number per day
    for i, (x, instance_gath, group_scores, group_lengths) in enumerate(train): #(array,list,array,array)原本x是instances,len(x)=32
        #print(x)
        #print(len(x)) #len(original)=32?个【[],[],】,len(originak_gath)=(320,
        #print(instance_gath)
        #print(len(x),len(x[0]),len(x[0][0]))
        #print('*'* 80)
        print("Train", i,'Total instances: ', np.shape(instance_gath), ' of Total group length: ', group_lengths, ' in ', len(group_scores), ' groups')
        #print('Positives: ', self.positives, ' Negatives: ', self.negatives)
    for i, (x, instance_gath,group_scores, group_lengths) in enumerate(validate):
        print("validate", i,'Total instances: ', np.shape(instance_gath), ' of Total group length: ', group_lengths, ' in ', len(group_scores), ' groups')
        #print("validate", i, 'Total instances: ', np.sum(group_lengths), ' in ', len(group_scores), ' groups')
    for i, (x, instance_gath,group_scores, group_lengths) in enumerate(test):
        print("Test", i,'Total instances: ', np.shape(instance_gath), ' of Total group length: ', group_lengths, ' in ', len(group_scores), ' groups')
        #print("test", i, 'Total instances: ', np.sum(group_lengths), ' in ', len(group_scores), ' groups')
    
if __name__ == '__main__':
    main()
