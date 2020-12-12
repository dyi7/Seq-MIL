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
    def __init__(self, instance, group_score,technical,
                 dict,delay_tech=5,types='title', #加emb
                 batch_size=32,
                 n_words=-1,
                 #max_size = 200,
                 cut_word=False, cut_news=False,
                 shuffle=True, shuffle_sentence=False,  quiet=False):  # delay means how many days over the past

        self.instance = pd.read_csv(instance).set_index('date')
        self.instance = self.instance[types].groupby(self.instance.index).apply(list).apply(pd.Series).fillna(
            '')  # group together
        self.group_score = pd.read_csv(group_score).set_index('Date')
        self.technical = pd.read_csv(technical)
        #self.min_instances = min_instances
        #self.max_size = max_size  # max number of groups

        
        with open(dict, 'rb') as f:
            self.dict = pkl.load(f)
        self.down = 0
        self.up = 0
        
        self.batch_size = batch_size
        self.n_words = n_words
        self.shuffle = shuffle
        self.shuffle_sentence = shuffle_sentence
        self.delay_tec = delay_tech  # delay_tec = 1 means one day ago
        self.types = types
        self.end_of_data = False
        self.cut_word = cut_word if cut_word else float('inf')  # cut the word
        self.cut_news = cut_news if cut_news else None  # cut the sentence
        self.instance_buffer = []
        self.group_score_buffer = []
        self.technical_buffer = []
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
        instances = []
        instan_gath = []
        group = []
        group_scores = []
        group_lengths = []  # needed to find limits of groups
        tech_final = []

        assert len(self.instance_buffer) == len(self.group_score_buffer), 'Buffer size mismatch!'

        if len(self.instance_buffer) == 0:
            for j, i in enumerate(self.group_score.index.values[self.index:self.index + self.k]):  # j for count i for value
                try:
                    ss = list(filter(lambda x: self.cut_word > len(x.split()) > 0,
                                     self.instance.loc[delay(i, 1)].values[:self.cut_news]))
                    ll = self.group_score.loc[i].values
                    idx = self.technical.index[self.technical['Date'] == i][0]
                    ## 8 means the index of column, T is transpose
                    tec = self.technical.iloc[idx - self.delay_tec:idx, 8:].values
                except KeyError as e:  # out of length
                    print(i + ' ' + str(e))
                    continue

                self.instance_buffer.append(ss)
                self.group_score_buffer.append(int(ll))
                self.technical_buffer.append(tec)
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
                _tech = [self.technical_buffer[i] for i in tidx]

                self.instance_buffer = _sbuf
                self.group_score_buffer = _lbuf
                self.technical_buffer = _tech
                ##TODO delete useless
                del _sbuf, _lbuf
            for i in self.instance_buffer:
                temp.append([j.strip().split() for j in i])  # split words and save to array
                
            self.instance_buffer = temp
            ##TODO delete useless
            del temp#,original_temp
            
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
                except IndexError:
                    break
                ##TODO do shuffle 
                if self.shuffle_sentence:
                    np.random.shuffle(j)
                for i in j:  # deal with 1 day before, each instance,一个[]
                    ss = [self.dict[w] if w in self.dict else 1 for w in i]  # 1 means _UNK_  i:each instance,w:each word
                    if self.n_words > 0:
                        ss = [w if w < self.n_words else 1 for w in ss]  # 1 means _UNK_
                    instance_temp.append(ss)  #ss(dict的数字！) = features(已*300) = one word, instance_temp = emb = #【[],[],[]..[]】
                    #print(instance_temp) 
                    #instance_temp1 = np.pad(instance_temp,((self.max_size-len(instance_temp),0),(0,0)),'constant',constant_values = (0,0))
                    #instances.append(instance_temp)  #instance_temp is one sentence  instances.append(instance_temp)

                
                # read label
                instances.append(instance_temp) #instance_temp[[]],32个[[]]
                instan_gath += instance_temp
                # (XXXX) instances += instance_temp
                #print('*'*80)  #32个循环
                #print(np.shape(instan_gath))
                
                
                score = self.group_score_buffer.pop(0)
                group_scores.append(score)   #label
                tech_tech = self.technical_buffer.pop(0)
                tech_final.append(tech_tech)
                
                ##TODO delete useless
                del instance_temp

                if len(group_scores) >= self.batch_size: #or d >= self.max_size:
                    break
        except IOError:
            self.end_of_data = True
        ####################################End of loop
        
        ################################## Can only be placed here!!!   
        for i in range(len(instances)):
            group_lengths.append(len(instances[i]))  #32-dim
        maxlen__x = max(group_lengths)

        if score == 0:  # for stats
            self.down += 1
        else:
            self.up += 1

        if len(group_scores) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        
    
        group_scores = np.array(group_scores, dtype='uint8')
        return instances,group_scores,np.array(tech_final)


        
def main():
    train = TextIterator('news_set/train.csv',
                         'price_set/train_label.csv',
                         'technical.csv',
                         #max_size = 200, # max number of groups
                         dict='news_set/vocab_cased_title.pickle',
                         types='title',
                         delay_tech=1,
                         n_words=43920,
                         batch_size=32, cut_word=False, cut_news=False,
                         shuffle=True,  quiet=False) 
    validate = TextIterator('news_set/validate.csv',
                            'price_set/validate_label.csv',
                            'technical.csv',
                            #max_size = 200,# max number of groups
                            dict='news_set/vocab_cased_title.pickle',
                            types='title',
                            delay_tech=1,
                            n_words=43920,
                            batch_size=32, cut_word=False, cut_news=False,
                            shuffle=True,  quiet=False)
    test = TextIterator('news_set/test.csv',
                        'price_set/test_label.csv',
                        'technical.csv',
                        #max_size = 200, # max number of groups
                        dict='news_set/vocab_cased_title.pickle',
                        types='title',
                        delay_tech=1,
                        n_words=43920,
                        batch_size=32, cut_word=False, cut_news=False,
                        shuffle=True,  quiet=False)
    # cut news: max news number per day
    for i, (x, group_scores,tech) in enumerate(train): #(array,list,array,array)原本x是instances,len(x)=32
        print("train", i, 'length', len(x), group_scores.shape, tech.shape)
    for i, (x, group_scores,tech) in enumerate(validate):
        print("Validate", i,'length', len(x), group_scores.shape, tech.shape)
    for i, (x, group_scores,tech) in enumerate(test):
        print("Test", i,'length', len(x), group_scores.shape, tech.shape)
    
if __name__ == '__main__':
    main()
