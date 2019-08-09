# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:53:41 2019

@author: ashima.garg
"""

import os
import config
import unicodedata
import torch
import string
import utils

class DATA():
    
    def __init__(self):
        self.dir_path = config.NAME_FILE
        self.filelist = os.listdir(self.dir_path)
        self.batch_size = config.BATCH_SIZE
        self.size = len(self.filelist)
        self.data_index = 0
        self.categories = []
        self.category_lines = {}
        self.all_letters = string.ascii_letters + ".,;'-"
        self.n_letters = len(self.all_letters)
        self.n_categories = None
        
    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    #Turn a line into <line_length, 1, n_letters>
    def lineTotensor(self, s):
        line_t = torch.zeros(len(s), 1, self.n_letters)
        for i in range(len(s)):
            line_t[i][0][self.all_letters.find(s[i])] = 1
        return line_t
        
    # Read a file and split into lines
    def readLines(self, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [self.unicodeToAscii(line) for line in lines]
        #return [self.lineTotensor(self.unicodeToAscii(line)) for line in lines]

    def read(self):
        for i in range(self.size):
            filename = os.path.join(self.dir_path, self.filelist[i])
            category = os.path.splitext(os.path.basename(filename))[0]
            self.categories.append(category)
            lines = self.readLines(filename)
            self.category_lines[category] = lines
            self.n_categories = len(self.categories)
            
    def randomTrainingExample(self):
        category = utils.randomChoice(self.categories)
        #print("category ", category)
        line = utils.randomChoice(self.category_lines[category])
        #print(line)
        category_tensor = torch.tensor([self.categories.index(category)], dtype = torch.long)
        #print(category_tensor)
        line_tensor = self.lineTotensor(line)
        return category, category_tensor, line, line_tensor