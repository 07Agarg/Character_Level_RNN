# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:55:34 2019

@author: ashima.garg
"""

import random

def categoryFromOutput(data, output):
    max_val, max_idx = output.topk(1)               # to get the hightest value
    #print("max_val, max_idx, ", max_val, max_idx)
    category_i = max_idx.item()  
    return category_i
    
def randomChoice(l):
        return l[random.randint(0, len(l) - 1)]
    