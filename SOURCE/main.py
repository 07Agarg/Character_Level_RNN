# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:52:40 2019

@author: ashima.garg
"""
#Classifying Names with a Character Level RNN

import data
import model
import config

if __name__ == "__main__":
    data_obj = data.DATA()
    data_obj.read()
    print("Train Data Loaded")
    # BUILD MODEL
    rnn = model.RNN(data_obj.n_letters, config.N_HIDDEN, data_obj.n_categories)
    modeloperator = model.Operator(rnn, config.LEARNING_RATE)
    # TRAIN MODEL
    modeloperator.train(data_obj)
    print("Model Trained")    
    # TEST MODEL
    print("all categories: ", len(data_obj.categories))
    modeloperator.predict(data_obj, 'Dovesky')
    modeloperator.predict(data_obj, 'Satoshi')
    modeloperator.predict(data_obj, 'Jackson')