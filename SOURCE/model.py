# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:53:51 2019

@author: ashima.garg
"""
import os
import config
import torch
import torch.nn as nn
import utils

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    
class Operator:
    
    def __init__(self, net, learning_rate):
        self.net = net
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr = learning_rate)
        self.criterion = nn.NLLLoss()
        
    def train(self, data_obj):
        total_loss = 0
        for epoch in range(config.NUM_EPOCHS):
            category, category_tensor, line, line_tensor = data_obj.randomTrainingExample()
            hidden = self.net.initHidden()
            self.optimizer.zero_grad()
            for i in range(len(line)):
                output, hidden = self.net(line_tensor[i], hidden)
            loss_val = self.criterion(output, category_tensor)
            loss_val.backward()
            #print("loss_val data: ", loss_val.data[0])
            total_loss += loss_val
            self.optimizer.step()
            if epoch % 1:
                print('Epoch: %d Loss: %.3f' % (epoch, loss_val))
        torch.save(self.net, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".pt"))
        
    def predict(self, data_obj, line):
        self.net = torch.load(os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".pt"))
        print("Loaded model: ", os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".pt"))
        line_tensor = data_obj.lineTotensor(data_obj.unicodeToAscii(line))
        hidden = self.net.initHidden()
        for i in range(len(line)):
            output, hidden = self.net(line_tensor[i], hidden)
        #print("Predicted_Output: ", output)
        category_i = utils.categoryFromOutput(data_obj, output)
        #print("category_i: ", category_i)
        print("Predicted Category for {} is {}: ".format(line, data_obj.categories[category_i]))