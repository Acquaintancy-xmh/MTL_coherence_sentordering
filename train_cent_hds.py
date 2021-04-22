import torch
from torch import nn
from tqdm import tqdm, trange
from random import shuffle

import os
# torch.cuda.set_device(3)
device_ids=[0]

print(torch.cuda.device_count())

"""
A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.
This implementation uses the nn package from PyTorch to build the network.
Rather than manually updating the weights of the model as we have been doing,
we use the optim package to define an Optimizer that will update the weights
for us. The optim package defines many optimization algorithms that are commonly
used for deep learning, including SGD+momentum, RMSProp, Adam, etc.
"""



wordvec_num, wordvec_dim, wordvec_hidden_size, speech_num, speech_dim, speech_hidden_size,  feature_num, feature_dim, feature_hidden_size =\
30, 300, 30, 30, 300, 30, 30 ,300, 30


# Create random Tensors to hold inputs and outputs.
sample_num = 37000
wordvec = torch.randn(sample_num ,wordvec_num, wordvec_dim).cuda()
speech = torch.randn(sample_num,speech_num, speech_dim).cuda()
feature = torch.randn(sample_num,feature_num, feature_dim).cuda()
y= torch.randint(0, 2, (sample_num,)).cuda()
# 占位变量
# var= torch.randn(sample_num, wordvec_num*4, wordvec_dim).cuda()

num_classes = 140
num_epochs = 1000000
batch_size =512
learning_rate = 0.0001
# num_layers = 10000000


# Use the nn package to define our model and loss function.
class MultiInputClassifierNet(nn.Module):
    def __init__(self,num_classes,
                 wordvec_num, wordvec_dim, wordvec_hidden_size,
                 speech_num, speech_dim, speech_hidden_size,
                 feature_num, feature_dim, feature_hidden_size):
        super(MultiInputClassifierNet, self).__init__()

        self.wordvec_num, self.speech_num ,self.feature_num = wordvec_num, speech_num, feature_num

        self.fc1 = nn.Linear(wordvec_dim, wordvec_hidden_size)
        self.fc2 = nn.Linear(speech_dim, speech_hidden_size)
        self.fc3 = nn.Linear(feature_dim, feature_hidden_size)
        self.fc4 = nn.Linear(wordvec_num * wordvec_hidden_size + speech_num* speech_hidden_size + \
                             feature_num * feature_hidden_size , num_classes)
        self.relu = nn.ReLU()

        # for i in range(num_layers):
        #     eval("fc_{}".format(i)) = nn.Linear(wordvec_dim, wordvec_hidden_size)

    def forward(self, wordvec, speech, feature):
        out1_list = []
        for i in range(self.wordvec_num):
            in1 = torch.squeeze(torch.index_select(wordvec, 1, torch.tensor(i).cuda()))
            out1 = self.relu(self.fc1(in1))
            in2 = torch.squeeze(torch.index_select(speech, 1, torch.tensor(i).cuda()))
            out2 = self.relu(self.fc2(in2))
            out = torch.cat((out1,out2), -1)
            out1_list.append(out)
        out1_list= torch.cat(tuple(out1_list), 1)

        out3_list = []
        for i in range(self.feature_num):
            in_ = torch.squeeze(torch.index_select(feature, 1, torch.tensor(i).cuda()))
            out = self.relu(self.fc3(in_))
            out3_list.append(out)
        out3_list = torch.cat(tuple(out3_list), 1)

        out = torch.cat((out1_list, out3_list), -1)
        out = self.fc4(out)
        x = input()
        return out

model = MultiInputClassifierNet(num_classes,
                 wordvec_num, wordvec_dim, wordvec_hidden_size,
                 speech_num, speech_dim, speech_hidden_size,
                 feature_num, feature_dim, feature_hidden_size)
nn.DataParallel(model,device_ids=device_ids)

# Loss
criterion = nn.CrossEntropyLoss()


# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in trange(num_epochs):
    # print("epoch {}".format(epoch))
    for i in range(0, sample_num, batch_size ):
        # Forward pass: compute predicted y by passing x to the model.

        batch_wordvec = wordvec[i:i+batch_size]
        batch_speech = speech[i:i+batch_size]
        batch_feature = feature[i:i+batch_size]
        batch_y = y[i:i+batch_size]

        batch_y_pred = model(batch_wordvec, batch_speech, batch_feature)

        # Compute and print loss.
        loss = criterion(batch_y_pred, batch_y)
        # print("epoch {} step {} loss:{}".format(epoch,i,loss))

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the Tensors it will update (which are the learnable weights
        # of the model)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        # loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        # optimizer.step()
