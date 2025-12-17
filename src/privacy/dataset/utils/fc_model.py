import os
import torch
from torch.utils import data
from sklearn import metrics
import math
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
from torch.utils import data
from pytorch_metric_learning.losses import NTXentLoss
import torch
import pandas as pd
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from dataset.SiameseFPIDataset import SiameseNCEDataset
"""
This file provides the most important functions that are used in our experiments. These functions are called in 
AgentSiameseNetwork.py which provides the actual training/validation loop and the code for evaluation.
"""


def train(net, training_loader, n_samples, batch_size, criterion, optimizer, epoch, n_epochs):
    net.train()
    running_loss = 0.0

    # Handle DistributedDataParallel wrapper
    model = net.module if hasattr(net, 'module') else net
        
    print('Training----->')
    for i, batch in enumerate(training_loader):
        inputs1, inputs2, labels = batch
        inputs1, inputs2, labels = inputs1.cuda(), inputs2.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

       
        # Compute the absolute difference between the n_features-dim feature vectors and pass it to the last FC-Layer
        difference = torch.abs(inputs1 - inputs2)
        outputs = model.fc_end(difference)

        outputs = outputs.squeeze()
        labels = labels.type_as(outputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print('Epoch [%d/%d], Iteration [%d/%d], Loss: %.4f' % 
              (epoch + 1, n_epochs, i + 1,
               math.ceil(n_samples / batch_size),
               loss.item()))

    # Compute the average loss per epoch
    training_loss = running_loss / math.ceil(n_samples / batch_size)
    return training_loss


def validate(net, validation_loader, n_samples, batch_size, criterion, epoch, n_epochs):
    net.eval()
    running_loss = 0

    print('Validating----->')
    # Handle DistributedDataParallel wrapper
    model = net.module if hasattr(net, 'module') else net
    with torch.no_grad():
        for i, batch in enumerate(validation_loader):
            inputs1, inputs2, labels= batch
            inputs1, inputs2, labels = inputs1.cuda(), inputs2.cuda(), labels.cuda()


            # Compute the absolute difference between the n_features-dim feature vectors and pass it to the last FC-Layer
            difference = torch.abs(inputs1 - inputs2)
            outputs = model.fc_end(difference)

            outputs = outputs.squeeze()
            labels = labels.type_as(outputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            print('Epoch [%d/%d], Iteration [%d/%d], Loss: %.4f' % 
              (epoch + 1, n_epochs, i + 1,
               math.ceil(n_samples / batch_size),
               loss.item()))

    # Compute the average loss per epoch
    validation_loss = running_loss / math.ceil(n_samples / batch_size)
    return validation_loss


def test(net, test_loader):
    net.eval()
    y_true = None
    y_pred = None

    print('Testing----->')
    # Handle DistributedDataParallel wrapper
    model = net.module if hasattr(net, 'module') else net
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs1, inputs2, labels= batch
            inputs1, inputs2 = inputs1.cuda(), inputs2.cuda()


            # Compute the absolute difference between the n_features-dim feature vectors and pass it to the last FC-Layer
            difference = torch.abs(inputs1 - inputs2)
            outputs = model.fc_end(difference)

            outputs = outputs.squeeze()
            labels = labels.type_as(outputs)


            if y_true is None:
                y_true = labels.cpu()
            else:
                y_true = torch.cat((y_true, labels.cpu()), 0)

            if y_pred is None:
                y_pred = outputs.cpu()
            else:
                y_pred = torch.cat((y_pred, outputs.cpu()), 0)

    y_pred = y_pred.squeeze()
    return y_true, y_pred