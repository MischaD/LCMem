import os
import time
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


# Function to get the data loader.
def get_data_loaders(phase='training',  n_channels=3, transform=None,
                     basedir='./', filelist='list.csv', batch_size=32, shuffle=True, num_workers=16, pin_memory=True, inpaint_size=256, processor=None, is_latent=False):

    dataset = get_data_sets(phase=phase, n_channels=n_channels, transform=transform, basedir=basedir, filelist=filelist, inpaint_size=inpaint_size, processor=processor, is_latent=is_latent)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                 pin_memory=pin_memory)
    return dataloader

def get_data_sets(phase='training',  n_channels=3, transform=None, basedir='./', filelist='list.csv', inpaint_size=256, aug_transform=None, processor=None, unsupervised=False, is_latent=False, distill=False):
    dataset = SiameseNCEDataset(phase=phase, n_channels=n_channels, transform=transform, basedir=basedir, filelist=filelist, aug_transform=aug_transform, processor=processor, unsupervised=unsupervised, is_latent=is_latent, distill=distill)
    return dataset

def get_siamese_data_sets(basedir="./", phase='testing', n_channels=3, transform=None, image_path='./', train_file="", val_file="", test_file="", processor=None):
    """Create SiameseDataset for testing with image pairs"""
    from dataset.SiameseFPIDataset import SiameseDataset
    dataset = SiameseDataset(basedir=basedir, phase=phase, n_channels=n_channels, transform=transform, 
                           image_path=image_path,  
                           train_file=train_file, val_file=val_file, test_file=test_file, processor=processor)
    return dataset


# This function represents the training loop for the standard case where we have two input images and one output node.
def train(net, training_loader, n_samples, batch_size, criterion, optimizer, epoch, n_epochs, alpha=0.5, distill=False, alpha_distill=1.0):
    net.train()
    running_loss = 0.0
    running_nce_loss = 0.0
    running_fc_loss = 0.0
    
    # For tracking iteration time
    running_time = 0.0
    window_size = 5  # Window size for running average
    times = []

    NCELoss = NTXentLoss(temperature=0.07)

    print('Training----->')
    for i, batch in enumerate(training_loader):
        iter_start = time.time()
        
        inputs1, inputs2 = batch
        inputs1, inputs2 = inputs1.cuda(), inputs2.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # Handle DistributedDataParallel wrapper
        model = net.module if hasattr(net, 'module') else net

        if distill: 
            output_student = model.forward_once_logits(inputs1)  # logits
            output_teacher = inputs2 # precomputed, logits 

            #Soften the student logits by applying softmax first and log() second
            nce_loss = torch.nn.functional.mse_loss(output_student, output_teacher)

            pos_end = len(output_student) // 2
            fc_labels = torch.zeros(len(output_student)).cuda()
            fc_labels[:pos_end] = 1
            output_student = torch.cat([output_student[:pos_end], torch.roll(output_student[pos_end:], 1, 0)])

            difference = torch.abs(output_student - output_teacher)
            outputs = model.fc_end(difference)

            outputs = outputs.squeeze()
            fc_loss = criterion(outputs, fc_labels)
            loss = alpha_distill * fc_loss + (1.0 - alpha_distill) * nce_loss
        else: 
            output1 = model.forward_once(inputs1)
            output2 = model.forward_once(inputs2)

            labels = torch.arange(output1.shape[0]).cuda()
            if alpha == 1: 
                nce_loss = torch.tensor([0]).cuda()
            else: 
                nce_loss = NCELoss(torch.cat((output1, output2), dim = 0), torch.cat((labels, labels), dim = 0))

            pos_end = len(output1) // 2

            fc_labels = torch.zeros(len(output1)).cuda()
            fc_labels[:pos_end] = 1
            output1 = torch.cat([output1[:pos_end], torch.roll(output1[pos_end:], 1, 0)])

            # Compute the absolute difference between the n_features-dim feature vectors and pass it to the last FC-Layer
            difference = torch.abs(output1 - output2)
            outputs = model.fc_end(difference)

            outputs = outputs.squeeze()
            labels = labels.type_as(outputs)
            fc_loss = criterion(outputs, fc_labels)
            loss = alpha * fc_loss + (1.0 - alpha) * nce_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_nce_loss += nce_loss.item()
        running_fc_loss += fc_loss.item()

        # Calculate iteration time and running average
        iter_time = time.time() - iter_start
        times.append(iter_time)
        if len(times) > window_size:
            times.pop(0)
        avg_time = sum(times) / len(times)
        # Only print if not using distributed training, or if this is the main process
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print('Epoch [%d/%d], Iteration [%d/%d], Loss: %.4f, NCE Loss: %.4f, FC Loss: %.4f, Avg Time/Iter: %.3fs' % 
                (epoch + 1, n_epochs, i + 1,
                math.ceil(n_samples / batch_size),
                loss.item(), nce_loss.item(), fc_loss.item(), avg_time))

        if torch.distributed.is_initialized(): 
            torch.distributed.barrier()

    # Compute the average loss per epoch
    num_iters = math.ceil(n_samples / batch_size)
    training_loss = running_loss / num_iters
    avg_nce = running_nce_loss / num_iters
    avg_fc = running_fc_loss / num_iters
    return training_loss, avg_nce, avg_fc


# This function represents the validation loop for the standard case where we have two input images and one output node.
def validate(net, validation_loader, n_samples, batch_size, criterion, epoch, n_epochs, alpha=0.5):
    net.eval()
    running_loss = 0
    running_nce_loss = 0.0
    running_fc_loss = 0.0
    NCELoss = NTXentLoss(temperature=0.07)


    print('Validating----->')
    # Handle DistributedDataParallel wrapper
    model = net.module if hasattr(net, 'module') else net
    with torch.no_grad():
        for i, batch in enumerate(validation_loader):
            inputs1, inputs2 = batch
            inputs1, inputs2 = inputs1.cuda(), inputs2.cuda()

            output1 = model.forward_once(inputs1)
            output2 = model.forward_once(inputs2)

            labels = torch.arange(output1.shape[0]).cuda()
            if alpha == 1: 
                nce_loss = torch.tensor([0]).cuda()
            else: 
                nce_loss = NCELoss(torch.cat((output1, output2), dim = 0), torch.cat((labels, labels), dim = 0))

            pos_end = len(output1) // 2

            fc_labels = torch.zeros(len(output1)).cuda()
            fc_labels[:pos_end] = 1
            output1 = torch.cat([output1[:pos_end], torch.roll(output1[pos_end:], 1, 0)])

            # Compute the absolute difference between the n_features-dim feature vectors and pass it to the last FC-Layer
            difference = torch.abs(output1 - output2)
            outputs = model.fc_end(difference)

            outputs = outputs.squeeze()
            labels = labels.type_as(outputs)
            fc_loss = criterion(outputs, fc_labels)
            loss = alpha * fc_loss + (1.0 - alpha) * nce_loss

            running_loss += loss.item()
            running_nce_loss += nce_loss.item()
            running_fc_loss += fc_loss.item()

            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print('Epoch [%d/%d], Iteration [%d/%d], Loss: %.4f, NCE Loss: %.4f, FC Loss: %.4f' % 
                (epoch + 1, n_epochs, i + 1,
                math.ceil(n_samples / batch_size),
                loss.item(), nce_loss.item(), fc_loss.item()))

    # Compute the average loss per epoch
    num_iters = math.ceil(n_samples / batch_size)
    validation_loss = running_loss / num_iters
    avg_nce = running_nce_loss / num_iters
    avg_fc = running_fc_loss / num_iters
    return validation_loss, avg_nce, avg_fc


# This function represents the test loop for the standard case where we have two input images and one output node.
# This function returns the true labels and the predicted values.
def test(net, test_loader):
    net.eval()
    y_true = None
    y_pred = None

    print('Testing----->')
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs1, inputs2, label_tgt, label_src, labels = batch

            if y_true is None:
                y_true = labels
            else:
                y_true = torch.cat((y_true, labels), 0)

            inputs1, inputs2, labels = inputs1.cuda(), inputs2.cuda(), labels.cuda()
            outputs = net(inputs1, inputs2)
            outputs = torch.sigmoid(outputs)

            if y_pred is None:
                y_pred = outputs.cpu()
            else:
                y_pred = torch.cat((y_pred, outputs.cpu()), 0)

    y_pred = y_pred.squeeze()
    return y_true, y_pred


# This function computes some standard evaluation metrics given the true labels and the predicted values.
def get_evaluation_metrics(y_true, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    report = metrics.classification_report(y_true, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    return accuracy, f1_score, precision, recall, report, confusion_matrix


# This function is used to apply a threshold to the predicted values before computing some evaluation metrics.
def apply_threshold(input_tensor, threshold):
    output = np.where(input_tensor > threshold, torch.ones(len(input_tensor)), torch.zeros(len(input_tensor)))
    return output


# This function implements bootstrapping, in order to get the mean AUC value and the 95% confidence interval.
def bootstrap(n_bootstraps, y_true, y_pred, path, experiment_description):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    bootstrapped_scores = []

    f = open(path + experiment_description + '_AUC_bootstrapped.txt', "w+")
    f.write('AUC_bootstrapped\n')

    for i in range(n_bootstraps):
        indices = np.random.randint(0, len(y_pred) - 1, len(y_pred))
        auc = metrics.roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(auc)
        f.write(str(auc) + '\n')
    f.close()
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    auc_mean = np.mean(sorted_scores)
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    f = open(path + experiment_description + '_AUC_confidence.txt', "w+")
    f.write('AUC_mean: %s\n' % auc_mean)
    f.write('Confidence interval for the AUC score: ' + str(confidence_lower) + ' - ' + str(confidence_upper))
    f.close()
    return auc_mean, confidence_lower, confidence_upper


# This is a function that plots the loss curves.
def plot_loss_curves(loss_dict, path, experiment_description):
    plt.figure()
    plt.plot(range(1, len(loss_dict['training']) + 1), loss_dict['training'], label='Training Loss')
    plt.plot(range(1, len(loss_dict['validation']) + 1), loss_dict['validation'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation loss curves')
    plt.legend()
    plt.savefig(path + experiment_description + '_loss_curves.png')


# This is a function that plots the ROC curve.
def plot_roc_curve(fp_rates, tp_rates, path, experiment_description):
    plt.figure()
    plt.plot(fp_rates, tp_rates, label='ROC Curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(path + experiment_description + '_ROC_curve.png')


# This is a function that saves the evaluation metrics to a file.
def save_results_to_file(auc, accuracy, f1_score, precision, recall, report, confusion_matrix, path,
                         experiment_description):
    f = open(path + experiment_description + '_results.txt', "w+")
    f.write('AUC: %s\n' % auc)
    f.write('Accuracy: %s\n' % accuracy)
    f.write('F1-Score: %s\n' % f1_score)
    f.write('Precision: %s\n' % precision)
    f.write('Recall: %s\n' % recall)
    f.write('Classification report: %s\n' % report)
    f.write('Confusion matrix: %s\n' % confusion_matrix)
    f.close()


# This function saves the ROC metrics to a file.
def save_roc_metrics_to_file(fp_rates, tp_rates, thresholds, path, experiment_description):
    f = open(path + experiment_description + '_ROC_metrics.txt', "w+")
    f.write('FP_rate\tTP_rate\tThreshold\n')
    for i in range(len(fp_rates)):
        f.write(str(fp_rates[i]) + '\t' + str(tp_rates[i]) + '\t' + str(thresholds[i]) + '\n')
    f.close()


# This function saves the training and validation loss values to a file.
def save_loss_curves(loss_dict, path, experiment_description):
    f = open(path + experiment_description + '_loss_values.txt', "w+")
    f.write('TrainingLoss\tValidationLoss\n')
    for i in range(len(loss_dict['training'])):
        f.write(str(loss_dict['training'][i]) + '\t' + str(loss_dict['validation'][i]) + '\n')
    f.close()


# This function is used to save a checkpoint. This enables us to resume an experiment at a later point if wanted.
def save_checkpoint(epoch, model, optimizer, scheduler, loss_dict, best_loss, num_bad_epochs, filename='checkpoint.pth'):
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss_dict': loss_dict,
        'best_loss': best_loss,
        'num_bad_epochs': num_bad_epochs
    }
    torch.save(state, filename)


# This function saves the true labels, the predicted vales, and the thresholded values to a file.
def save_labels_predictions(y_true, y_pred, y_pred_thresh, path, experiment_description):
    f = open(path + experiment_description + '_labels_predictions.txt', "w+")
    f.write('Label\tPrediction\tPredictionThresholded\n')
    for i in range(len(y_true)):
        f.write(str(y_true[i]) + '\t' + str(y_pred[i]) + '\t' + str(y_pred_thresh[i]) + '\n')
    f.close()


# This function was utilized to construct the positive and negative pairs needed for training and testing our
# verification network. This function takes filenames as an input argument and returns both the list of tuples and the
# list of corresponding labels. The constructed pairs were later saved to a .txt file. Now available in the folder
# './image_pairs/' (pairs_training.txt, pairs_validation.txt, pairs_testing.txt).
# Example:
#
# train_val_filenames = np.loadtxt('train_val_list.txt', dtype=str)
# test_filenames = np.loadtxt('test_list.txt', dtype=str)
# tuples_train, labels_train = get_tuples_labels(train_val_filenames[:75708])
# tuples_val, labels_val = get_tuples_labels(train_val_filenames[75708:])
# tuples_test, labels_test = get_tuples_labels(test_filenames)
#
def get_tuples_labels(filenames):
    tuples_list = []
    labels_list = []
    patients = []

    for i, element in enumerate(filenames):
        element = element[:-8]
        patients.append(element)

    unique_patients, counts_patients = np.unique(patients, return_counts=True)
    patients_dict = dict(zip(unique_patients, counts_patients))

    start_idx = 0
    patients_files_dict = {}

    for key in patients_dict:
        patients_files_dict[key] = filenames[start_idx:start_idx + patients_dict[key]]
        start_idx += patients_dict[key]

        if len(patients_files_dict[key]) > 1:
            # samples = list(itertools.product(patients_files_dict[key], patients_files_dict[key]))
            samples = list(itertools.combinations(patients_files_dict[key], 2))
            tuples_list.append(samples)
            labels_list.append(np.ones(len(samples)).tolist())

    tuples_list = list(itertools.chain.from_iterable(tuples_list))
    labels_list = list(itertools.chain.from_iterable(labels_list))

    N = len(tuples_list)
    i = 0

    while i < N:
        file1 = random.choice(filenames)
        file2 = random.choice(filenames)

        if file1[:-8] != file2[:-8]:
            sample = (file1, file2)
            tuples_list.append(sample)
            labels_list.append(0.0)
            i += 1

    return tuples_list, labels_list



