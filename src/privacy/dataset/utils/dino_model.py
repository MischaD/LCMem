import torch
import math 
import tqdm
from pytorch_metric_learning.losses import NTXentLoss

# This function represents the training loop for the standard case where we have two input images and one output node.
def train(net, training_loader, n_samples, batch_size, criterion, optimizer, epoch, n_epochs, alpha=0.5):
    net.train()
    running_loss = 0.0
    running_nce_loss = 0.0
    running_fc_loss = 0.0

    NCELoss = NTXentLoss(temperature=0.07)

    print('Training----->')
    for i, batch in enumerate(training_loader):
        inputs1, inputs2 = batch
        inputs1, inputs2 = inputs1.cuda(), inputs2.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        output1 = net.forward_once(inputs1)
        output2 = net.forward_once(inputs2)

        labels = torch.arange(output1.shape[0]).cuda()
        nce_loss = NCELoss(torch.cat((output1, output2), dim = 0), torch.cat((labels, labels), dim = 0))

        pos_end = len(output1) // 2

        fc_labels = torch.zeros(len(output1)).cuda()
        fc_labels[:pos_end] = 1
        output1 = torch.cat([output1[:pos_end], torch.roll(output1[pos_end:], 1, 0)])

        # Compute the absolute difference between the n_features-dim feature vectors and pass it to the last FC-Layer
        difference = torch.abs(output1 - output2)
        outputs = net.fc_end(difference)

        outputs = outputs.squeeze()
        labels = labels.type_as(outputs)
        fc_loss = criterion(outputs, fc_labels)
        loss = alpha * fc_loss + (1.0 - alpha) * nce_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_nce_loss += nce_loss.item()
        running_fc_loss += fc_loss.item()

        print('Epoch [%d/%d], Iteration [%d/%d], Loss: %.4f, NCE Loss: %.4f, FC Loss: %.4f' % 
              (epoch + 1, n_epochs, i + 1,
               math.ceil(n_samples / batch_size),
               loss.item(), nce_loss.item(), fc_loss.item()))

    # Compute the average loss per epoch
    training_loss = running_loss / math.ceil(n_samples / batch_size)
    return training_loss


# This function represents the validation loop for the standard case where we have two input images and one output node.
def validate(net, validation_loader, n_samples, batch_size, criterion, epoch, n_epochs, alpha=0.5):
    net.eval()
    running_loss = 0
    running_nce_loss = 0.0
    running_fc_loss = 0.0
    NCELoss = NTXentLoss(temperature=0.07)


    print('Validating----->')
    with torch.no_grad():
        for i, batch in enumerate(validation_loader):
            inputs1, inputs2 = batch
            inputs1, inputs2 = inputs1.cuda(), inputs2.cuda()

            output1 = net.forward_once(inputs1)
            output2 = net.forward_once(inputs2)

            labels = torch.arange(output1.shape[0]).cuda()
            nce_loss = NCELoss(torch.cat((output1, output2), dim = 0), torch.cat((labels, labels), dim = 0))

            pos_end = len(output1) // 2

            fc_labels = torch.zeros(len(output1)).cuda()
            fc_labels[:pos_end] = 1
            output1 = torch.cat([output1[:pos_end], torch.roll(output1[pos_end:], 1)])

            # Compute the absolute difference between the n_features-dim feature vectors and pass it to the last FC-Layer
            difference = torch.abs(output1 - output2)
            outputs = net.fc_end(difference)

            outputs = outputs.squeeze()
            labels = labels.type_as(outputs)
            fc_loss = criterion(outputs, fc_labels)
            loss = alpha * fc_loss + (1.0 - alpha) * nce_loss

            running_loss += loss.item()
            running_nce_loss += nce_loss.item()
            running_fc_loss += fc_loss.item()

            print('Epoch [%d/%d], Iteration [%d/%d], Loss: %.4f, NCE Loss: %.4f, FC Loss: %.4f' % 
              (epoch + 1, n_epochs, i + 1,
               math.ceil(n_samples / batch_size),
               loss.item(), nce_loss.item(), fc_loss.item()))

    # Compute the average loss per epoch
    validation_loss = running_loss / math.ceil(n_samples / batch_size)
    return validation_loss


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

