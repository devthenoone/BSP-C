import copy

import torch
from loaddata import ADMdataset
from torch.utils.data.dataloader import DataLoader

from model import ConcatHNN2FC
import numpy as np
from tqdm import *

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from itertools import cycle

from torch.autograd import Variable

import matplotlib.pyplot as plt
import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from tool import specificity, ACC

decay=0.9

train_txt='train-oasis.txt'
valid_txt='val-oasis.txt'

model=ConcatHNN2FC()
model.cuda()

traindata=ADMdataset(train_txt)
validdata=ADMdataset(valid_txt)

train_loader = DataLoader(traindata, batch_size=8, num_workers=0, pin_memory=True, shuffle=True, drop_last=True)
valid_loader = DataLoader(validdata, batch_size=8, num_workers=0, pin_memory=True, shuffle=False, drop_last=True)

loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0023, weight_decay=0.001)
train_data_size = len(traindata)
print(train_data_size)
valid_data_size = len(validdata)


Use_gpu = torch.cuda.is_available()
if Use_gpu:
    model = model.cuda()

epoch_n = 30
time_open = time.time()
best_acc = 0.0
best_epoch = 0
best_model_wts = copy.deepcopy(model.state_dict())

training_loss = []
val_loss = []
training_accuracy = []
val_accuracy = []
num_class = 3

score_list = []
label_list = []
epoch_record1 = []
epoch_record2 = []

classes = ('AD', 'CN', 'MCI')

for epoch in range(epoch_n):
    epoch_start = time.time()

    epoch_record3 = []

    epoch_record11 = []
    epoch_record12 = []
    epoch_record13 = []

    model.train()

    train_loss = 0.0
    train_acc = 0.0
    valid_loss = 0.0
    valid_acc = 0.0
    correct = 0


    print('epoch {}/{}'.format(epoch, epoch_n - 1))
    print('-' * 10)
    pbar = tqdm(enumerate(train_loader), total=len(traindata), desc=f'epoch {epoch + 1}/{epoch_n}', unit='patient')

    for idx, (image, label, tabular) in pbar:
        X1 = image
        X2 = tabular
        Y = label
        X1, X2, Y = Variable(X1).cuda(), Variable(X2).cuda(), Variable(Y).cuda()
        y_pred = model(X1, X2)
        score_tmp = y_pred
        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(Y.cpu().numpy())
        loss = loss_f(y_pred, Y.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()*X1.size(0)
        ret, predictions = torch.max(y_pred.data, 1)
        correct_counts = predictions.eq(Y.data.view_as(predictions))
        acc=torch.mean(correct_counts.type(torch.FloatTensor))
        train_acc+=acc.item()*X1.size(0)
        print(train_acc / train_data_size)
    score_array = np.array(score_list)

    label_tensor = torch.tensor(label_list, dtype=torch.int64)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_onehot.scatter_(1, label_tensor, 1)
    label_onehot = np.array(label_onehot)
    AUC = roc_auc_score(label_onehot, score_array, multi_class='ovr')
    print(AUC)
    with torch.no_grad():
        model.eval()

        score_list = []
        label_list = []

        for j, (inputs, labels, tabular) in enumerate(valid_loader):
            inputs1 = Variable(inputs).cuda()
            inputs2 = Variable(tabular).cuda()
            labels = Variable(labels).cuda()
            outputs = model(inputs1, inputs2)
            score_tmp = outputs
            score_list.extend(score_tmp.detach().cpu().numpy())
            label_list.extend(labels.cpu().numpy())
            loss = loss_f(outputs, labels.long())
            valid_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            valid_acc += acc.item() * inputs1.size(0)
            epoch_record11.extend(labels.cpu().numpy())
            epoch_record12.extend(predictions.cpu().numpy())
        score_array = np.array(score_list)
        label_tensor = torch.tensor(label_list, dtype=torch.int64)
        label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
        label_onehot = torch.zeros(label_tensor.shape[0], num_class)
        label_onehot.scatter_(1, label_tensor, 1)
        label_onehot = np.array(label_onehot)
        AUC = roc_auc_score(label_onehot, score_array, multi_class='ovr')
        print(AUC)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            print(roc_auc[i])
        lw = 2
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(len(classes)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label = 'ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('epoch {}/{}'.format(epoch, epoch_n - 1))
        plt.legend(loc='lower right')
        plt.show()
        acc = ACC(epoch_record11, epoch_record12, 3)
        print(acc)
        SPE1 = specificity(epoch_record11, epoch_record12, 3)
        print(SPE1)
        cr = classification_report(epoch_record11, epoch_record12, target_names=classes)
        print(cr)
    avg_train_loss = train_loss / train_data_size
    avg_train_acc = train_acc / train_data_size
    training_loss.append(train_loss / train_data_size)
    training_accuracy.append(avg_train_acc)

    avg_valid_loss = valid_loss / valid_data_size
    avg_valid_acc = valid_acc / valid_data_size
    val_loss.append(valid_loss / valid_data_size)
    val_accuracy.append(avg_valid_acc)

    if best_acc < avg_valid_acc:
        best_acc = avg_valid_acc
        best_epoch = epoch + 1
        best_model_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), 'best_model.pt')
    epoch_end = time.time()
    print(
        "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100, epoch_end - epoch_start
        ))
    print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))


plt.plot(np.arange(0, epoch_n), val_loss, label = 'val', marker = 'o')
plt.plot(np.arange(0, epoch_n), training_loss, label = 'train', marker = 'o')
plt.title('loss per epoch 8')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

plt.plot(np.arange(0, epoch_n), val_accuracy, label = 'val_acc', marker = 'x')
plt.plot(np.arange(0, epoch_n), training_accuracy, label = 'train_acc', marker = 'x')
plt.title('Accuracy per epoch 8')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()



