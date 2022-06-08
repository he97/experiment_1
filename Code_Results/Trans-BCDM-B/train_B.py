from __future__ import print_function
import os
import random
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import time
import matplotlib.pyplot as plt
from models.net_B import *
from utils_B import *

batch_size = 128
epochs = 1000
lr = 1e-6
gamma = 0.9
seed = 1330

nDataSet = 10
HalfWidth = 2
SAMPLE_NUM = 200

# Update
dim = 48
patch_dim = 512 #512
CLASS_NUM = 7

SRC_PATH = '../Datasets/houston13-18/Houston13.mat'
SRC_LABEL_PATH = '../Datasets/houston13-18/Houston13_7gt.mat'
TGT_PATH = '../Datasets/houston13-18/Houston18.mat'
TGT_LABEL_PATH = '../Datasets/houston13-18/Houston18_7gt.mat'

file_path = '/data/Yeahomous/Datasets/indian(Jiasen)/indian_220.mat'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed)
use_cuda = torch.cuda.is_available()
device = 'cuda'

accuracy = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])

seeds = [1330, 1220, 1336, 1337, 1334, 1236, 1226, 1235, 1228, 1229]

# Load data
source_img, source_label, target_img, target_label = cubeData(SRC_PATH, SRC_LABEL_PATH , TGT_PATH, TGT_LABEL_PATH)
# source_img, source_label, target_img, target_label = cubeData1(file_path)

# Load test data
test_img, test_label = get_all_data(target_img, target_label, HalfWidth)  # 目标域全部样本
test_dataset = TensorDataset(torch.tensor(test_img), torch.tensor(test_label))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load train data
src_img, src_label, val_img, val_label = get_sample_data(source_img, source_label, HalfWidth, SAMPLE_NUM)
train_dataset = TensorDataset(torch.tensor(src_img), torch.tensor(src_label))
src_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
tgt_train_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

val_dataset = TensorDataset(torch.tensor(val_img), torch.tensor(val_label))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print('source data shape:')
print(src_img.shape)
print(src_label.shape)
print('target data shape:')
print(test_img.shape)
print(test_label.shape)

E = DTransformer(
    num_patches = dim,
    patch_dim=patch_dim,
    image_size = 5,
    patch_size = 5,
    attn_layers = Encoder(
        dim = patch_dim,
        depth = 2,
        heads = 2)).to(device)
C1 = Classifier(
    num_classes = CLASS_NUM,
    attn_layers = Encoder(
        dim = patch_dim,
        depth = 1,
        heads = 2),
        dropout = 0.1).to(device)
C2 = Classifier(
    num_classes = CLASS_NUM,
    attn_layers = Encoder(
        dim = patch_dim,
        depth = 1,
        heads = 2),
        dropout = 0.1).to(device)
print('#############Transformer model##############')
print(E)
print('#############Classifier model##############')
print(C1)

E_optim = optim.Adam(E.parameters(), lr=lr)
C_optim = optim.Adam(list(C1.parameters()) + list(C2.parameters()), lr=lr)
criterion = nn.CrossEntropyLoss().cuda()

eta = 0.01
best_acc = 0

train_loss = []
val_loss = []
acc_total = []

for epoch in range(epochs):
    E.train()
    C1.train()
    C2.train()

    train_pred_all = []
    train_all = []
    correct = 0
    total = 0

    time_start_per_epoch = time.time()
    for batch_idx, data in enumerate(zip(src_train_loader,tgt_train_loader)):
        (data_s, label_s), (data_t, label_t) = data

        data_s, label_s = data_s.cuda(), label_s.cuda()
        data_t, label_t = data_t.cuda(), label_t.cuda()
        data_all = Variable(torch.cat((data_s, data_t), 0))
        label_s = Variable(label_s)
        bs = len(label_s)

        """source domain discriminative"""
        # Step A train all networks to minimize loss on source
        E_optim.zero_grad()
        C_optim.zero_grad()

        output = E(data_all)
        output1 = C1(output)
        output2 = C2(output)
        output_s1 = output1[:bs, :]
        output_s2 = output2[:bs, :]
        output_t1 = output1[bs:, :]
        output_t2 = output2[bs:, :]
        output_t1 = F.softmax(output_t1, dim=1)
        output_t2 = F.softmax(output_t2, dim=1)
        entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
        entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))
        loss1 = criterion(output_s1, label_s)
        loss2 = criterion(output_s2, label_s)
        all_loss = loss1 + loss2 + 0.01 * entropy_loss
        all_loss.backward()
        E_optim.step()
        C_optim.step()

        """target domain discriminative"""
        # Step B train classifier to maximize discrepancy
        E_optim.zero_grad()
        C_optim.zero_grad()

        output = E(data_all)
        output1 = C1(output)
        output2 = C2(output)
        output_s1 = output1[:bs, :]
        output_s2 = output2[:bs, :]
        output_t1 = output1[bs:, :]
        output_t2 = output2[bs:, :]
        output_t1 = F.softmax(output_t1, dim=1)
        output_t2 = F.softmax(output_t2, dim=1)

        loss1 = criterion(output_s1, label_s)
        loss2 = criterion(output_s2, label_s)
        entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
        entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))
        loss_dis = cdd(output_t1, output_t2)

        F_loss = loss1 + loss2 - eta * loss_dis + 0.01 * entropy_loss
        F_loss.backward()
        C_optim.step()

        # Step C train genrator to minimize discrepancy
        NUM_K = 8
        for i in range(NUM_K):
            E.zero_grad()
            C_optim.zero_grad()

            output = E(data_all)
            features_source = output[:bs, :]
            features_target = output[bs:, :]
            output1 = C1(output)
            output2 = C2(output)
            output_s1 = output1[:bs, :]
            output_s2 = output2[:bs, :]
            output_t1 = output1[bs:, :]
            output_t2 = output2[bs:, :]
            output_t1 = F.softmax(output_t1, dim=1)
            output_t2 = F.softmax(output_t2, dim=1)

            entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
            entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))
            loss_dis = cdd(output_t1, output_t2)
            D_loss = eta * loss_dis + 0.01 * entropy_loss

            D_loss.backward()
            E_optim.step()
    print('Train Ep: {} \tLoss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f} Entropy: {:.6f} '.format(
        epoch, loss1.item(), loss2.item(), loss_dis.item(), entropy_loss.item()))
    time_end_per_epoch = time.time()
    print(f'time_{epoch}_epoch:{(time_end_per_epoch-time_start_per_epoch)}')

    test_iter = 20
    if (epoch + 1) % test_iter == 0:
        E.eval()
        C1.eval()
        C2.eval()
        val_pred_all = []
        val_all = []
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (valX, valY) in enumerate(test_loader):
                valX, valY = valX.cuda(), valY.cuda()

                # with torch.no_grad():
                #     valX = make_feature(valX)

                output = E(valX)
                output1 = C1(output)
                output2 = C2(output)

                output_add = output1 + output2

                _, predicted = torch.max(output_add.data, 1)
                total += valY.size(0)

                val_all = np.concatenate([val_all, valY.data.cpu().numpy()])
                val_pred_all = np.concatenate([val_pred_all, predicted.cpu().numpy()])
                correct += predicted.eq(valY.data.view_as(predicted)).cpu().sum().item()

            acc = 100. * correct / total
            acc_total.append(acc)
            if acc > best_acc:
                best_acc = acc
            print('\tEpoch: {}\tval_Accuracy: {}/{} ({:.2f}%)\tbest_Accuracy:{:.2f}%\n'.format(epoch,correct,total,100. * correct / total,best_acc))

    # Val
    E.eval()
    C1.eval()
    C2.eval()
    with torch.no_grad():
        loss_temp = 0
        for batch_idx, (valX, valY) in enumerate(val_loader):
            valX, valY = valX.cuda(), valY.cuda()

            output = E(valX)
            output1 = C1(output)
            output2 = C2(output)

            loss_temp += ((criterion(output1, valY) + criterion(output2, valY)) / 2)
        val_loss.append(loss_temp/(batch_idx+1))

        loss_temp = 0
        for batch_idx, (valX, valY) in enumerate(src_train_loader):
            valX, valY = valX.cuda(), valY.cuda()

            output = E(valX)
            output1 = C1(output)
            output2 = C2(output)

            loss_temp += ((criterion(output1, valY) + criterion(output2, valY)) / 2)
        train_loss.append(loss_temp/(batch_idx+1))


fig = plt.figure(1)
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2)
acc_len = len(acc_total)
x1 = np.linspace(0, acc_len-1, acc_len)
x2 = np.linspace(0, epochs-1, epochs)

y1 = torch.tensor(train_loss, device='cpu')
y2 = torch.tensor(val_loss, device='cpu')
y3 = torch.tensor(acc_total, device='cpu')

ax1.plot(x1, y3)
ax2.plot(x2, y1, color='blue')
ax2.plot(x2, y2, color='red')
# plt.ylim(0, 2)
plt.show()