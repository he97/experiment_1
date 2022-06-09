from __future__ import print_function
import os
import random
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import time

from models.Trans_BCDM_A.net_A import *
from models.Trans_BCDM_A.utils_A import *

import numpy as np

batch_size = 64
epochs = 200
lr = 1e-6
gamma = 0.9

nDataSet = 10
HalfWidth = 2
SAMPLE_NUM = 180

# Update
dim = 48
patch_dim = 512 #512
CLASS_NUM = 7
os.chdir('C:/Users/Dell/hwq/git_code/SimMIM')
SRC_PATH = '/dataset/houston13-18/Houston13.mat'
SRC_LABEL_PATH = '/dataset/houston13-18/Houston13_7gt.mat'
TGT_PATH = '/dataset/houston13-18/Houston18.mat'
TGT_LABEL_PATH = '/dataset/houston13-18/Houston18_7gt.mat'

file_path = '/data/Yeahomous/Datasets/indian(Jiasen)/indian_220.mat'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


use_cuda = torch.cuda.is_available()
device = 'cuda'

acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
K = np.zeros([nDataSet, 1])

seeds = [1330, 1220, 1336, 1337, 1334, 1236, 1226, 1235, 1228, 1229]

# Load data
source_img, source_label, target_img, target_label = cubeData(SRC_PATH, SRC_LABEL_PATH , TGT_PATH, TGT_LABEL_PATH)
# source_img, source_label, target_img, target_label = cubeData1(file_path)

# Load test data
test_img, test_label = get_all_data(target_img, target_label, HalfWidth)  # 目标域全部样本
test_dataset = TensorDataset(torch.tensor(test_img), torch.tensor(test_label))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

all_time_start =  time.time()
for iDataSet in range(nDataSet):
    seed_everything(seeds[iDataSet])
    # Load train data
    src_img, src_label = get_sample_data(source_img, source_label, HalfWidth, SAMPLE_NUM)
    train_dataset = TensorDataset(torch.tensor(src_img), torch.tensor(src_label))
    src_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    tgt_train_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


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

    C1 = T_Classifier(
        num_classes=CLASS_NUM,
        attn_layers=Encoder(
            dim=patch_dim,
            depth=1,
            heads=2),
        dropout=0.1).to(device)
    C2 = T_Classifier(
        num_classes=CLASS_NUM,
        attn_layers=Encoder(
            dim=patch_dim,
            depth=1,
            heads=2),
        dropout=0.1).to(device)

    # C1 = ResClassifier(num_classes = CLASS_NUM, num_unit=patch_dim, middle=1024).to(device)
    # C2 = ResClassifier(num_classes = CLASS_NUM, num_unit=patch_dim, middle=1024).to(device)
    # C1 = Classifier(patch_dim, dim_out = CLASS_NUM).to(device)
    # C2 = Classifier(patch_dim, dim_out = CLASS_NUM).to(device)
    # C1 = Linear_Classifier(patch_dim, dim_out=CLASS_NUM).to(device)
    # C2 = Linear_Classifier(patch_dim, dim_out=CLASS_NUM).to(device)

    # C1.apply(classifier_weights_init)
    # C2.apply(classifier_weights_init)

    E_optim = optim.Adam(E.parameters(), lr=lr)
    C_optim = optim.Adam(list(C1.parameters()) + list(C2.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss().cuda()

    eta = 0.01

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
            NUM_K = 4
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
            output = E(valX)
            output1 = C1(output)
            output2 = C2(output)
            output_add = output1 + output2
            _, predicted = torch.max(output_add.data, 1)
            total += valY.size(0)
            val_all = np.concatenate([val_all, valY.data.cpu().numpy()])
            val_pred_all = np.concatenate([val_pred_all, predicted.cpu().numpy()])
            correct += predicted.eq(valY.data.view_as(predicted)).cpu().sum().item()
        test_accuracy = 100. * correct / total

        acc[iDataSet] = test_accuracy
        OA = acc
        C = metrics.confusion_matrix(val_all, val_pred_all)
        A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)
        K[iDataSet] = metrics.cohen_kappa_score(val_all, val_pred_all)
        print('\tRun: {}\tval_Accuracy: {}/{} ({:.2f}%)\t'.format(iDataSet,correct,total,100. * correct / total))

all_time_end = time.time()
all_time = (all_time_end - all_time_start) / 3600
AA = np.mean(A, 1)
AAMean = np.mean(AA,0)
AAStd = np.std(AA)
AMean = np.mean(A, 0)
AStd = np.std(A, 0)
OAMean = np.mean(acc)
OAStd = np.std(acc)
kMean = np.mean(K)
kStd = np.std(K)
for iDataSet in range(nDataSet):
    print('Run: {}\tval_Accuracy: ({:.2f}%)\t'.format(iDataSet, acc[iDataSet][0]))
print ("RUN TIME: " + "{:.5f}".format(all_time))
print ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
print ("accuracy for each class: ")
for i in range(CLASS_NUM):
    print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))
if __name__=='__main__':


