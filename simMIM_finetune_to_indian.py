# Taking the Indian Pines dataset as an example

from __future__ import print_function

import argparse
import os
import random
from sklearn import metrics
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from config import get_config
from data import build_loader
from logger import create_logger
from models import build_model
from Code_Results.CNN_Trans.A.utils_A import *

import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_pretrained

print(f"Torch: {torch.__version__}")


def parse_option():
    """
    获取参数 返回的是args和cfgNode。
    """
    parser = argparse.ArgumentParser('SimMIM pre-training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument('--gamma', default = 0.9,type=float, required=False, help='data of gamma', )
    parser.add_argument('--halfwidth', default=0, type=float, required=False, help='data of seed', )
    parser.add_argument('--sample_num', default=180, type=int, required=False, help='sum of sample in each category', )
    parser.add_argument('--n_class', default=7, type=int, required=False, help='sum of categories', )
    parser.add_argument('--channel_dim', default=48, type=int, required=False, help='dimesion of channel', )

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--data-source-path', type=str, help='path to source dataset')
    parser.add_argument('--label-source-path', type=str, help='path to source label')
    parser.add_argument('--data-target-path', type=str, help='path to target dataset')
    parser.add_argument('--label-target-path', type=str, help='path to target label')
    parser.add_argument('--pretrained', type=str, help='path to pre-trained model')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    # 解析参数
    args = parser.parse_args()
    # 得到yacs cfgNOde，值是原有的值
    config = get_config(args)

    return args, config

# Training settings
# batch_size = 128
# epochs = 80
# lr = 1e-4
# gamma = 0.9
# seed = 0
# HalfWidth = 2
# SAMPLE_NUM = 180
#
# nClass = 16
# dim = 112

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

#
# seed_everything(seed)
# use_cuda = torch.cuda.is_available()
# device = 'cuda'
#
# accuracy = np.zeros([1, 1])
# A = np.zeros([1, nClass])
# k = np.zeros([1, 1])
# # 数据加载
# os.chdir('C:/Users/Dell/hwq/git_code/SimMIM')
# img_path = 'dataset/houston13-18/Houston13.mat'
# label_path = 'dataset/houston13-18/Houston13_7gt.mat'
#
# img, label= cubeData1(img_path, label_path)
#
#
# train_img, train_label, val_img, val_label = get_sample_data(img, label, HalfWidth, SAMPLE_NUM)
#
# train_dataset = TensorDataset(torch.tensor(train_img), torch.tensor(train_label))
# val_dataset = TensorDataset(torch.tensor(val_img), torch.tensor(val_label))
# test_dataset = TensorDataset(torch.tensor(val_img), torch.tensor(val_label))
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# model = DTransformer(
#     num_patches=dim,
#     image_size=5,
#     patch_size=5,
#     num_classes=nClass,
#     attn_layers=Encoder(
#         dim=512,
#         depth=2,
#         heads=2),
#     dropout=0.1).to(device)
# print(model)
# # loss function
# criterion = nn.CrossEntropyLoss()
# # optimizer
# optimizer = optim.Adam(model.parameters(), lr=lr)
# # scheduler
# scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
#
#
# train_loss_all = []
# val_loss_all = []
# acc_all = []

# for epoch in range(epochs):
#     model.train()
#     train_pred_all = []
#     train_all = []
#     correct = 0
#     total = 0
#     for batch_idx, (trainX, trainY) in enumerate(train_loader):
#         trainX, trainY = trainX.cuda(), trainY.cuda()
#         N = trainY.size(0)
#         optimizer.zero_grad()
#         _, _, _, output = model(trainX)
#         train_loss = criterion(output, trainY)
#         train_loss.backward()
#         optimizer.step()
#         _, predicted = torch.max(output.data, 1)
#         total += trainY.size(0)
#
#         train_all = np.concatenate([train_all, trainY.data.cpu().numpy()])
#         train_pred_all = np.concatenate([train_pred_all, predicted.cpu().numpy()])
#         correct += predicted.eq(trainY.data.view_as(predicted)).cpu().sum().item()
#         print('train loss: ', train_loss)
#     train_loss_all.append(train_loss)
#     print('\tEpoch: {}\tTain_Accuracy: {}/{} ({:.2f}%)\tTrain_Loss: {:.6f}\n'.format(epoch,
#                                                                                      correct, len(train_loader.dataset),
#                                                                                      100. * correct / len(
#                                                                                          train_loader.dataset),
#                                                                                      train_loss.item()))
#
#     model.eval()
#     val_pred_all = []
#     val_all = []
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (valX, valY) in enumerate(valid_loader):
#             valX, valY = valX.cuda(), valY.cuda()
#
#             optimizer.zero_grad()
#             _, _, _, output = model(valX)
#             val_loss = criterion(output, valY)
#
#             optimizer.step()
#             _, predicted = torch.max(output.data, 1)
#             total += valY.size(0)
#
#             val_all = np.concatenate([val_all, valY.data.cpu().numpy()])
#             val_pred_all = np.concatenate([val_pred_all, predicted.cpu().numpy()])
#             correct += predicted.eq(valY.data.view_as(predicted)).cpu().sum().item()
#
#         acc = 100. * correct / len(valid_loader.dataset)
#         val_loss_all.append(val_loss)
#         acc_all.append(acc)
#         print('\tEpoch: {}\tval_Accuracy: {}/{} ({:.2f}%)\tval_Loss: {:.6f}\n'.format(epoch,
#                                                                                       correct,
#                                                                                       len(valid_loader.dataset),
#                                                                                       100. * correct / len(
#                                                                                           valid_loader.dataset),
#                                                                                       val_loss.item()))

def caculate(feature, value):
    y = torch.mean(feature, 1)
    value = torch.mean(value, 2)

    NDVI = torch.div(value[:,0]-value[:,2] , value[:,0]+value[:,2])
    PRI = torch.div(value[:,1]-value[:,3] , value[:,1]+value[:,3])
    CII = torch.div(value[:,0] , value[:,2]) - 1

    SS_res_1 = torch.sum(torch.pow(y - NDVI.unsqueeze(1),2),dim=0)
    SS_res_2 = torch.sum(torch.pow(y - PRI.unsqueeze(1), 2), dim=0)
    SS_res_3 = torch.sum(torch.pow(y - CII.unsqueeze(1), 2), dim=0)
    SS_tot = (batch_size-1)*torch.var(y, 0)

    relation1 = 1 - torch.div(SS_res_1, SS_tot)
    relation2 = 1 - torch.div(SS_res_2, SS_tot)
    relation3 = 1 - torch.div(SS_res_3, SS_tot)
    relation = torch.zeros(3,112)
    relation[0, :] = relation1
    relation[1, :] = relation2
    relation[2, :] = relation3

    return relation

# test_pred_all = []
# test_all = []
#
# label_all = []
# feature_all = torch.empty(0,512)
#
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     cnt = 0
#     for batch_idx, (testX, testY) in enumerate(test_loader):
#         testX, testY = testX.cuda(), testY.cuda()
#
#         value, fea, feature, output = model(testX)
#         RR = caculate(fea, value)
#         _, predicted = torch.max(output.data, 1)
#         total += testY.size(0)
#         test_all = np.concatenate([test_all, testY.data.cpu().numpy()])
#         test_pred_all = np.concatenate([test_pred_all, predicted.cpu()])
#         correct += predicted.eq(testY.data.view_as(predicted)).cpu().sum().item()
#
#         if cnt < 30:
#             feature_all = np.concatenate([feature_all, feature.data.cpu().numpy()])
#             label_all = np.concatenate([label_all, testY.data.cpu().numpy()])
#             cnt += 1
#
#     print("正确率test_acc : " + "{:.2f}".format(100. * correct / len(test_loader.dataset)))
#
# accuracy[0] = 100. * (correct) / len(test_loader.dataset)
# OA = accuracy
# C = metrics.confusion_matrix(test_all, test_pred_all)
# A[0, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)
# k = metrics.cohen_kappa_score(test_all, test_pred_all)
# AA = np.mean(A, 1)
#
# print("OA: " + "{:.2f}".format(OA.item()))
# print("AA: " + "{:.2f}".format(100 * AA.item()))
# print("kappa: " + "{:.4f}".format(100 * k))
# print("accuracy for each class: ")
# for i in range(nClass):
#     print("Class " + str(i) + ": " + "{:.2f}".format(100 * A[0, i]))
#
# names = ['NDVI', 'PRI', 'CIred-edge']
# # fig = plt.figure()
# fig, ax = plt.subplots()
# ax = sns.heatmap(RR)
# # ax = sns.heatmap(RR,cmap=plt.cm.Greys,linewidths=0.05,vmax=1,vmin=0,annot=True,annot_kws={'size':6,'weight':'bold'})
# plt.xticks(np.arange(112)+0.5, np.arange(1,113))
# plt.yticks(np.arange(3)+0.5, names)
# plt.show()

if __name__ == '__main__':
    _, config = parse_option()
    torch.cuda.set_device(config.LOCAL_RANK)
    # Training settings
    batch_size = config.DATA.BATCH_SIZE
    epochs = config.TRAIN.EPOCHS
    lr = config.DATA.LEARNING_RATE
    gamma = config.DATA.GAMMA
    seed = config.DATA.SEED
    HalfWidth = config.DATA.HALFWIDTH
    SAMPLE_NUM = config.DATA.SAMPLE_NUM

    nClass = config.DATA.N_CLASS
    dim = config.DATA.CHANNEL_DIM
    # over
    seed_everything(seed)
    use_cuda = torch.cuda.is_available()
    device = 'cuda'
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    accuracy = np.zeros([1, 1])
    A = np.zeros([1, nClass])
    k = np.zeros([1, 1])
    # 数据加载
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config, logger,is_pretrain=False)
    # os.chdir('C:/Users/Dell/hwq/git_code/SimMIM')
    # img_path = 'dataset/houston13-18/Houston13.mat'
    # label_path = 'dataset/houston13-18/Houston13_7gt.mat'
    #
    # img, label = cubeData1(img_path, label_path)
    #
    # train_img, train_label, val_img, val_label = get_sample_data(img, label, HalfWidth, SAMPLE_NUM)
    #
    # train_dataset = TensorDataset(torch.tensor(train_img), torch.tensor(train_label))
    # val_dataset = TensorDataset(torch.tensor(val_img), torch.tensor(val_label))
    # test_dataset = TensorDataset(torch.tensor(val_img), torch.tensor(val_label))
    #
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # model = DTransformer(
    #     num_patches=dim,
    #     image_size=5,
    #     patch_size=5,
    #     num_classes=nClass,
    #     attn_layers=Encoder(
    #         dim=512,
    #         depth=2,
    #         heads=2),
    #     dropout=0.1).to(device)
    model = build_model(config, is_pretrain=False)
    print(model)
    # remap model
    model_without_ddp = model
    if config.PRETRAINED:
        load_pretrained(config, model_without_ddp, logger)
    model = model_without_ddp
    model = model.cuda()

    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    train_loss_all = []
    val_loss_all = []
    acc_all = []

    for epoch in range(epochs):
        model.train()
        train_pred_all = []
        train_all = []
        correct = 0
        total = 0
        for batch_idx, (trainX, trainY) in enumerate(data_loader_train):
            trainX, trainY = trainX.cuda(), trainY.cuda()
            N = trainY.size(0)
            optimizer.zero_grad()
            # _, _, _, output = model(trainX)
            output = model(trainX)
            train_loss = criterion(output, trainY)
            train_loss.backward()
            optimizer.step()
            _, predicted = torch.max(output.data, 1)
            total += trainY.size(0)

            train_all = np.concatenate([train_all, trainY.data.cpu().numpy()])
            train_pred_all = np.concatenate([train_pred_all, predicted.cpu().numpy()])
            correct += predicted.eq(trainY.data.view_as(predicted)).cpu().sum().item()
            # print('train loss: ', train_loss)
        train_loss_all.append(train_loss)
        print('\tEpoch: {}\tTain_Accuracy: {}/{} ({:.2f}%)\tTrain_Loss: {:.6f}\n'.format(epoch,
                                                                                         correct,
                                                                                         len(data_loader_train.dataset),
                                                                                         100. * correct / len(
                                                                                             data_loader_train.dataset),
                                                                                         train_loss.item()))

        model.eval()
        val_pred_all = []
        val_all = []
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (valX, valY) in enumerate(data_loader_val):
                valX, valY = valX.cuda(), valY.cuda()

                optimizer.zero_grad()
                # _, _, _, output = model(valX)
                output = model(valX)
                val_loss = criterion(output, valY)

                optimizer.step()
                _, predicted = torch.max(output.data, 1)
                total += valY.size(0)

                val_all = np.concatenate([val_all, valY.data.cpu().numpy()])
                val_pred_all = np.concatenate([val_pred_all, predicted.cpu().numpy()])
                correct += predicted.eq(valY.data.view_as(predicted)).cpu().sum().item()

            acc = 100. * correct / len(data_loader_val.dataset)
            val_loss_all.append(val_loss)
            acc_all.append(acc)
            print('\tEpoch: {}\tval_Accuracy: {}/{} ({:.2f}%)\tval_Loss: {:.6f}\n'.format(epoch,
                                                                                          correct,
                                                                                          len(data_loader_val.dataset),
                                                                                          100. * correct / len(
                                                                                              data_loader_val.dataset),
                                                                                          val_loss.item()))

    test_pred_all = []
    test_all = []

    label_all = []
    feature_all = torch.empty(0, 512)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        cnt = 0
        for batch_idx, (testX, testY) in enumerate(data_loader_val):
            testX, testY = testX.cuda(), testY.cuda()

            value, fea, feature, output = model(testX)
            RR = caculate(fea, value)
            _, predicted = torch.max(output.data, 1)
            total += testY.size(0)
            test_all = np.concatenate([test_all, testY.data.cpu().numpy()])
            test_pred_all = np.concatenate([test_pred_all, predicted.cpu()])
            correct += predicted.eq(testY.data.view_as(predicted)).cpu().sum().item()

            if cnt < 30:
                feature_all = np.concatenate([feature_all, feature.data.cpu().numpy()])
                label_all = np.concatenate([label_all, testY.data.cpu().numpy()])
                cnt += 1

        print("正确率test_acc : " + "{:.2f}".format(100. * correct / len(data_loader_val.dataset)))

    accuracy[0] = 100. * (correct) / len(data_loader_val.dataset)
    OA = accuracy
    C = metrics.confusion_matrix(test_all, test_pred_all)
    A[0, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)
    k = metrics.cohen_kappa_score(test_all, test_pred_all)
    AA = np.mean(A, 1)

    print("OA: " + "{:.2f}".format(OA.item()))
    print("AA: " + "{:.2f}".format(100 * AA.item()))
    print("kappa: " + "{:.4f}".format(100 * k))
    print("accuracy for each class: ")
    for i in range(nClass):
        print("Class " + str(i) + ": " + "{:.2f}".format(100 * A[0, i]))

    names = ['NDVI', 'PRI', 'CIred-edge']
    # fig = plt.figure()
    fig, ax = plt.subplots()
    ax = sns.heatmap(RR)
    # ax = sns.heatmap(RR,cmap=plt.cm.Greys,linewidths=0.05,vmax=1,vmin=0,annot=True,annot_kws={'size':6,'weight':'bold'})
    plt.xticks(np.arange(112) + 0.5, np.arange(1, 113))
    plt.yticks(np.arange(3) + 0.5, names)
    plt.show()





    # C:/ProgramData/Anaconda3/envs/CGDM/Lib/site-packages/apex/amp/_amp_state.py 修改了调用问题
    # if config.AMP_OPT_LEVEL != "O0":
    #     assert amp is not None, "amp not installed!"

    # if not config.IS_DIST:
    #     os.environ['RANK'] = '-1'
    #     os.environ['world_size'] = '-1'
    #     os.environ['MASTER_ADDR'] = 'localhost'
    #     os.environ['MASTER_PORT'] = '1080'
    #
    # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    #     rank = int(os.environ["RANK"])
    #     world_size = int(os.environ['WORLD_SIZE'])
    #     print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    # else:
    #     rank = -1
    #     world_size = -1
    # torch.cuda.set_device(config.LOCAL_RANK)
    # if config.IS_DIST:
    #     torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    #     torch.distributed.barrier()
    # if config.IS_DIST:
    #     seed = config.SEED + dist.get_rank()
    # else:
    #     seed = config.SEED
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # cudnn.benchmark = True
    # if config.IS_DIST:
    #     # linear scale the learning rate according to total batch size, may not be optimal
    #     linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    #     linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    #     linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # else:
    #     # linear scale the learning rate according to total batch size, may not be optimal
    #     linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
    #     linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
    #     linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
    # # gradient accumulation also need to scale the learning rate
    # if config.TRAIN.ACCUMULATION_STEPS > 1:
    #     linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
    #     linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
    #     linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    # # 先是可变参数，然后变完参数后冻结 上面的没太看懂，去除dist之后查看main
    # config.defrost()
    # config.TRAIN.BASE_LR = linear_scaled_lr
    # config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    # config.TRAIN.MIN_LR = linear_scaled_min_lr
    # config.freeze()
    #
    # os.makedirs(config.OUTPUT, exist_ok=True)
    #
    # # 估摸着也就是看看是不是主机
    # if 1:
    #     path = os.path.join(config.OUTPUT, "config.json")
    #     with open(path, "w") as f:
    #         f.write(config.dump())
    #     logger.info(f"Full config saved to {path}")
    #
    # # print config
    # logger.info(config.dump())
    #
    # main(config)





# # T-SNE
# plt.figure(1)
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
# Y = tsne.fit_transform(feature_all)
# plt.title("feature t-sne")
# plt.scatter(Y[:,0], Y[:,1], s=10, c=label_all)
# # plt.show()
#
# plt.figure(2)
# ax1 = plt.subplot(2,1,1)
# ax2 = plt.subplot(2,1,2)
# acc_len = len(acc_all)
# x1 = np.linspace(0, acc_len-1, acc_len)
# x2 = np.linspace(0, epochs-1, epochs)
#
# y1 = torch.tensor(train_loss_all, device='cpu')
# y2 = torch.tensor(val_loss_all, device='cpu')
# y3 = torch.tensor(acc_all, device='cpu')
#
# ax1.plot(x1, y3)
# ax2.plot(x2, y1, color='blue')
# ax2.plot(x2, y2, color='red')
# # plt.ylim(0, 2)
# plt.show()