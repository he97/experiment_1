# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
import numpy as np
from scipy import interpolate
import scipy.io as sio
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader

from data.data_simmim import HsiMaskTensorDataSet, HsiMaskGenerator

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f">>>>>>>>>> Resuming from {config.MODEL.RESUME} ..........")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir, logger):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    logger.info(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        logger.info(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def load_pretrained(config, model, logger):
    logger.info(f">>>>>>>>>> Fine-tuned from {config.PRETRAINED} ..........")
    checkpoint = torch.load(config.PRETRAINED, map_location='cpu')
    checkpoint_model = checkpoint['model']
    # 发现了预训练模型 去除encoder.前缀
    if any([True if 'encoder.' in k else False for k in checkpoint_model.keys()]):
        checkpoint_model = {k.replace('encoder.', ''): v for k, v in checkpoint_model.items() if
                            k.startswith('encoder.')}
        logger.info('Detect pre-trained model, remove [encoder.] prefix.')
    else:
        logger.info('Detect non-pre-trained model, pass without doing anything.')

    if config.MODEL.TYPE == 'swin':
        logger.info(f">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
        checkpoint = remap_pretrained_keys_swin(model, checkpoint_model, logger)
    elif config.MODEL.TYPE == 'vit':
        logger.info(f">>>>>>>>>> Remapping pre-trained keys for VIT ..........")
        checkpoint = remap_pretrained_keys_vit(model, checkpoint_model, logger)
    elif config.MODEL.TYPE == 'Dtransformer':
        logger.info(f">>>>>>>>>> Remapping pre-trained keys for Dtransformer ..........")
        checkpoint = remap_pretrained_keys_Dtransformer(model, checkpoint_model, logger)
    else:
        raise NotImplementedError
    # 这步是将保存的dict加载到现在的模型上
    msg = model.load_state_dict(checkpoint_model, strict=False)
    logger.info(msg)

    del checkpoint
    torch.cuda.empty_cache()
    logger.info(f">>>>>>>>>> loaded successfully '{config.PRETRAINED}'")


def remap_pretrained_keys_swin(model, checkpoint_model, logger):
    state_dict = model.state_dict()

    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            relative_position_bias_table_pretrained = checkpoint_model[key]
            relative_position_bias_table_current = state_dict[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                logger.info(f"Error in loading {key}, passing......")
            else:
                if L1 != L2:
                    logger.info(f"{key}: Interpolate relative_position_bias_table using geo.")
                    src_size = int(L1 ** 0.5)
                    dst_size = int(L2 ** 0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    logger.info("Original positions = %s" % str(x))
                    logger.info("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = relative_position_bias_table_pretrained[:, i].view(src_size, src_size).float().numpy()
                        f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(torch.Tensor(f_cubic(dx, dy)).contiguous().view(-1, 1).to(
                            relative_position_bias_table_pretrained.device))

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    checkpoint_model[key] = new_rel_pos_bias

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in checkpoint_model.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [k for k in checkpoint_model.keys() if "relative_coords_table" in k]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del checkpoint_model[k]

    return checkpoint_model


def remap_pretrained_keys_Dtransformer(model, checkpoint_model, logger):
    logger.info('delete pos_embedding cls_token mask_token')
    # checkpoint_model.pop('pos_embedding')
    # checkpoint_model.pop('cls_token')
    # checkpoint_model.pop('mask_token')
    return checkpoint_model


def remap_pretrained_keys_vit(model, checkpoint_model, logger):
    # Duplicate shared rel_pos_bias to each layer
    if getattr(model, 'use_rel_pos_bias', False) and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
        logger.info("Expand the shared relative position embedding to each transformer block.")
        #     获取网络层数 为的是将
        #     我感觉这代码写得有问题啊，if语句的范围有错误吧 改一下（5.30）
        #     而且我自己并没有在每个attention的block中使用绝对位置编码
        num_layers = model.get_num_layers()
        rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
        for i in range(num_layers):
            checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()
        checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")

    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    # 获取所有key的名字
    # 没什么
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)
        # 在相对位置 偏差表
        if "relative_position_bias_table" in key:
            # 相对位置编码的具体数值
            rel_pos_bias = checkpoint_model[key]
            # 保存模型的 样本数量 注意力头数量
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            # 现在模型的 样本数量 注意头数量
            dst_num_pos, _ = model.state_dict()[key].size()
            # 现在模型的patch的数目， self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
            dst_patch_shape = model.patch_embed.patch_shape
            # patch 必须是n*n?
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()

            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                logger.info(
                    "Position interpolate for %s from %dx%d to %dx%d" % (key, src_size, src_size, dst_size, dst_size))
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.090307:
                #     q = 1.090307

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                logger.info("Original positions = %s" % str(x))
                logger.info("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                checkpoint_model[key] = new_rel_pos_bias

    return checkpoint_model


# 获取高光谱数据
def cubeData1(img_path, label_path):
    temp1 = sio.loadmat(img_path)
    print(temp1.keys())
    data1 = temp1[list(temp1.keys())[-2]]
    print(data1.shape)
    temp2 = sio.loadmat(label_path)
    print(temp2.keys())
    gt1 = temp2[list(temp2.keys())[-1]]
    print(gt1.shape)
    #6.5 之前没有。可能是因为从别的文件中复制过来的，那个没使用归一化？
    data_s = data1.reshape(np.prod(data1.shape[:2]), np.prod(data1.shape[2:]))  # (111104,204)
    data_scaler_s = preprocessing.scale(data_s)  #标准化 (X-X_mean)/X_std,
    Data_Band_Scaler_s = data_scaler_s.reshape(data1.shape[0], data1.shape[1],data1.shape[2])

    return data1, gt1
    # return Data_Band_Scaler_s, gt1


"""
Halfwidth=2
"""


def get_sample_data_without_train_val(Sample_data, Sample_label, HalfWidth, num_per_class):
    print('get_sample_data() run...')
    print('The original sample data shape:', Sample_data.shape)
    # 波段数
    nBand = Sample_data.shape[2]
    # 数据变成了 214 958 48 原因是因为之后要将有标记的点旁边的值一起取出来
    data = np.pad(Sample_data, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode='constant')

    label = np.pad(Sample_label, HalfWidth, mode='constant')

    train = {}
    train_indices = []
    # 返回索引
    [Row, Column] = np.nonzero(label)
    # 类别数
    m = int(np.max(label))
    print(f'num_class : {m}')
    for i in range(m):
        # ravel: return a 1D array
        # 类别数 从 1 到 m 0代表的应该是默认情况
        # 这个索引的值，是column和row的索引值，下面的式子
        # [j for j, x in enumerate(len(Row)) if label[Row[j], Column[j]] == i + 1]
        # 等价的
        # 本质就是把所有不为0的点，都拿出来查一遍
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if label[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        # num_per_class 有什么作用？ 选择训练和测试的数目
        num_per_class = int(len(indices))
        train[i] = indices[:num_per_class]
        # val[i] = indices[num_per_class:]

    for i in range(m):
        train_indices += train[i]
        # val_indices += val[i]
    #     再次打乱
    np.random.shuffle(train_indices)
    # np.random.shuffle(val_indices)

    # #val
    # print('the number of val data:', len(val_indices))
    # nVAL = len(val_indices)
    # val_data = np.zeros([nVAL, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    # val_label = np.zeros([nVAL], dtype=np.int64)
    # RandPerm = val_indices
    # RandPerm = np.array(RandPerm)
    #
    # for i in range(nVAL):
    #     # 将原数据的不为0的点作为中心，一个5*5的块，整体作为数据。
    #     val_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
    #                                               Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1,
    #                                               :],
    #                                               (2, 0, 1))
    #     val_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    # val_label = val_label - 1

    # train
    # 和之前test的做同样的处理
    print('the number of processed data:', len(train_indices))
    nTrain = len(train_indices)
    index = np.zeros([nTrain], dtype=np.int64)
    processed_data = np.zeros([nTrain, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    processed_label = np.zeros([nTrain], dtype=np.int64)
    RandPerm = train_indices
    RandPerm = np.array(RandPerm)

    for i in range(nTrain):
        index[i] = i
        processed_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                                  Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1,
                                                  :],
                                                  (2, 0, 1))
        processed_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    processed_label = processed_label - 1

    print('sample data shape', processed_data.shape)
    print('sample label shape', processed_label.shape)
    print('get_sample_data() end...')
    return processed_data, processed_label


def get_hsi_dataloader(config):
    halfwidth = 2
    img_source, label_source = cubeData1(config.DATA.DATA_SOURCE_PATH, config.DATA.LABEL_SOURCE_PATH)
    img_target, label_target = cubeData1(config.DATA.DATA_TARGET_PATH, config.DATA.LABEL_TARGET_PATH)
    source_samples = get_sample_data_without_train_val(img_source, label_source, halfwidth, 0)
    target_samples = get_sample_data_without_train_val(img_target, label_target, halfwidth, 0)

    all_samples = np.concatenate((source_samples[0], target_samples[0]))
    all_labels = np.concatenate((source_samples[1], target_samples[1]))

    transform = HsiMaskGenerator(config.DATA.MASK_RATIO, all_samples.shape[1],
                                 mask_patch_size=config.DATA.MASK_PATCH_SIZE)

    dataset = HsiMaskTensorDataSet(torch.tensor(all_samples), torch.tensor(all_labels), transform=transform)
    data_loader = DataLoader(dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=False, num_workers=0, sampler=None,
                             pin_memory=True, drop_last=True)

    return data_loader


"""
Halfwidth=2
"""


def get_sample_data(Sample_data, Sample_label, HalfWidth, num_per_class, train_ratio=0.8):
    print('get_sample_data() run...')
    print('The original sample data shape:', Sample_data.shape)
    assert train_ratio <= 1, "train_ratio is too big"
    # 波段数
    nBand = Sample_data.shape[2]
    # 数据变成了 214 958 48
    data = np.pad(Sample_data, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode='constant')

    label = np.pad(Sample_label, HalfWidth, mode='constant')

    train = {}
    train_indices = []
    # 返回索引
    [Row, Column] = np.nonzero(label)
    # 类别数
    m = int(np.max(label))
    print(f'num_class : {m}')

    val = {}
    val_indices = []

    for i in range(m):
        # ravel: return a 1D array
        # 类别数 从 1 到 m 0代表的应该是默认情况
        # 这个索引的值，是column和row的索引值，下面的式子
        # [j for j, x in enumerate(len(Row)) if label[Row[j], Column[j]] == i + 1]
        # 等价的
        # 本质就是把所有不为0的点，都拿出来查一遍
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if label[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        # num_per_class 有什么作用？ 选择训练和测试的数目
        num_per_class = int(train_ratio * len(indices))
        train[i] = indices[:num_per_class]
        val[i] = indices[num_per_class:]

    for i in range(m):
        train_indices += train[i]
        val_indices += val[i]
    #     再次打乱
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    # val
    print('the number of val data:', len(val_indices))
    nVAL = len(val_indices)
    val_data = np.zeros([nVAL, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    val_label = np.zeros([nVAL], dtype=np.int64)
    RandPerm = val_indices
    RandPerm = np.array(RandPerm)

    for i in range(nVAL):
        # 将原数据的不为0的点作为中心，一个5*5的块，整体作为数据。
        val_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                            Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1,
                                            :],
                                            (2, 0, 1))
        val_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    val_label = val_label - 1

    # train
    # 和之前test的做同样的处理
    print('the number of processed data:', len(train_indices))
    nTrain = len(train_indices)
    index = np.zeros([nTrain], dtype=np.int64)
    processed_data = np.zeros([nTrain, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    processed_label = np.zeros([nTrain], dtype=np.int64)
    RandPerm = train_indices
    RandPerm = np.array(RandPerm)

    for i in range(nTrain):
        index[i] = i
        processed_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                                  Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1,
                                                  :],
                                                  (2, 0, 1))
        processed_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    processed_label = processed_label - 1

    print('sample data shape', processed_data.shape)
    print('sample label shape', processed_label.shape)
    print('get_sample_data() end...')
    return processed_data, processed_label, val_data, val_label


"""
6.1 设置训练集为源域样本，测试集为目标域样本
"""


def get_hsi_train_and_val_dataset(config):
    halfwidth = 2
    img_source, label_source = cubeData1(config.DATA.DATA_SOURCE_PATH, config.DATA.LABEL_SOURCE_PATH)
    img_target, label_target = cubeData1(config.DATA.DATA_TARGET_PATH, config.DATA.LABEL_TARGET_PATH)
    # 源域分为训练集，测试集
    # source_samples = get_sample_data(img_source, label_source, halfwidth, 0,train_ratio=1)
    source_samples = get_sample_data(img_source, label_source, halfwidth, 0, train_ratio=0.8)
    target_samples = get_sample_data(img_target, label_target, halfwidth, 0, train_ratio=0.8)
    # 6.3 以后测试下样本是不是均匀分布在测试集和训练集中
    train_samples = np.concatenate((source_samples[0], target_samples[0]))
    train_labels = np.concatenate((source_samples[1], target_samples[1]))
    val_samples = np.concatenate((source_samples[2], target_samples[2]))
    val_labels = np.concatenate((source_samples[3], target_samples[3]))
    train_dataset = TensorDataset(torch.tensor(train_samples), torch.tensor(train_labels))
    val_dataset = TensorDataset(torch.tensor(val_samples), torch.tensor(val_labels))
    # transform = HsiMaskGenerator(config.DATA.MASK_RATIO, train_samples.shape[1],mask_patch_size=config.DATA.MASK_PATCH_SIZE)

    # train_dataset = HsiMaskTensorDataSet(torch.tensor(train_samples),torch.tensor(train_labels),transform= transform)
    # val_dataset = HsiMaskTensorDataSet(torch.tensor(val_samples), torch.tensor(val_labels), transform=transform)

    # train_data_loader = DataLoader(train_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=False, num_workers=0,
    #                                sampler=None,
    #                                pin_memory=True, drop_last=True)
    return train_dataset, val_dataset


def get_virtual_dataset(config, train_size, val_size, tensor_type='full'):
    """
    获取虚拟数据集，默认的type为full
    :param tensor_type:
    :param config:
    :param train_size:
    :param val_size:
    :return:
    """
    train_dataset = get_tensor_dataset(train_size, tensor_type=tensor_type, have_label=True)
    val_dataset = get_tensor_dataset(val_size, tensor_type=tensor_type, have_label=True)
    return train_dataset, val_dataset



def get_tensor_dataset(size, tensor_type, have_label=True):
    """
    通过size和类型设置张量。
    eye 不能用 没改了
    :param size:
    :param tensor_type:
    :param have_label:
    :return:
    """
    vector = torch.randn(size)
    if tensor_type == 'eye':
        vector = torch.eye(size)
    elif tensor_type == 'randn':
        vector = torch.randn(size)
    elif tensor_type == 'randint':
        vector = torch.randint(1, 3, size)
    elif tensor_type == 'full':
        vector = torch.full(size, 1)
    elif tensor_type == 'arange':
        s = 1
        for x in size:
            s *= x
        vector = torch.arange(s)
        vector.reshape(size)
    if have_label:
        label = torch.full((size[0],), 1.0)
        return TensorDataset(vector, label)
    else:
        return TensorDataset(vector)
