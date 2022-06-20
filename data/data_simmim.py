# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import math
import random
import numpy as np

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class HsiMaskGenerator:
    #     我要做的 应该就是在48个channel 随机选择几个channel就可以了
    def __init__(self, mask_ratio=0.6, in_channel=48,mask_patch_size = 1):
        self.mask_ratio = mask_ratio
        self.in_channel = in_channel
        assert in_channel % mask_patch_size == 0, 'in_channel not match mask_patch_size'
        self.mask_patch_size = mask_patch_size
        self.mask_patch_count = self.in_channel // self.mask_patch_size
        self.mask_count = int(np.ceil(self.mask_patch_count * self.mask_ratio))

    #     def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
    #         self.input_size = input_size
    #         self.mask_patch_size = mask_patch_size
    #         self.model_patch_size = model_patch_size
    #         self.mask_ratio = mask_ratio

    #         assert self.input_size % self.mask_patch_size == 0
    #         assert self.mask_patch_size % self.model_patch_size == 0

    #         self.rand_size = self.input_size // self.mask_patch_size
    #         # 尺度 一个mask点代表几个原本的图片的点
    #         self.scale = self.mask_patch_size // self.model_patch_size

    #         self.token_count = self.rand_size ** 2
    #         # 有多少个mask的区域
    #         self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
    def __call__(self):
        mask_idx = np.random.permutation(self.mask_patch_count)[:self.mask_count]
        # 建立一个所有token的零数组
        mask = np.zeros(self.mask_patch_count, dtype=int)
        # 被选取中的mask区域置1
        mask[mask_idx] = 1
        # mask = mask.transpose()
        # mask = mask[:, np.newaxis]
        return mask
class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask


#     def __call__(self):
#         # 所有区域，随机排序。然后选取前mask_token个
#         mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
#         # 建立一个所有token的零数组
#         mask = np.zeros(self.token_count, dtype=int)
#         # 被选取中的mask区域置1
#         mask[mask_idx] = 1
#         # 重构成二维
#         # 这时的0 1 代表的是mask_patch_si上一个patch.
#         mask = mask.reshape((self.rand_size, self.rand_size))
#         # 转变为model_patch_size
#         mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

#         return mask

def to_img(img):
    if img.mode!='RGB':
        return img.convert('RGB')
    else:
        return img
class SimMIMTransform:

    def __init__(self, config):

        # self.transform_img = None
        self.transform_img = T.Compose([
            T.Lambda(to_img),
            T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
 
        if config.MODEL.TYPE == 'swin':
            model_patch_size=config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size=config.MODEL.VIT.PATCH_SIZE
        elif config.MODEL.TYPE == 'Dtransformer':
            model_patch_size = config.MODEL.Dtransformer.PATCH_SIZE
        else:
            raise NotImplementedError
        
        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        
        return img, mask


from torch.utils.data import Dataset


class HsiMaskTensorDataSet(Dataset):
    def __init__(self, data, label, transform=None):
        """
        dataset_type: ['train', 'test']
        """

        self.transform = transform
        self.label = label
        self.data = data

    def __getitem__(self, index):
        img = self.data[index]
        mask = self.transform()
        _ = self.label[index]
        return img, mask, _

    def __len__(self):
        return self.data.size(0)



class SimMIMTransformForHsi:

    def __init__(self, config):

        # self.transform_img = None
        # self.transform_img = T.Compose([
        #     T.Lambda(to_img),
        #     T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
        #     T.RandomHorizontalFlip(),
        #     T.ToTensor(),
        #     T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD)),
        # ])

        if config.MODEL.TYPE == 'swin':
            model_patch_size = config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size = config.MODEL.VIT.PATCH_SIZE
        elif config.MODEL.TYPE == 'Dtransformer':
            model_patch_size = config.MODEL.Dtransformer.PATCH_SIZE
        else:
            raise NotImplementedError

        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )

    def __call__(self, img):
        # img = img
        mask = self.mask_generator()

        return img, mask


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_loader_simmim(config, logger):
    transform = SimMIMTransform(config)
    logger.info(f'Pre-train data transform:\n{transform}')

    dataset = ImageFolder(config.DATA.DATA_PATH, transform)
    logger.info(f'Build dataset: train images = {len(dataset)}')
    # sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    # dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler, num_workers=config.DATA.NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    if config.IS_DIST:
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)

        dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler, num_workers=config.DATA.NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    else:
        dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=None, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=True, drop_last=True, collate_fn=collate_fn,shuffle=True)
    
    return dataloader