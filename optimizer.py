# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

import json
from functools import partial
from torch import optim as optim


def build_optimizer(config, model, logger, is_pretrain):
    if is_pretrain:
        return build_pretrain_optimizer(config, model, logger)
    else:
        return build_his_finetune_optimizer(config, model, logger)


def build_pretrain_optimizer(config, model, logger):
    logger.info('>>>>>>>>>> Build Optimizer for Pre-training Stage')
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
        logger.info(f'No weight decay: {skip}')
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
        logger.info(f'No weight decay keywords: {skip_keywords}')

    parameters = get_pretrain_param_groups(model, logger, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    logger.info(optimizer)
    return optimizer
    

def get_pretrain_param_groups(model, logger, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    logger.info(f'No decay params: {no_decay_name}')
    logger.info(f'Has decay params: {has_decay_name}')
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def freeze_parameter(model):
    key_word = model.freeze()
    names = []
    for name, param in model.named_parameters():
        if check_keywords_in_name(name, key_word):
            names.append(name)
            param.requires_grad = False
    return names


def build_his_finetune_optimizer(config, model, logger):
    logger.info('>>>>>>>>>> Build Optimizer for fine_tune Stage')
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
        logger.info(f'No weight decay: {skip}')
        logger.info('skip len:', len(skip))
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
        logger.info(f'No weight decay keywords: {skip_keywords}')
        logger.info('skip_keywords len:', len(skip_keywords))
    freeze = freeze_parameter(model)
    logger.info(f'has frozen param:{freeze}')
    parameters = get_pretrain_param_groups(model, logger, skip, skip_keywords)

    opt_lower = 'adamw'
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=1e-08, betas=(0.9, 0.999),
                                lr=6.25e-06, weight_decay=0.05)
    for name, param in enumerate(optimizer.param_groups[0]):
        logger.info(name)
    return optimizer


def build_finetune_optimizer(config, model, logger):
    logger.info('>>>>>>>>>> Build Optimizer for Fine-tuning Stage')
    if config.MODEL.TYPE == 'swin':
        depths = config.MODEL.SWIN.DEPTHS
        num_layers = sum(depths)
        get_layer_func = partial(get_swin_layer, num_layers=num_layers + 2, depths=depths)
    elif config.MODEL.TYPE == 'vit':
        num_layers = config.MODEL.VIT.DEPTH
        get_layer_func = partial(get_vit_layer, num_layers=num_layers + 2)
    elif config.MODEL.TYPE == 'Dtransformer':
        num_layers = config.MODEL.VIT.DEPTH
        get_layer_func = partial(get_vit_layer, num_layers=num_layers + 2)
    else:
        raise NotImplementedError

    scales = list(config.TRAIN.LAYER_DECAY ** i for i in reversed(range(num_layers + 2)))

    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
        logger.info(f'No weight decay: {skip}')
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
        logger.info(f'No weight decay keywords: {skip_keywords}')

    parameters = get_finetune_param_groups(
        model, logger, config.TRAIN.BASE_LR, config.TRAIN.WEIGHT_DECAY,
        get_layer_func, scales, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    logger.info(optimizer)
    return optimizer


def get_vit_layer(name, num_layers):
    if name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("rel_pos_bias"):
        return num_layers - 1
    elif name.startswith("blocks"):
        layer_id = int(name.split('.')[1])
        return layer_id + 1
    else:
        return num_layers - 1


#  根据网络变量名返回其在哪层
def get_swin_layer(name, num_layers, depths):
    # mask——token 与 patch——embd 都是第0层的
    if name in ("mask_token"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("layers"):
        layer_id = int(name.split('.')[1])
        block_id = name.split('.')[3]
        if block_id == 'reduction' or block_id == 'norm':
            return sum(depths[:layer_id + 1])
        layer_id = sum(depths[:layer_id]) + int(block_id)
        return layer_id + 1
    else:
        return num_layers - 1


def get_finetune_param_groups(model, logger, lr, weight_decay, get_layer_func, scales, skip_list=(), skip_keywords=()):
    parameter_group_names = {}
    parameter_group_vars = {}
    # 读取的是网络的参数。名字和值
    for name, param in model.named_parameters():
        # 不需要梯度 下一个
        if not param.requires_grad:
            continue
        #
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale
            }
        # 添加值和名字
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    logger.info("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
