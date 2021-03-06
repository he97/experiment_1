# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

from .swin_transformer import build_swin
from .vision_transformer import build_vit
from .simmim import build_simmim
from models.Trans_BCDM_A.net_A import build_Dtransformer_finetune


def build_model(config, is_pretrain=True,is_hsi = False):
    if is_pretrain:
        model = build_simmim(config,is_hsi)
    else:
        model_type = config.MODEL.TYPE
        if model_type == 'swin':
            model = build_swin(config)
        elif model_type == 'vit':
            model = build_vit(config)
        elif model_type == 'Dtransformer':
            model = build_Dtransformer_finetune(config)
        else:
            raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

    return model

