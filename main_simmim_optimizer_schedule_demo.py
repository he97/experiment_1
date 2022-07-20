# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np
import torch
import dill
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.utils import AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, cubeData1, \
    get_sample_data_without_train_val, get_hsi_dataloader, get_tensor_dataset, get_mask_dataloader
from PIL import Image
import matplotlib.pyplot as plt
# from apex import amp
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

'''
获取参数 返回的是args和cfgNode。
'''
def parse_option():
    parser = argparse.ArgumentParser('SimMIM pre-training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    # 定义是否分布式
    parser.add_argument('--is_dist', default=False, type=bool, help="is distrubution")
    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-source-path', type=str, help='path to source dataset')
    parser.add_argument('--label-source-path', type=str, help='path to source label')
    parser.add_argument('--data-target-path', type=str, help='path to target dataset')
    parser.add_argument('--label-target-path', type=str, help='path to target label')
    # 继续？继续什么呢
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    # checkpoint pytorch 推出的一个节省缓存的功能
    # action store_true 当在命令行中不指定时 为默认值。 如果加入了 use-checkpoint 不指定值就可以设置为true。粗浅说明
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    # apex 的参数 混合精度加速 choices是对应函数的参数
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    # 输出文件的根目录
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    # 实验的tag
    parser.add_argument('--tag', help='tag of experiment')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    # 解析参数
    args = parser.parse_args()
    # 得到yacs cfgNOde，值是原有的值
    config = get_config(args)

    return args, config


def main(config):
    #
    # data_loader_train = build_loader(config, logger, is_pretrain=True)
    # 测试得到已经加载了train文件夹

    # 加载高光谱数据集
    if on_mac:
        data_loader_train = get_mask_dataloader(config)
    else:
        data_loader_train = get_hsi_dataloader(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{'demo'+config.MODEL.NAME}")
    model = build_model(config, is_pretrain=True,is_hsi=True)
    if not on_mac:
        model.cuda()
    logger.info(str(model))
    # 构建优化器
    optimizer = build_optimizer(config, model, logger, is_pretrain=True)

    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)

    if config.IS_DIST:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # 为什么sample要设置epoch
        # data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, data_loader_train, optimizer, epoch, lr_scheduler)
        # 保存断点的函数吧 先不看了
        # if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
        #     save_checkpoint(config, epoch, model_without_ddp, 0., optimizer, lr_scheduler, logger)
        # if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
        #     save_checkpoint(config, epoch, model_without_ddp, 0., optimizer, lr_scheduler, logger)
    print('lr_1:', lr_1)
    print('lr_2:', lr_2)
    l1 = plt.plot(lr_1, 'r--', label='type1')
    l2 = plt.plot(lr_2, 'g--', label='type2')
    # plt.plot(lr_1, 'ro-', lr_2, 'g+-')
    plt.legend()
    plt.show()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def show_image(image):
    c = image[0].cpu()
    c = c.permute(1,2,0)
    c_array = c.numpy()
    c_PIL = Image.fromarray(c_array.astype('uint8'))
    c_PIL.save('others/1.jpg')

def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler):
    # model.train()

    optimizer.zero_grad()



    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    # index_count = 0
    for idx, (img, mask, _) in enumerate(data_loader):
        # index_count += 1
        #non-blocking 不会堵塞与其无关的的事情
        # img size 128 192 192
        # mask size 128 48 48
        # 遮盖比率为0.75
        if not on_mac:
            img = img.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
        # 从模型的结果得到一个loss
        loss = model(img, mask)
        # 训练的梯度累计次数
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            # loss 除 梯度累计次数
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            # 这应该是amp那个混合精度损失
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                #     梯度截断？把梯度控制在一个范围内
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            #         更新优化器
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)
        if not on_mac:
            torch.cuda.synchronize()

        loss_meter.update(loss.item(), img.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        lr_1.append(optimizer.state_dict()['param_groups'][0]['lr'])
        lr_2.append(optimizer.state_dict()['param_groups'][1]['lr'])
        a = optimizer.state_dict()['param_groups'][0]['lr']
        b = optimizer.state_dict()['param_groups'][1]['lr']
        logger.info(
            f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
            f'0:{a}\t'
            f'1:{b}\t'
        )
        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    # logger.info(f"INDEX_COUNT {epoch} index_count is {index_count}")
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


if __name__ == '__main__':
    _, config = parse_option()
    on_mac = True
    lr_1, lr_2 = [], []
    # C:/ProgramData/Anaconda3/envs/CGDM/Lib/site-packages/apex/amp/_amp_state.py 修改了调用问题
    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if not config.IS_DIST:
        os.environ['RANK'] = '-1'
        os.environ['world_size'] = '-1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '1080'

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    if on_mac:
        torch.cuda.set_device(config.LOCAL_RANK)
    else:
        torch.cuda.set_device(config.LOCAL_RANK)
    if config.IS_DIST:
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.distributed.barrier()
    if config.IS_DIST:
        seed = config.SEED + dist.get_rank()
    else:
        seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    if config.IS_DIST:
        # linear scale the learning rate according to total batch size, may not be optimal
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    else:
        # linear scale the learning rate according to total batch size, may not be optimal
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    # 先是可变参数，然后变完参数后冻结 上面的没太看懂，去除dist之后查看main
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{'demo_'+config.MODEL.NAME}")
    # 估摸着也就是看看是不是主机
    if 1:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
