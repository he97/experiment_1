from .data_simmim import build_loader_simmim
from .data_finetune import build_loader_finetune, build_loader_finetune_for_hsi


def build_loader(config, logger, is_pretrain):
    if is_pretrain:
        return build_loader_simmim(config, logger)
    else:
        if config.IS_HSI:
            return build_loader_finetune_for_hsi(config, logger)
        return build_loader_finetune(config, logger)