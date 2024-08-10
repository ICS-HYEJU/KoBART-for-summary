from core.engine import Engine

import torch

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    from config.config import get_config_dict
    cfg = get_config_dict()
    #
    if cfg.device['gpu_id'] is not None:
        device = torch.device('cuda:{}'.format(cfg.device['gpu_id']))
        torch.cuda.set_device(cfg.device['gpu_id'])
    else:
        device = torch.device('cpu')
    #
    engine = Engine(cfg, mode='fit', device=device)
    engine.start_train()
    checkpoint = torch.load('/home/hjchoi/PycharmProjects/KoBART-for-summary/checkpoint/last.ckpt')
