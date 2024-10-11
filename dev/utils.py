import os
import argparse
import json
import yaml
import torch
import random
import numpy as np
import dgl
from datetime import datetime
from pathlib import Path
import logging
import psutil

def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter
        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False
    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def setup_logger(exp_dir, log_prefix, log_level='info', use_console=True):
    """
    Log setup
    Args:
        exp_dir:
        log_prefix:
        log_level: "debug", "info", "warning", "error", "critical"
        use_console: if True, will also print logs to console
    """
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d")
    log_path = Path(exp_dir) / 'log'
    # if not log_path.exists():
    log_path.mkdir(parents=True, exist_ok=True)

    # formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    formatter = '%(asctime)s - %(levelname)s: %(message)s'
    log_filename = os.fspath(log_path / f"{log_prefix}-{date_time}.log")

    level = logging.ERROR
    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "warning":
        level = logging.WARNING
    elif log_level == "critical":
        level = logging.CRITICAL

    logging.basicConfig(
        filename=log_filename,
        format=formatter,
        level=level,
        filemode="w",
    )

    if use_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(formatter, datefmt='%Y-%m-%d %H:%M:%S'))
        logging.getLogger().addHandler(console)



def set_seed(seed: int, device=None):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if device is not None and device.type == 'cuda':
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available


def load_input_to_device(input_data, device, exclude_keys=None):
    if not exclude_keys:
        exclude_keys = []
    if isinstance(input_data, dict):
        for k, v in input_data.items():
            if k not in exclude_keys and isinstance(v, (torch.Tensor, dgl.DGLGraph)):
                input_data[k] = v.to(device)
    
    return input_data


def load_config(cfg_file, format='yaml'):
    if format.lower() == 'yaml':
        with open(cfg_file, 'r') as f:
            config = yaml.safe_load(f)
    elif format.lower() == 'json':
        with open(cfg_file, 'r') as f:
            config = json.load(cfg_file)
    else:
        raise ValueError(f'{format} not supported! Please use one of [JSON, YAML]')
    
    return config


def gpu_setup(device='cpu'):
    """
    Setup GPU device
    """
    
    if torch.cuda.is_available() and device != 'cpu':
        device = torch.device(device)
    else:
        device = torch.device('cpu')
        logging.info('GPU not available, running on CPU')

    return device


def env_setup(args, config, use_timestamp=True):
    
    device = gpu_setup(config['device'])
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    if args.exp_dir is not None:
        config['exp_dir'] = args.exp_dir
    if args.experiment is not None:
        config['experiment'] = args.experiment
    if use_timestamp:
        config['exp_dir'] = '{exp_root}/{name}/{date_time}'.format(exp_root=config['exp_dir'],
                                                                name=config['experiment'],
                                                                date_time=date_time)
    else:
        config['exp_dir'] = '{exp_root}/{name}'.format(exp_root=config['exp_dir'],
                                                                name=config['experiment'])
    # Set up logging file
    setup_logger(config['exp_dir'], log_prefix=config['mode'], log_level=args.log_level)
    logging.info(json.dumps(config, indent=4))
    
    set_seed(args.seed, device)

    return config, device

def view_model_param(model):
    # model = GraphTransformer(net_params)
    # total_param = 0
    # for param in model.parameters():
    #     total_param += np.prod(list(param.data.size()))
    total_param = sum([p.numel() for p in model.parameters()])

    return total_param


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def _save_scores(var_ids, target, pred, name, epoch='', exp_dir='./output', mode='train'):
    if mode == 'train':
        fname = f'{exp_dir}/result/epoch_{epoch}_{name}_score.txt'
    else:
        fname = f'{exp_dir}/pred_{name}_score.txt'
    with open(fname, 'w') as f:
        f.write('var\ttarget\tscore\n')
        for a, c, d in zip(var_ids, target, pred):
            f.write('{}\t{}\t{:f}\n'.format(a, c, d))


def format_metadata(var_ids, labels):
    """
    Format metadata information for tensorboard projector
    """
    meta_info = []
    for i in range(len(var_ids)):
        prot_ids, pos, amino_acids = var_ids[i].split('_')
        ref_aa, alt_aa = amino_acids.split('/')
        meta_info.append((prot_ids, amino_acids, ref_aa, alt_aa, labels[i]))
    
    return meta_info