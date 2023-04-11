import os
import argparse
import json
import torch
import random
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

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


def gpu_setup(use_gpu, gpu_id):
    if torch.cuda.is_available() and use_gpu:
        device = torch.device('cuda:%s'%str(gpu_id))
    else:
        device = torch.device('cpu')
        logging.info('GPU not available, running on CPU')
    return device


def env_setup(args, config):
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    if args.exp_dir is not None:
        config['exp_dir'] = args.exp_dir
    if args.experiment is not None:
        config['experiment'] = args.experiment
    config['exp_dir'] = '{exp_root}/{name}/{date_time}'.format(exp_root=config['exp_dir'],
                                                               name=config['experiment'],
                                                               date_time=date_time)
    # Set up logging file
    setup_logger(config['exp_dir'], log_prefix=config['mode'], log_level=args.log_level)
    logging.info(json.dumps(config, indent=4))

    if args.data_dir is not None:
        config['data_dir'] = args.data_dir

    data_params = config['data_params']

    net_params = config['net_params']
    net_params['device'] = device
    net_params['use_gpu'] = config['gpu']['use']
    net_params['gpu_id'] = config['gpu']['id']
    net_params['exp_dir'] = config['exp_dir']
    net_params['lap_pos_enc'] = data_params['lap_pos_enc']
    net_params['wl_pos_enc'] = data_params['wl_pos_enc']
    net_params['pos_enc_dim'] = data_params['pos_enc_dim']

    # Graph cache config (for structure based graph)
    if not data_params['cache_only'] and data_params['graph_type'] in ['seq', 'hetero']:
        graph_cache_root = Path(data_params['graph_cache_root'])
        if data_params['method'] == 'radius':
            graph_cache = graph_cache_root / f'radius{data_params["radius"]}'
        else:
            graph_cache = graph_cache_root / f'knn{data_params["num_neighbors"]}'

        data_params['graph_cache'] = os.fspath(graph_cache)

    # setting seeds
    random.seed(net_params['seed'])
    np.random.seed(net_params['seed'])
    torch.manual_seed(net_params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(net_params['seed'])

    return net_params, data_params


def view_model_param(model):
    # model = GraphTransformer(net_params)
    # total_param = 0
    # for param in model.parameters():
    #     total_param += np.prod(list(param.data.size()))
    total_param = sum([p.numel() for p in model.parameters()])

    return total_param


def _save_scores(var_ids, target, pred, name, epoch, exp_dir):
    with open(f'{exp_dir}/result/epoch_{epoch}_{name}_score.txt', 'w') as f:
        f.write('var\ttarget\tscore\n')
        for a, c, d in zip(var_ids, target, pred):
            f.write('{}\t{:d}\t{:f}\n'.format(a, int(c), d))
