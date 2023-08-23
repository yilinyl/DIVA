import os
import sys

# sys.path.append('..')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.abspath('..'))

import random

from datetime import datetime
import torch.optim as optim
from data import build_dataset
from data.dataset_multi import *
from models import build_model
# from models.multimodal import MultiModel
# from dev.models.graphtransformer import GraphTransformer
from torch.utils.tensorboard import SummaryWriter
from metrics import *
from utils import str2bool, env_setup, _save_scores
from preprocess.utils import parse_fasta
from hooks import register_inf_check_hooks
import diagnostics
import argparse

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)


def gpu_setup(use_gpu, gpu_id):
    if torch.cuda.is_available() and use_gpu:
        device = torch.device('cuda:%s'%str(gpu_id))
    else:
        device = torch.device('cpu')
        logging.info('GPU not available, running on CPU')
    return device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/config_inference.json', help="Config file path (.json)")
    parser.add_argument('--gpu_id', type=int, help="GPU device ID")
    parser.add_argument('--model_path', type=str, help="Path to model")
    
    # parser.add_argument('--model_dir', help="Model directory")
    parser.add_argument('--data_dir', help='Data directory')
    parser.add_argument('--exp_dir', help='Directory for all training related files, e.g. checkpoints, log')
    parser.add_argument('--experiment', help='Experiment name')
    parser.add_argument('--log_level', default='info', help='Log level')
    parser.add_argument('--tensorboard', type=str2bool, default=False,
                        help='Option to write log information in tensorboard')
    parser.add_argument('--test_data', default='test.csv', help='Testing data file')
    parser.add_argument('--save_freq', type=int, default=5, help='Frequency to save models')
    args = parser.parse_args()

    return args


def predict(model, device, data_loader, dtype='multi'):
    model.eval()
    all_vars, all_labels, all_scores = [], [], []
    all_emb = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            batch_labels = batch_data[1]  # binary label or experimental score
            batch_alt_aa = batch_data[2].to(device)
            batch_vars = batch_data[-1]
            if dtype == 'multi':
                batch_dict = batch_data[0]
                batch_seq_graph = batch_dict['seq_graph'].to(device)
                batch_str_graph = batch_dict['struct_graph'].to(device)
                batch_logits, batch_emb = model(batch_seq_graph, batch_str_graph, batch_alt_aa)
            else:
                batch_graphs = batch_data[0].to(device)
                batch_logits = model.forward(batch_graphs, batch_alt_aa)

            shapes = batch_logits.size()
            batch_logits = batch_logits.view(shapes[0] * shapes[1])

            batch_scores = model.predict(batch_logits)
            batch_labels_ = batch_labels.detach().cpu().numpy()
            batch_scores_ = batch_scores.detach().cpu().numpy()
            # batch_embeds_ = batch_emb.detach().cpu()
            all_scores.append(batch_scores_)
            all_labels.append(batch_labels_)
            all_vars.extend(batch_vars)
            # all_emb.append(batch_embeds_)

        all_labels = np.concatenate(all_labels, 0)
        all_scores = np.concatenate(all_scores, 0)
    
    return all_scores, all_labels, all_vars


if __name__ == '__main__':
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)
        
    update_dict, data_params = env_setup(args, config)
    device = update_dict['device']
    # key_update = ['device', 'exp_dir']
    # update_dict = {key: net_params_cur[key] for key in key_update}
    if args.model_path is not None:
        config['model_path'] = args.model_path

    data_path = Path(config['data_dir'])
    df_test = pd.read_csv(data_path / args.test_data)
    if not config['out_prefix']:
        name_prefix = args.test_data.split('.')[0]
    else:
        name_prefix = config['out_prefix']
    sift_map = None
    if config['data_type'] != 'seq':
        sift_map = pd.read_csv(data_params['sift_file'], sep='\t').dropna().reset_index(drop=True)
        sift_map = sift_map.merge(df_test, how='inner').drop_duplicates().reset_index(drop=True)

        if Path(data_params['seq2struct_cache']).exists():
            with open(data_params['seq2struct_cache'], 'rb') as f_pkl:
                seq_struct_dict = pickle.load(f_pkl)
            data_params['seq2struct_dict'] = seq_struct_dict
    
    seq_dict = dict()
    for fname in data_params['seq_fasta']:
        try:
            seq_dict.update(parse_fasta(fname))
        except FileNotFoundError:
            pass
    data_params['seq_dict'] = seq_dict
    if config['use_esm']:
        tokenizer, lm_model = init_pretrained_lm(config['pretrained_esm'])
        lm_model = lm_model.to(device)
        test_dataset = MultiModalLMDataset(df_test, tokenizer, lm_model, device, **data_params)
    else:
        test_dataset = build_dataset(config['data_type'], df_test, sift_map=sift_map, **data_params)
        # test_dataset = MultiModalDataSet(df_test, sift_map=sift_map, **data_params)  # TODO: var_db
    if config['model_name'] == 'multi':
        logging.info("Sequential graph summary: (average) nodes {:.1f}, edges {:.1f}".format(*test_dataset.dataset_summary(test_dataset.seq_graph_stats)))
        logging.info("Structural graph summary: (average) nodes {:.1f}, edges {:.1f}".format(*test_dataset.dataset_summary(test_dataset.struct_graph_stats)))
    else:
        logging.info("Graph summary for {}: (average) nodes {:.1f}, edges {:.1f}".format(config['data_type'], *test_dataset.dataset_summary()))
    # Load model
    checkpt_dict = torch.load(config['model_path'], map_location='cpu')
    net_params = checkpt_dict['args']
    net_params.update(update_dict)
    if config['model_name'] == 'multi':
        net_params['seq_params'].update(update_dict)
        net_params['struct_params'].update(update_dict)
    state_dict = checkpt_dict['state_dict']
    
    logging.info('Loading model weights...')
    model = build_model(config['model_name'], **net_params)
    model.load_state_dict(state_dict)
    logging.info(f'Model Architecture:\n{model}')
    model = model.to(device)

    test_loader = dgl.dataloading.GraphDataLoader(test_dataset, batch_size=net_params['batch_size'],
                                                  shuffle=False, pin_memory=True, drop_last=False)
    exp_dir = net_params['exp_dir']
    result_path = Path(exp_dir)
    if not result_path.exists():
        result_path.mkdir(parents=True)

    if args.tensorboard:
        tb_writer = SummaryWriter(log_dir='{}/tensorboard'.format(config['exp_dir']))
    else:
        tb_writer = None

    logging.info("Inference on test set...")
    test_scores, test_labels, test_vars = predict(model, device, test_loader, dtype=config['data_type'])
    _save_scores(test_vars, test_labels, test_scores, name_prefix, exp_dir=exp_dir, mode='pred')
