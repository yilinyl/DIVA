import os
import sys

sys.path.append(os.path.abspath('..'))
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import random
import gzip
import logging

from datetime import datetime
import torch.optim as optim
from data.datasets import *
from models import build_model
from torch.utils.tensorboard import SummaryWriter
from metrics import *
from utils import str2bool, env_setup, _save_scores
from preprocess.utils import parse_fasta
from hooks import register_inf_check_hooks
import argparse

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)


def train_epoch(model, optimizer, device, data_loader):
    model.train()
    running_loss = 0
    n_sample = 0
    for batch_idx, batch_data in enumerate(data_loader):
        # TODO: check batch_data structure
        batch_graphs = batch_data[0].to(device)
        batch_labels = batch_data[1].to(device)
        batch_alt_aa = batch_data[2].to(device)
        batch_n_nodes = torch.cumsum(batch_data[0].batch_num_nodes(), 0)
        batch_n_nodes = torch.cat((torch.tensor([0]), batch_n_nodes), 0)
        batch_var_idx = batch_data[3] + batch_n_nodes[:-1]
        batch_var_idx = batch_var_idx.to(device)

        optimizer.zero_grad()

        batch_logits = model.forward(batch_graphs, batch_alt_aa, batch_var_idx)
        shapes = batch_logits.size()
        batch_logits = batch_logits.view(shapes[0]*shapes[1])

        loss = model.loss(batch_logits, batch_labels)
        loss.backward()
        optimizer.step()
        
        loss_ = loss.detach().item()
        size = batch_labels.size()[0]
        running_loss += loss_* size
        n_sample += size

    epoch_loss = running_loss / n_sample
    
    return epoch_loss, optimizer


def evaluation_epoch(model, device, data_loader):
    model.eval()
    running_loss = 0
    n_sample = 0
    all_vars, all_labels, all_scores = [], [], []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):

            batch_graphs = batch_data[0].to(device)
            batch_labels = batch_data[1].to(device)
            batch_alt_aa = batch_data[2].to(device)
            batch_n_nodes = torch.cumsum(batch_data[0].batch_num_nodes(), 0)
            batch_n_nodes = torch.cat((torch.tensor([0]), batch_n_nodes), 0)
            batch_var_idx = batch_data[3] + batch_n_nodes[:-1]
            batch_var_idx = batch_var_idx.to(device)
            batch_vars = batch_data[-1]  # TODO: add var_id

            batch_logits = model.forward(batch_graphs, batch_alt_aa, batch_var_idx)
            shapes = batch_logits.size()
            batch_logits = batch_logits.view(shapes[0] * shapes[1])

            loss = model.loss(batch_logits, batch_labels)
            loss_ = loss.detach().item()
            size = batch_labels.size()[0]
            running_loss += loss_ * size
            n_sample += size

            batch_scores = model.predict(batch_logits)

            # if device.type == 'cuda':
            batch_scores_ = batch_scores.detach().cpu().numpy()
            batch_labels_ = batch_labels.detach().cpu().numpy()
            
            all_labels.append(batch_labels_)
            all_scores.append(batch_scores_)
            all_vars.extend(batch_vars)

        epoch_loss = running_loss / n_sample
        all_labels = np.concatenate(all_labels, 0)
        all_scores = np.concatenate(all_scores, 0)
    
    return epoch_loss, all_labels, all_scores, all_vars


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/config_hetero.json', help="Config file path (.json)")
    parser.add_argument('--gpu_id', type=int, help="GPU device ID")
    parser.add_argument('--tensorboard', type=str2bool, default=False,
                        help='Option to write log information in tensorboard')
    # parser.add_argument('--model_dir', help="Model directory")
    parser.add_argument('--data_dir', help='Data directory')
    parser.add_argument('--exp_dir', help='Directory for all training related files, e.g. checkpoints, log')
    parser.add_argument('--experiment', help='Experiment name')
    parser.add_argument('--log_level', default='info', help='Log level')

    parser.add_argument('--inf-check', type=str2bool, default=False,
                        help='add hooks to check for infinite module outputs and gradients')
    # parser.add_argument('--var_info')
    parser.add_argument('--train_data', default='train.csv', help='Training data file')
    parser.add_argument('--test_data', default='test.csv', help='Testing data file')
    parser.add_argument('--val_data', default='val.csv', help='Validation data file')
    parser.add_argument('--save_freq', type=int, default=5, help='Frequency to save models')
    args = parser.parse_args()

    return args


def pipeline():
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)

    net_params, data_params = env_setup(args, config)

    if args.tensorboard:
        tb_writer = SummaryWriter(log_dir='{}/logs'.format(config['exp_dir']))
    else:
        tb_writer = None

    # --------------- Load data ---------------
    data_path = Path(config['data_dir'])

    df_train = pd.read_csv(data_path / args.train_data)
    df_val = pd.read_csv(data_path / args.val_data)
    df_test = pd.read_csv(data_path / args.test_data)
    prot_cols = ['UniProt','PDB', 'Chain']
    var_prot_df = pd.concat([df_train[prot_cols], df_val[prot_cols], df_test[prot_cols]]).drop_duplicates()
    sift_map = pd.read_csv(data_params['sift_file'], sep='\t').dropna().reset_index(drop=True)
    sift_map = sift_map.merge(var_prot_df, how='inner').drop_duplicates().reset_index(drop=True)

    if 'feat_stats_file' in data_params and Path(data_params['feat_stats_file']).exists():
        with open(data_params['feat_stats_file'], 'rb') as f_pkl:
            feat_stats, feat_cols = pickle.load(f_pkl)
            for key in feat_stats:
                feat_stats[key] = torch.tensor(feat_stats[key])
    else:
        feat_stats = None

    if Path(data_params['seq2struct_cache']).exists():
        with open(data_params['seq2struct_cache'], 'rb') as f_pkl:
            seq_struct_dict = pickle.load(f_pkl)
    else:
        seq_struct_dict = dict()

    seq_dict = dict()
    for fname in data_params['seq_fasta']:
        try:
            seq_dict.update(parse_fasta(fname))
        except FileNotFoundError:
            pass
    data_params['seq_dict'] = seq_dict
    if data_params['cache_only']:
        train_dataset = VariantGraphCacheDataSet(df_train, feat_stats=feat_stats,
                                                 seq2struct_all=seq_struct_dict, **data_params)
    else:
        train_dataset = VariantGraphDataSet(df_train, sift_map=sift_map, feat_stats=feat_stats,
                                            seq2struct_all=seq_struct_dict, **data_params)

    seq_struct_dict.update(train_dataset.seq2struct_dict)
    var_ref = train_dataset.get_var_db()
    logging.info('Training data summary (average) nodes: {:.0f}; edges: {:.0f}'.format(*train_dataset.dataset_summary()))

    if data_params['cache_only']:
        validation_dataset = VariantGraphCacheDataSet(df_val, feat_stats=feat_stats,
                                                      seq2struct_all=seq_struct_dict, var_db=var_ref, **data_params)
    else:
        validation_dataset = VariantGraphDataSet(df_val, sift_map=sift_map, feat_stats=feat_stats,
                                                 seq2struct_all=seq_struct_dict, var_db=var_ref, **data_params)

    seq_struct_dict.update(validation_dataset.seq2struct_dict)
    var_ref = pd.concat([var_ref, validation_dataset.get_var_db()])

    if data_params['cache_only']:
        test_dataset = VariantGraphCacheDataSet(df_test, feat_stats=feat_stats,
                                                seq2struct_all=seq_struct_dict, var_db=var_ref, **data_params)
    else:
        test_dataset = VariantGraphDataSet(df_test, sift_map=sift_map, feat_stats=feat_stats,
                                           seq2struct_all=seq_struct_dict, var_db=var_ref, **data_params)
    seq_struct_dict.update(test_dataset.seq2struct_dict)

    # with open(data_params['seq2struct_cache'], 'wb') as f_pkl:
    #     pickle.dump(seq_struct_dict, f_pkl)

    logging.info('Training set: {}; Positive: {}'.format(len(train_dataset), train_dataset.count_positive()))
    logging.info('Training data summary (average) nodes: {:.0f}; edges: {:.0f}'.format(*train_dataset.dataset_summary()))
    # logging.info('Average number of pathogenic variants in graph: {:.1f}'.format(train_dataset.get_patho_num()))
    logging.info('Test set: {}; Positive: {}'.format(len(test_dataset), test_dataset.count_positive()))
    logging.info('Validation set: {}; Positive: {}'.format(len(validation_dataset), validation_dataset.count_positive()))

    device = net_params['device']
    exp_dir = net_params['exp_dir']
    model_save_path = Path(exp_dir) / 'checkpoints'

    if not model_save_path.exists():
        model_save_path.mkdir(parents=True)

    result_path = Path(exp_dir) / 'result'
    if not result_path.exists():
        result_path.mkdir(parents=True)

    # --------------- Build Model ---------------
    net_params['ndata_dim_in'] = train_dataset.get_ndata_dim('feat')
    net_params['meta_paths'] = ['seq', 'struct']
    model = build_model(config['model_name'], **net_params)
    logging.info(f'Model Architecture:\n{model}')
    total_param = sum([p.numel() for p in model.parameters()])
    logging.info(f'Number of model parameters: {total_param}')

    model = model.to(device)

    if args.inf_check:
        register_inf_check_hooks(model)

    optimizer = optim.Adam(model.parameters(), lr=net_params['init_lr'], weight_decay=net_params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=net_params['lr_reduce_factor'],
                                                     patience=net_params['lr_schedule_patience'],
                                                     verbose=True)

    train_loader = dgl.dataloading.GraphDataLoader(train_dataset, batch_size=net_params['batch_size'],
                                                   shuffle=False, pin_memory=True, drop_last=False)
    validation_loader = dgl.dataloading.GraphDataLoader(validation_dataset, batch_size=net_params['batch_size'],
                                                        shuffle=False, pin_memory=True, drop_last=False)
    test_loader = dgl.dataloading.GraphDataLoader(test_dataset, batch_size=net_params['batch_size'],
                                                  shuffle=False, pin_memory=True, drop_last=False)

    logging.info("Training starts...")
    best_ep_scores_train = None
    best_ep_labels_train = None
    best_val_loss = float('inf')
    best_weights = None
    best_epoch = 0

    for epoch in range(net_params['epochs']):
        logging.info('Epoch %d' % epoch)
        # if tb_writer:
        #     tb_writer.add_scalar("train/epoch", epoch)

        train_loss, optimizer = train_epoch(model, optimizer, device, train_loader)
        if epoch % args.save_freq == 0:
            torch.save({'args': net_params, 'state_dict': model.state_dict()},
                       model_save_path / 'model-ep{}.pt'.format(epoch))
            # torch.save(model.state_dict(), os.path.join(net_params['model_dir'], 'model%s.dat'%(str(epoch))))
        train_loss, train_labels, train_scores, train_vars = evaluation_epoch(model, device, train_loader)
        train_aupr = compute_aupr(train_labels, train_scores)
        train_auc = compute_roc(train_labels, train_scores)

        data_name = 'train'
        logging.info(f'<{data_name}> loss={train_loss:.4f} auPR={train_aupr:.4f} auROC={train_auc:.4f}')

        val_loss, val_labels, val_scores, val_vars = evaluation_epoch(model, device, validation_loader)
        scheduler.step(val_loss)

        val_aupr = compute_aupr(val_labels, val_scores)
        val_auc = compute_roc(val_labels, val_scores)

        data_name = 'validation'
        logging.info(f'<{data_name}> loss={val_loss:.4f} auPR={val_aupr:.4f} auROC={val_auc:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            best_ep_scores_train = train_scores
            best_ep_labels_train = train_labels

        test_loss, test_labels, test_scores, test_vars = evaluation_epoch(model, device, test_loader)
        # print('# Loss: train= {0:.5f}; validation= {1:.5f}; test= {2:.5f};'.format(train_loss, val_loss, test_loss))

        test_aupr = compute_aupr(test_labels, test_scores)
        test_auc = compute_roc(test_labels, test_scores)
        data_name = 'test'
        logging.info(f'<{data_name}> loss={test_loss:.4f} auPR={test_aupr:.4f} auROC={test_auc:.4f}')

        if epoch % args.save_freq == 0:
            _save_scores(train_vars, train_labels, train_scores, 'train', epoch, exp_dir)
            _save_scores(val_vars, val_labels, val_scores, 'val', epoch, exp_dir)
            _save_scores(test_vars, test_labels, test_scores, 'test', epoch, exp_dir)

        if tb_writer:
            tb_writer.add_scalar('train/loss', train_loss, epoch)
            tb_writer.add_scalar('validation/loss', val_loss, epoch)
            tb_writer.add_scalar('test/loss', train_loss, epoch)

        if optimizer.param_groups[0]['lr'] < net_params['min_lr']:
            logging.info("!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD.")
            break

    if tb_writer:
        tb_writer.add_pr_curve('Train/PR-curve', best_ep_labels_train, best_ep_scores_train, best_epoch)
        tb_writer.close()
    logging.info('Save best model at epoch {}:'.format(best_epoch))
    torch.save({'args': net_params, 'state_dict': best_weights,
                'train_labels': best_ep_labels_train, 'train_scores': best_ep_scores_train},
               model_save_path / 'bestmodel-ep{}.pt'.format(best_epoch))


if __name__ == '__main__':
    pipeline()
    
    logging.info('Done!')
