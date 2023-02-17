import os
import sys

# sys.path.append('..')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import random
import logging

from datetime import datetime
import torch.optim as optim
from data.datasets import *
from graphtransformer import GraphTransformer
from torch.utils.tensorboard import SummaryWriter
from metrics import *
from utils import str2bool, setup_logger
from hooks import register_inf_check_hooks
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


def train_epoch(model, optimizer, device, data_loader):
    model.train()
    running_loss = 0
    n_sample = 0
    for batch_idx, batch_data in enumerate(data_loader):
        # TODO: check batch_data structure
        batch_graphs = batch_data[0].to(device)
        batch_labels = batch_data[1].to(device)
        batch_alt_aa = batch_data[2].to(device)
        # batch_aa_indice = batch_data[3].to(device)
        # batch_aa_mask = batch_data[4].to(device)

        if model.lap_pos_enc:
            # sign flip as in Bresson et al. for laplacian PE
            batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        else:
            batch_lap_pos_enc = None

        # try:
        #     batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
        # except:
        #     batch_wl_pos_enc = None
            
        optimizer.zero_grad()

        batch_logits = model.forward(batch_graphs, batch_lap_pos_enc, batch_alt_aa)
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
            batch_vars = batch_data[3]  # TODO: add var_id

            if model.lap_pos_enc:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            else:
                batch_lap_pos_enc = None

            batch_logits = model.forward(batch_graphs, batch_lap_pos_enc, batch_alt_aa)
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
            # batch_vars_ = batch_vars.detach().cpu().numpy()
            # else:
            #     batch_scores_ = batch_scores.detach().numpy()
            #     batch_labels_ = batch_labels.detach().numpy()
            #     batch_vars_ = batch_vars.detach().numpy()
            
            all_labels.append(batch_labels_)
            all_scores.append(batch_scores_)
            all_vars.extend(batch_vars)

        epoch_loss = running_loss / n_sample
        all_labels = np.concatenate(all_labels, 0)
        all_scores = np.concatenate(all_scores, 0)
    
    return epoch_loss, all_labels, all_scores, all_vars


def run_pipeline(net_params, train_dataset, validation_dataset, test_dataset, save_freq,
                 inf_check=False, tb_writer=None):
    device = net_params['device']
    exp_dir = net_params['exp_dir']
    model_save_path = Path(exp_dir) / 'checkpoints'

    if not model_save_path.exists():
        model_save_path.mkdir(parents=True)

    result_path = Path(exp_dir) / 'result'
    if not result_path.exists():
        result_path.mkdir(parents=True)

    # print('-------------------------------------')

    # print('-------------------------------------')
    # setting seeds
    random.seed(net_params['seed'])
    np.random.seed(net_params['seed'])
    torch.manual_seed(net_params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(net_params['seed'])
        
    model = GraphTransformer(net_params)
    # logging.info('Total Parameters: {}\n\n'.format(view_model_param(net_params, model)))
    total_param = sum([p.numel() for p in model.parameters()])
    logging.info(f'Number of model parameters: {total_param}')

    model = model.to(device)

    if inf_check:
        register_inf_check_hooks(model)

    optimizer = optim.Adam(model.parameters(), lr=net_params['init_lr'], weight_decay=net_params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=net_params['lr_reduce_factor'],
                                                     patience=net_params['lr_schedule_patience'],
                                                     verbose=True)
    
    train_loader = dgl.dataloading.GraphDataLoader(train_dataset, batch_size=net_params['batch_size'],
                                                   shuffle=True, pin_memory=True, drop_last=False)
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
        if epoch % save_freq == 0:
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

        # f_max, p_max, r_max, t_max, predictions_max = compute_performance_max(validation_labels, validation_scores)
        # if predictions_max is not None:
        #     acc = acc_score(validation_labels, predictions_max)
        #     mcc = compute_mcc(validation_labels, predictions_max)
        # else:
        #     acc, mcc = 0.0, 0.0
        # if epoch % print_freq == 0:
        #     print('validation, loss:%0.6f, aupr:%0.6f, auc:%0.6f, F_value:%0.6f, mcc:%0.6f, '
        #           'precision:%0.6f, recall:%0.6f, acc:%0.6f, threshold:%0.6f' % (validation_loss, aupr, auc, f_max,
        #                                                                          mcc, p_max, r_max, acc, t_max))

        test_loss, test_labels, test_scores, test_vars = evaluation_epoch(model, device, test_loader)
        # print('# Loss: train= {0:.5f}; validation= {1:.5f}; test= {2:.5f};'.format(train_loss, val_loss, test_loss))

        test_aupr = compute_aupr(test_labels, test_scores)
        test_auc = compute_roc(test_labels, test_scores)
        data_name = 'test'
        logging.info(f'<{data_name}> loss={test_loss:.4f} auPR={test_aupr:.4f} auROC={test_auc:.4f}')

        # thres = 0.6
        # f, p, r, predictions = compute_performance(test_labels, test_scores, thres)
        # if predictions is not None:
        #     acc = acc_score(test_labels, predictions)
        #     mcc = compute_mcc(test_labels, predictions)
        # else:
        #     acc, mcc = 0, 0
        # print('test, loss:%0.6f, aupr:%0.6f, auc:%0.6f, F_value:%0.6f, mcc:%0.6f, '
        #       'precision:%0.6f, recall:%0.6f, acc:%0.6f, threshold:%0.6f' % (test_loss, aupr, auc, f, mcc, p, r, acc, thres))
        if epoch % save_freq == 0:
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.json', help="Config file path (.json)")
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
    parser.add_argument('--save_freq', type=int, default=1, help='Frequency to save models')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)
        
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])

    if args.exp_dir is not None:
        config['exp_dir'] = args.exp_dir
    if args.experiment is not None:
        config['experiment'] = args.experiment
    config['exp_dir'] = '{exp_root}/{name}'.format(exp_root=config['exp_dir'], name=config['experiment'])
    # Set up logging file
    setup_logger(config['exp_dir'], log_prefix=config['mode'], log_level=args.log_level)
    logging.info(json.dumps(config, indent=4))

    if args.tensorboard:
        tb_writer = SummaryWriter(log_dir='{}/logs'.format(config['exp_dir']))
    else:
        tb_writer = None

    if args.data_dir is not None:
        config['data_dir'] = args.data_dir

    data_path = Path(config['data_dir'])
    data_params = config['data_params']

    net_params = config['net_params']
    net_params['device'] = device
    net_params['use_gpu'] = config['gpu']['use']
    net_params['gpu_id'] = config['gpu']['id']
    net_params['exp_dir'] = config['exp_dir']
    net_params['lap_pos_enc']  = data_params['lap_pos_enc']
    net_params['wl_pos_enc']  = data_params['wl_pos_enc']
    net_params['pos_enc_dim']  = data_params['pos_enc_dim']

    df_train = pd.read_csv(data_path / args.train_data)
    df_val = pd.read_csv(data_path / args.val_data)
    df_test = pd.read_csv(data_path / args.test_data)
    prot_cols = ['UniProt','PDB', 'Chain']
    var_prot_df = pd.concat([df_train[prot_cols], df_val[prot_cols], df_test[prot_cols]]).drop_duplicates()
    sift_map = pd.read_csv(data_params['sift_file'], sep='\t').dropna().reset_index(drop=True)
    sift_map = sift_map.merge(var_prot_df, how='inner').drop_duplicates().reset_index(drop=True)

    with open(data_params['feat_stats_file'], 'rb') as f_pkl:
        feat_stats, feat_cols = pickle.load(f_pkl)
        for key in feat_stats:
            feat_stats[key] = torch.tensor(feat_stats[key])

    if Path(data_params['seq2struct_cache']).exists():
        with open(data_params['seq2struct_cache'], 'rb') as f_pkl:
            seq_struct_dict = pickle.load(f_pkl)
    else:
        seq_struct_dict = dict()

    # Graph cache config
    graph_cache_root = Path(data_params['graph_cache_root'])
    if data_params['method'] == 'radius':
        graph_cache = graph_cache_root / f'radius{data_params["radius"]}'
    else:
        graph_cache = graph_cache_root / f'knn{data_params["num_neighbors"]}'

    data_params['graph_cache'] = os.fspath(graph_cache)

    train_dataset = VariantGraphDataSet(df_train, sift_map=sift_map, feat_stats=feat_stats,
                                        seq2struct_all=seq_struct_dict, **data_params)
    seq_struct_dict.update(train_dataset.seq2struct_dict)
    var_ref = train_dataset.get_var_db()
    logging.info('Training data summary (average) nodes: {:.0f}; edges: {:.0f}'.format(*train_dataset.dataset_summary()))

    validation_dataset = VariantGraphDataSet(df_val, sift_map=sift_map, feat_stats=feat_stats,
                                             seq2struct_all=seq_struct_dict, var_db=var_ref, **data_params)
    seq_struct_dict.update(validation_dataset.seq2struct_dict)
    var_ref = pd.concat([var_ref, validation_dataset.get_var_db()])

    test_dataset = VariantGraphDataSet(df_test, sift_map=sift_map, feat_stats=feat_stats,
                                       seq2struct_all=seq_struct_dict, var_db=var_ref, **data_params)
    seq_struct_dict.update(test_dataset.seq2struct_dict)

    with open(data_params['seq2struct_cache'], 'wb') as f_pkl:
        pickle.dump(seq_struct_dict, f_pkl)

    logging.info('Training set: {}'.format(len(train_dataset)))
    logging.info('Test set: {}'.format(len(test_dataset)))
    logging.info('Validation set: {}'.format(len(validation_dataset)))

    run_pipeline(net_params, train_dataset, validation_dataset, test_dataset, save_freq=args.save_freq,
                 inf_check=args.inf_check, tb_writer=tb_writer)


if __name__ == '__main__':
    main()
    
    logging.info('Done!')
