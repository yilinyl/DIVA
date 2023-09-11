import os
import sys

# Pipeline for running ProNet on protein structure graph

# sys.path.append('..')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.abspath('..'))

import random

from datetime import datetime
import torch.optim as optim
from data.pronet_dataset import *
from torch_geometric.loader import DataLoader
from models.pronet import ProNet
# from dev.models.graphtransformer import GraphTransformer
from torch.utils.tensorboard import SummaryWriter
from metrics import *
from utils import str2bool, env_setup, _save_scores, format_metadata
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


def train_epoch(model, optimizer, criterion, device, data_loader, args, diagnostic=None):
    model.train()
    running_loss = 0
    n_sample = 0
    for batch_idx, batch_data in enumerate(data_loader):
        # TODO: check batch_data structure
        
        # if args.mask:
        #     # random mask node aatype
        #     mask_indice = torch.tensor(np.random.choice(batch.num_nodes, int(batch.num_nodes * args.mask_aatype), replace=False))
        #     batch.x[:, 0][mask_indice] = 25
        if args.noise:
            # add gaussian noise to atom coords
            gaussian_noise = torch.clip(torch.normal(mean=0.0, std=0.1, size=batch_data.coords.shape), min=-0.3, max=0.3)
            batch_data.coords += gaussian_noise
            # if args.level != 'aminoacid':
            #     batch.coords_n += gaussian_noise
            #     batch.coords_c += gaussian_noise
        if args.deform:
            # Anisotropic scale
            deform = torch.clip(torch.normal(mean=1.0, std=0.1, size=(1, 3)), min=0.9, max=1.1)
            batch_data.coords *= deform
            # if args.level != 'aminoacid':
            #     batch.coords_n *= deform
            #     batch.coords_c *= deform
        batch_data = batch_data.to(device)
        batch_labels = batch_data.y
            
        optimizer.zero_grad()

        batch_logits = model.forward(batch_data)
        shapes = batch_logits.size()
        batch_logits = batch_logits.view(shapes[0]*shapes[1])

        loss = criterion(batch_logits, batch_labels.to(torch.float64))
        loss.backward()
        optimizer.step()
        
        loss_ = loss.detach().item()
        size = batch_labels.size()[0]
        running_loss += loss_* size
        n_sample += size

        if diagnostic and batch_idx == 5:
            diagnostic.print_diagnostics()
            break
    epoch_loss = running_loss / n_sample
    
    return epoch_loss, optimizer


def evaluation_epoch(model, device, criterion, data_loader):
    model.eval()
    running_loss = 0
    n_sample = 0
    all_vars, all_labels, all_scores = [], [], []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            batch_vars = batch_data.id
            batch_data = batch_data.to(device)
            batch_labels = batch_data.y
            batch_logits = model.forward(batch_data)
            shapes = batch_logits.size()
            batch_logits = batch_logits.view(shapes[0] * shapes[1])

            loss = criterion(batch_logits, batch_labels.to(torch.float64))
            loss_ = loss.detach().item()
            size = batch_labels.size()[0]
            running_loss += loss_ * size
            n_sample += size

            batch_scores = torch.sigmoid(batch_logits)  # predict from logits

            batch_scores_ = batch_scores.detach().cpu().numpy()
            batch_labels_ = batch_labels.detach().cpu().numpy()
            
            find_na = np.isnan(batch_scores_)
            if find_na.any():
                logging.critical('NA values in validation score: {} for variants: {}'.format(batch_scores_[find_na], np.array(batch_vars)[find_na]))
                logging.critical('Current running loss: {}'.format(loss_))
            all_labels.append(batch_labels_)
            all_scores.append(batch_scores_)
            all_vars.extend(batch_vars)

        epoch_loss = running_loss / n_sample
        all_labels = np.concatenate(all_labels, 0)
        all_scores = np.concatenate(all_scores, 0)
    
    return epoch_loss, all_labels, all_scores, all_vars


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/config_pronet.json', help="Config file path (.json)")
    parser.add_argument('--gpu_id', type=int, help="GPU device ID")
    parser.add_argument('--tensorboard', type=str2bool, default=False,
                        help='Option to write log information in tensorboard')
    # parser.add_argument('--model_dir', help="Model directory")
    parser.add_argument('--data_dir', help='Data directory')
    parser.add_argument('--exp_dir', help='Directory for all training related files, e.g. checkpoints, log')
    parser.add_argument('--experiment', help='Experiment name')
    parser.add_argument('--log_level', default='info', help='Log level')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--inf-check', type=str2bool, default=False,
                        help='add hooks to check for infinite module outputs and gradients')
    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )
    # data augmentation tricks from ProNet
    # see appendix E in the paper (https://openreview.net/pdf?id=9X-hgLDLYkQ)
    parser.add_argument('--mask', type=str2bool, default=True, help='Random mask some node type')
    parser.add_argument('--noise', type=str2bool, default=True, help='Add Gaussian noise to node coords')
    parser.add_argument('--deform', type=str2bool, default=True, help='Deform node coords')

    parser.add_argument('--train_data', default='train.csv', help='Training data file')
    parser.add_argument('--test_data', default='test.csv', help='Testing data file')
    parser.add_argument('--val_data', default='val.csv', help='Validation data file')
    parser.add_argument('--save_freq', type=int, default=5, help='Frequency to save models')

    parser.add_argument('--lr_decay_step_size', type=int, default=15, help='Learning rate step size')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='Learning rate factor') 
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight Decay')

    args = parser.parse_args()

    return args


def pipeline():
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)

    net_params, data_params = env_setup(args, config)

    if args.tensorboard:
        tb_writer = SummaryWriter(log_dir='{}/tensorboard'.format(config['exp_dir']))
    else:
        tb_writer = None

    if args.data_dir is not None:
        config['data_dir'] = args.data_dir

    data_path = Path(config['data_dir'])

    df_train = pd.read_csv(data_path / args.train_data)
    df_val = pd.read_csv(data_path / args.val_data)
    df_test = pd.read_csv(data_path / args.test_data)
    prot_cols = ['UniProt','PDB', 'Chain']
    var_prot_df = pd.concat([df_train[prot_cols], df_val[prot_cols], df_test[prot_cols]]).drop_duplicates()
    sift_map = pd.read_csv(data_params['sift_file'], sep='\t').dropna().reset_index(drop=True)
    sift_map = sift_map.merge(var_prot_df, how='inner').drop_duplicates().reset_index(drop=True)

    if Path(data_params['seq2struct_cache']).exists():
        with open(data_params['seq2struct_cache'], 'rb') as f_pkl:
            seq_struct_dict = pickle.load(f_pkl)
        data_params['seq2struct_dict'] = seq_struct_dict
    
    device = net_params['device']

    train_dataset = ProNetDataset(df_train, sift_map=sift_map, **data_params)
    val_dataset = ProNetDataset(df_val, sift_map=sift_map, **data_params)
    test_dataset = ProNetDataset(df_test, sift_map=sift_map, **data_params)

    logging.info('Training set: {}; Positive: {}'.format(len(train_dataset), train_dataset.count_positive()))
    logging.info('Training data summary (average) nodes: {:.0f}; edges: {:.0f}'.format(*train_dataset.dataset_summary()))
    logging.info('Test set: {}; Positive: {}'.format(len(test_dataset), test_dataset.count_positive()))
    logging.info('Validation set: {}; Positive: {}'.format(len(val_dataset), val_dataset.count_positive()))


    exp_dir = net_params['exp_dir']
    model_save_path = Path(exp_dir) / 'checkpoints'

    if not model_save_path.exists():
        model_save_path.mkdir(parents=True)

    result_path = Path(exp_dir) / 'result'
    if not result_path.exists():
        result_path.mkdir(parents=True)

    # --------------- Build Model ---------------
    nfeat_key = train_dataset.nfeat_key
    efeat_key = train_dataset.efeat_key
    ndata_dims = train_dataset.get_ndata_dim()
    edata_dims = train_dataset.get_edata_dim()

    net_params['ndata_dim_in'] = ndata_dims
    net_params['out_channels'] = net_params['n_classes']
    model = ProNet(**net_params)  # TODO: add model initialization
    logging.info(f'Model Architecture:\n{model}')
    # total_param = sum([p.numel() for p in model.parameters()])
    logging.info(f'Number of model parameters: {model.num_params}')

    # logging.info('Total Parameters: {}\n\n'.format(view_model_param(net_params, model)))

    model = model.to(device)

    if args.inf_check:
        register_inf_check_hooks(model)

    if args.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )
        diagnostic = diagnostics.attach_diagnostics(model, opts)
    else:
        diagnostic = None

    optimizer = optim.Adam(model.parameters(), lr=net_params['init_lr'], weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor)

    train_loader = DataLoader(train_dataset, batch_size=data_params['batch_size'], shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=data_params['batch_size'], shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=data_params['batch_size'], shuffle=False, num_workers=args.num_workers)

    logging.info("Training starts...")
    # best_ep_scores_train = None
    # best_ep_labels_train = None
    best_val_loss = float('inf')
    best_weights = None
    best_optim = None
    best_epoch = 0
    best_results = {'train': None, 'test': None, 'val': None}

    for epoch in range(net_params['epochs']):
        logging.info('Epoch %d' % epoch)
        # if tb_writer:
        #     tb_writer.add_scalar("train/epoch", epoch)

        train_loss, optimizer = train_epoch(model, optimizer, criterion, device, train_loader, args, diagnostic=diagnostic)
        if epoch % args.save_freq == 0:
            torch.save({'args': net_params, 
                        'state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict()},
                       model_save_path / 'model-ep{}.pt'.format(epoch))
            # torch.save(model.state_dict(), os.path.join(net_params['model_dir'], 'model%s.dat'%(str(epoch))))
        train_loss, train_labels, train_scores, train_vars = evaluation_epoch(model, device, criterion, train_loader)
        train_aupr = compute_aupr(train_labels, train_scores)
        train_auc = compute_roc(train_labels, train_scores)

        data_name = 'train'
        logging.info(f'<{data_name}> loss={train_loss:.4f} auPR={train_aupr:.4f} auROC={train_auc:.4f}')

        val_loss, val_labels, val_scores, val_vars = evaluation_epoch(model, device, criterion, val_loader)

        val_aupr = compute_aupr(val_labels, val_scores)
        val_auc = compute_roc(val_labels, val_scores)

        data_name = 'validation'
        logging.info(f'<{data_name}> loss={val_loss:.4f} auPR={val_aupr:.4f} auROC={val_auc:.4f}')

        test_loss, test_labels, test_scores, test_vars = evaluation_epoch(model, device, criterion, test_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            best_optim = copy.deepcopy(optimizer.state_dict())
            # best_ep_scores_train = train_scores
            # best_ep_labels_train = train_labels
            best_results['train'] = (train_vars, train_labels, train_scores)
            best_results['test'] = (test_vars, test_labels, test_scores)
            best_results['val'] = (val_vars, val_labels, val_scores)
        # print('# Loss: train= {0:.5f}; validation= {1:.5f}; test= {2:.5f};'.format(train_loss, val_loss, test_loss))

        test_aupr = compute_aupr(test_labels, test_scores)
        test_auc = compute_roc(test_labels, test_scores)
        data_name = 'test'
        logging.info(f'<{data_name}> loss={test_loss:.4f} auPR={test_aupr:.4f} auROC={test_auc:.4f}')

        if tb_writer:
            tb_writer.add_pr_curve('Train/PR-curve', train_labels, train_scores, epoch)
            tb_writer.add_pr_curve('Test/PR-curve', test_labels, test_scores, epoch)
            tb_writer.add_pr_curve('Val/PR-curve', val_labels, val_scores, epoch)

            tb_writer.add_scalar('train/loss', train_loss, epoch)
            tb_writer.add_scalar('validation/loss', val_loss, epoch)
            tb_writer.add_scalar('test/loss', test_loss, epoch)

        scheduler.step()

        if optimizer.param_groups[0]['lr'] < net_params['min_lr']:
            logging.info("!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD.")
            break

    logging.info('Save best model at epoch {}:'.format(best_epoch))
    torch.save({'args': net_params, 'state_dict': best_weights,
                'optimizer_state_dict': best_optim},
               model_save_path / 'bestmodel-ep{}.pt'.format(best_epoch))
    for key in best_results:
        _save_scores(best_results[key][0], best_results[key][1], best_results[key][2], key, best_epoch, exp_dir)


if __name__ == '__main__':
    pipeline()
    
    logging.info('Done!')
