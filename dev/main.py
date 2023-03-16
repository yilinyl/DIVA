import os, pickle, json, glob
import random

import torch, dgl
import torch.optim as optim
import data_generator
from dev.models.graphtransformer import GraphTransformer
from metrics import *
from arguments import parse_args


def gpu_setup(use_gpu, gpu_id):
    if torch.cuda.is_available() and use_gpu:
        device = torch.device('cuda:%s'%str(gpu_id))
    else:
        device = torch.device('cpu')
        print('GPU not available, running on CPU')
    return device

def view_model_param(net_params):
    model = GraphTransformer(net_params)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    return total_param
        
def train_epoch(model, optimizer, device, data_loader):
    model.train()
    epoch_loss = 0
    for batch_idx, batch_data in enumerate(data_loader):
        # TODO: check batch_data structure
        batch_graphs = batch_data[0][0][0].to(device)
        batch_n_nodes = torch.cumsum(batch_data[1], 0)
        batch_n_nodes = torch.cat((torch.tensor([0]), batch_n_nodes), 0)
        batch_node_ids = batch_data[2] + batch_n_nodes[:-1]
        # batch_x = batch_graphs.ndata['feat'].float().to(device)
        # batch_e = batch_graphs.edata['feat'].float().to(device)
        batch_labels = batch_data[0][0][1].squeeze(1).to(device)
        
        # batch_partners = batch_data[0][1].squeeze(1).to(device)
        
        # batch_n_nodes = batch_graphs.num_nodes().detach().cpu()
        # for i in range(1, batch_n_nodes.size()[0]):
        #     batch_n_nodes[i] += batch_n_nodes[i-1]
        # batch_n_nodes = torch.cat((torch.tensor([0]), batch_n_nodes), 0)
    
        # batch_node_ids = batch_data[2]
        # for i in range(batch_node_ids.size()[0]):
        #     batch_node_ids[i] += batch_n_nodes[i]

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
        
        batch_scores = model.forward(batch_graphs, batch_lap_pos_enc,
                                     batch_n_nodes, batch_node_ids)
        shapes = batch_scores.size()
        batch_scores = batch_scores.view(shapes[0]*shapes[1])

        loss = model.loss(batch_scores, batch_labels.float())
        loss.backward()
        optimizer.step()
        
        loss_ = loss.detach().item()
        epoch_loss += loss_/batch_labels.size()[0]
        
    epoch_loss /= (batch_idx + 1)
    
    return epoch_loss, optimizer


def evaluation_epoch(model, device, data_loader):
    model.eval()
    epoch_loss = 0
    all_labels, all_scores = [], []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):

            batch_graphs = batch_data[0][0][0].to(device)
            # batch_x = batch_graphs.ndata['feat'].float().to(device)
            # batch_e = batch_graphs.edata['feat'].float().to(device)
            batch_labels = batch_data[0][0][1].squeeze(1).to(device)
            batch_n_nodes = torch.cat((torch.tensor([0]), torch.cumsum(batch_data[1], 0)), 0)
            batch_node_ids = batch_data[2] + batch_n_nodes[:-1]
            # batch_partners = batch_data[0][1].squeeze(1).to(device)

            # batch_n_nodes = batch_data[1]
            # for i in range(1, batch_n_nodes.size()[0]):
            #     batch_n_nodes[i] += batch_n_nodes[i-1]
            # batch_n_nodes = torch.cat((torch.tensor([0]), batch_n_nodes), 0)
            #
            # batch_node_ids = batch_data[2]
            # for i in range(batch_node_ids.size()[0]):
            #     batch_node_ids[i] += batch_n_nodes[i]
                
            if model.lap_pos_enc:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            else:
                batch_lap_pos_enc = None

            batch_scores = model.forward(batch_graphs, batch_lap_pos_enc, batch_n_nodes, batch_node_ids)
            shapes = batch_scores.size()
            batch_scores = batch_scores.view(shapes[0]*shapes[1])

            loss = model.loss(batch_scores, batch_labels.float())
            
            loss_ = loss.detach().item()
            epoch_loss += loss_/batch_labels.size()[0]

            if device.type == 'cuda':
                batch_scores_ = batch_scores.detach().cpu().numpy()
                batch_labels_ = batch_labels.detach().cpu().numpy()
            else:
                batch_scores_ = batch_scores.detach().numpy()
                batch_labels_ = batch_labels.detach().numpy()
            
            all_labels.append(batch_labels_)
            all_scores.append(batch_scores_)

        epoch_loss /= (batch_idx + 1)
        all_labels = np.concatenate(all_labels, 0)
        all_scores = np.concatenate(all_scores, 0)
    
    return epoch_loss, all_labels, all_scores


def model_building_pipeline(net_params, train_dataset, validation_dataset, test_dataset):
    device = net_params['device']
    
    print('net_params={}\n\nTotal Parameters: {}\n\n'.format(net_params, view_model_param(net_params)))

    # setting seeds
    random.seed(net_params['seed'])
    np.random.seed(net_params['seed'])
    torch.manual_seed(net_params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(net_params['seed'])
        
    model = GraphTransformer(net_params)
    model = model.to(device)

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
    
    for epoch in range(net_params['epochs']):
        print('Epoch %d' % epoch)
        
        train_loss, optimizer = train_epoch(model, optimizer, device, train_loader)
        torch.save(model.state_dict(), os.path.join(net_params['model_dir'], 'model%s.dat'%(str(epoch))))
        
        train_loss, train_labels, train_scores = evaluation_epoch(model, device, train_loader)
        
        aupr = compute_aupr(train_labels, train_scores)
        auc = compute_roc(train_labels, train_scores)
        print('train, loss:%0.6f, aupr:%0.6f, auc:%0.6f' % (train_loss, aupr, auc))
        
        
        validation_loss, validation_labels, validation_scores = evaluation_epoch(model, device, validation_loader)
        scheduler.step(validation_loss)
        
        aupr = compute_aupr(validation_labels, validation_scores)
        auc = compute_roc(validation_labels, validation_scores)
        f_max, p_max, r_max, t_max, predictions_max = compute_performance_max(validation_labels, validation_scores)
        if predictions_max is not None:
            acc = acc_score(validation_labels, predictions_max)
            mcc = compute_mcc(validation_labels, predictions_max)
        else:
            acc, mcc = 0.0, 0.0
        print('validation, loss:%0.6f, aupr:%0.6f, auc:%0.6f, F_value:%0.6f, mcc:%0.6f, '
              'precision:%0.6f, recall:%0.6f, acc:%0.6f, threshold:%0.6f' % (validation_loss, aupr, auc, f_max, mcc, p_max, r_max, acc, t_max))
        

        test_loss, test_labels, test_scores = evaluation_epoch(model, device, test_loader)
        
        aupr = compute_aupr(test_labels, test_scores)
        auc = compute_roc(test_labels, test_scores)
        f, p, r, predictions = compute_performance(test_labels, test_scores, t_max)
        if predictions is not None:
            acc = acc_score(test_labels, predictions)
            mcc = compute_mcc(test_labels, predictions)
        else:
            acc, mcc = 0, 0
        print('test, loss:%0.6f, aupr:%0.6f, auc:%0.6f, F_value:%0.6f, mcc:%0.6f, '
              'precision:%0.6f, recall:%0.6f, acc:%0.6f, threshold:%0.6f' % (test_loss, aupr, auc, f, mcc, p, r, acc, t_max))
        
        if optimizer.param_groups[0]['lr'] < net_params['min_lr']:
            print("\n!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD.")
            break

def main():
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)
        
    # device
    # if args.gpu_id is not None:
    #     config['gpu']['id'] = int(args.gpu_id)
    #     config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    
    # if args.model_dir is not None:
    #     config['model_dir'] = args.model_dir
        
    if not os.path.exists(config['model_dir']):
        os.makedirs(config['model_dir'])
        
    net_params = config['net_params']
    net_params['device'] = device
    net_params['use_gpu'] = config['gpu']['use']
    net_params['gpu_id'] = config['gpu']['id']
    net_params['model_dir'] = config['model_dir']

    # arg_dict = vars(args)

    # update args
    # for par in net_params.keys():
    #     if par in arg_dict.keys() and arg_dict[par] != None:
    #         net_params[par] = arg_dict[par]

    # net_params['layer_norm'] = not net_params['batch_norm']

    train_data_files = glob.glob(os.path.join(args.data_dir, 'train_*.pkl'))
    test_data_files = glob.glob(os.path.join(args.data_dir, 'test_*.pkl'))
    validation_data_files = glob.glob(os.path.join(args.data_dir, 'validation_*.pkl'))

    if os.path.exists(args.train_feat_stats):
        with open(args.train_feat_stats, 'rb') as f:
            feat_stats = pickle.load(f)

    else:
        feat_stats = dict()
        graph_x_feats, graph_e_feats = [], []
        # partner_feats = []
        for f in train_data_files:
            with open(f, 'rb') as infile:
                data_raw = pickle.load(infile)

            graph_x_feats.append(data_raw[0][0].ndata['feat'])
            graph_e_feats.append(data_raw[0][0].edata['feat'])

            # partner_feats.append(data_raw[1])

        graph_x_feats = torch.cat(graph_x_feats, 0)
        graph_e_feats = torch.cat(graph_e_feats, 0)
        # partner_feats = torch.cat(partner_feats, 0)

        # net_params['in_dim1_node'] = graph_x_feats.size(1)  # moved to arguments
        # net_params['in_dim1_edge'] = graph_e_feats.size(1)
        # net_params['in_dim2'] = partner_feats.size(1)

        for c in range(net_params['in_dim1_node']):
            assert not torch.isnan(graph_x_feats[:, c]).all()
            # In training dataset, check if there exists one feature column having exactly same values.
            # If so, that feature column could be removed
            assert torch.unique(graph_x_feats[:, c][~torch.isnan(graph_x_feats[:, c])]).size()[0] != 1
        for c in range(net_params['in_dim1_edge']):
            assert not torch.isnan(graph_e_feats[:, c]).all()
            assert torch.unique(graph_e_feats[:, c][~torch.isnan(graph_e_feats[:, c])]).size()[0] != 1
        # for c in range(net_params['in_dim2']):
        #     assert not torch.isnan(partner_feats[:, c]).all()
        #     assert torch.unique(partner_feats[:, c][~torch.isnan(partner_feats[:, c])]).size()[0] != 1
        # graph_x_col_mean = torch.nanmean(graph_x_feats, 0)
        feat_stats['graph_x_col_mean'] = torch.mean(graph_x_feats[~graph_x_feats.isnan()], 0)
        # graph_e_col_mean = torch.nanmean(graph_e_feats, 0)
        feat_stats['graph_e_col_mean'] = torch.mean(graph_e_feats[~graph_e_feats.isnan()], 0)

        # partner_col_mean = torch.nanmean(partner_feats, 0)
        # partner_col_mean = torch.mean(partner_feats[~partner_feats.isnan()], 0)

        feat_stats['graph_x_min'] = torch.from_numpy(np.nanmin(graph_x_feats, 0, keepdims=True))
        feat_stats['graph_x_max'] = torch.from_numpy(np.nanmax(graph_x_feats, 0, keepdims=True))
        feat_stats['graph_e_min'] = torch.from_numpy(np.nanmin(graph_e_feats, 0, keepdims=True))
        feat_stats['graph_e_max'] = torch.from_numpy(np.nanmax(graph_e_feats, 0, keepdims=True))
        with open(args.train_feat_stats, 'wb') as f:
            print('save trainig set feature statistics to pickle file...')
            pickle.dump(feat_stats, f)
        # partner_min = torch.from_numpy(np.nanmin(partner_feats, 0, keepdims=True))
        # partner_max = torch.from_numpy(np.nanmax(partner_feats, 0, keepdims=True))
            
    train_dataset = data_generator.LoadDataSet(train_data_files, net_params, **feat_stats)
    validation_dataset = data_generator.LoadDataSet(validation_data_files, net_params, **feat_stats)
    test_dataset = data_generator.LoadDataSet(test_data_files, net_params, **feat_stats)
    
    model_building_pipeline(net_params, train_dataset, validation_dataset, test_dataset)
    
    
if __name__ == '__main__':
    main()
    
    print('done!')
