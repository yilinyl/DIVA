import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.json', help="Config file path (.json)")
    parser.add_argument('--gpu_id', type=int, help="GPU device ID")
    parser.add_argument('--model_dir', help="Model directory")

    parser.add_argument('--data_dir', default='/local/storage/yl986/3d_vip/test_data', help="Data Directory")
    # data_args = parser.add_argument_group('data')
    # data_args.add_argument('--graph_cache', help='Directory for pre-constructed protein graphs')
    # data_args.add_argument('--pdb_root_dir', help='PDB structure directory')
    # data_args.add_argument('--af_root_dir', help='Alpha-fold structure directory')
    # data_args.add_argument('--feat_dir', help='Precomputed feature directory')
    # data_args.add_argument('--cov_thres', default=0.5, type=float, help='Threshold for PDB coverage')
    # data_args.add_argument('--num_neighbors', default=10, type=int, help='Number of neighbors in building k-NN protein graph')
    # data_args.add_argument('--distance_type', default='centroid', help='Distance type')
    # data_args.add_argument('--method', default='radius', help='Method for building protein graph')
    # data_args.add_argument('--radius', default=10, type=int, help='radius')
    # data_args.add_argument('--save', default=False, type=bool, help='Save protein graph to cache')
    # data_args.add_argument('--anno_ires', default=False, type=bool, help='Annotate interface information or not')
    # # data_args.add_argument('--coord_option', default=0.5, type=float, help='Threshold for PDB coverage')
    # data_args.add_argument('--pos_enc_dim', type=int, help="Position encoding dimension")
    # data_args.add_argument('--lap_pos_enc', type=bool, help="Apply Laplacian position encoding")
    # data_args.add_argument('--wl_pos_enc', type=bool, help="Apply WL position encoding")

    # model_args = parser.add_argument_group('model')

    # parser.add_argument('--train_feat_stats', default='train_stats.pkl', help="Feature statistics among training set (for feature scaling)")
    parser.add_argument('--seed', default=100, type=int, help="Please give a value for seed")
    parser.add_argument('--epochs', type=int, help="Training epochs")
    parser.add_argument('--batch_size', type=int, help="Please give a value for batch_size")
    parser.add_argument('--init_lr', default=0.0001, type=float, help="Initial learning rate")
    parser.add_argument('--lr_reduce_factor', default=0.5, help="Learning rate reduce factor")
    parser.add_argument('--lr_schedule_patience', type=int,
                        help="Number of epochs with no improvement after which learning rate will be reduced")
    parser.add_argument('--min_lr', default=1e-6, help="Mininum learning rate")
    parser.add_argument('--weight_decay', default=0.0, help="weight decay value")
    parser.add_argument('--print_epoch_interval', default=5, type=int, help="Print frequency")
    parser.add_argument('--L', type=int, help="Graph Transformer number of layers")
    parser.add_argument('--in_dim1_node', type=int, help="Node feature input dimension")
    parser.add_argument('--in_dim1_edge', type=int, help="Edge feature input dimension")
    parser.add_argument('--hidden_dim1', type=int, help="Graph transformer embedding dimension")
    # parser.add_argument('--hidden_dim2', type=int, help="Please give a value for hidden_dim2")
    parser.add_argument('--out_dim1', type=int, help="Graph transformer output dimension")
    # parser.add_argument('--out_dim2', type=int, help="Please give a value for out_dim2")
    parser.add_argument('--residual', default=True, type=bool, help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', default="mean", help="Readout method")
    parser.add_argument('--n_heads', type=int, help="Number of attention heads")
    parser.add_argument('--in_feat_dropout', default=0.0, help="dropout rate on input feature")
    parser.add_argument('--dropout', default=0.0, help="Dropout rate")
    parser.add_argument('--layer_norm', default=False, type=bool, help="Option to use layer_norm (default=False)")
    parser.add_argument('--batch_norm', default=True, type=bool, help="Option to use batch_norm (defaul=True)")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")

    parser.add_argument('--embed_graph', default=False, type=bool, help="Graph embedding (default=False)")

    args = parser.parse_args()
    # arg_groups = dict()
    #
    # for group in parser._action_groups:
    #     group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
    #     arg_groups[group.title] = argparse.Namespace(**group_dict)

    return args


# if __name__ == '__main__':
#     import json
#     args = parse_args()
#     arg_dict = vars(args)
#
#     with open(args.config) as f:
#         config = json.load(f)
#
#     # device
#     if args.gpu_id is not None:
#         config['gpu']['id'] = int(args.gpu_id)
#         config['gpu']['use'] = True
#
#     net_params = config['net_params']
#     for par in net_params.keys():
#         if par in arg_dict.keys() and arg_dict[par] != None:
#             net_params[par] = arg_dict[par]

