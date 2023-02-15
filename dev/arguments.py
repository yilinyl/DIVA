import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.json', help="Config file path (.json)")
    parser.add_argument('--gpu_id', type=int, help="GPU device ID")
    parser.add_argument('--model_dir', help="Model directory")

    args = parser.parse_args()

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

