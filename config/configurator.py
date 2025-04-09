import os
import yaml
import argparse

def parse_configure():
    parser = argparse.ArgumentParser(description='KGMBRec')
    parser.add_argument('--model', type=str, default='kgmbr', help='Model name')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    args = parser.parse_args()

    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    if args.model == None:
        raise Exception("Please provide the model name through --model.")
    model_name = args.model.lower()
    if not os.path.exists('./config/modelconf/{}.yml'.format(model_name)):
        raise Exception("Please create the yaml file for your model first.")

    with open('./config/modelconf/{}.yml'.format(model_name), encoding='utf-8') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)
        configs['model']['name'] = configs['model']['name'].lower()
        if 'tune' not in configs:
            configs['tune'] = {'enable': False}
        configs['device'] = args.device
        if args.dataset is not None:
            configs['data']['name'] = args.dataset
        return configs

configs = parse_configure()
