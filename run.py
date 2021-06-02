import random
import argparse, yaml
import cfgs.config as config
from easydict import EasyDict as edict
# from transformers.modeling_bert import

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='CoES')

    parser.add_argument('--run', dest='run_mode',
                        choices=['pretrain', 'train', 'val', 'test'],
                        help='{pretrain, train, val, test,}',
                        type=str, required=True)

    parser.add_argument('--model',
                        choices=['lstm'],
                        help='{lstm, cnn, cnn-lstm}',
                        type=str, required=True
                        )

    parser.add_argument('--dataset', dest='dataset',
                        choices=['mr','sst2', 'sst5'],
                        help='{datasets for downsteam tasks}',
                        default='None of it', type=str)

    parser.add_argument('--gpu', dest='gpu',
                        help="gpu select, eg.'0, 1, 2'",
                        type=str,
                        default="")

    parser.add_argument('--cpu',
                        action='store_true',
                        default=False,
                        )

    parser.add_argument('--fine_tune',
                        action='store_true',
                        default=False,
                        )

    parser.add_argument('--seed', dest='seed',
                        help='fix random seed',
                        type=int,
                        default=random.randint(0, 99999999))

    parser.add_argument('--version', dest='version',
                        help='version control',
                        type=str,
                        default="default")
    args = parser.parse_args()
    return args

# run example:
# python run.py --run pretrain --gpu 0,1 --version test
# python run.py --run train --dataset mr --gpu 0,1 --version test
if __name__ == '__main__':
    __C = config.__C

    args = parse_args()
    cfg_file = "cfgs/model.yml"
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.load(f)
        f.close()
    cfg_file_special = "cfgs/{}.yml".format(args.model)
    with open(cfg_file_special, 'r') as f_s:
        yaml_dict_special = yaml.load(f_s)
        f_s.close()
    args_dict = edict({**yaml_dict, **yaml_dict_special, **vars(args)})

    config.add_edit(args_dict, __C)
    config.proc(__C)

    print('Hyper Parameters:')
    config.config_print(__C)

    # __C.check_path()
    from trainer.trainer import Trainer

    execution = Trainer(__C)
    execution.run(__C.run_mode)
