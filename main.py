import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from utils import *


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist
    mkdir(config.log_path)
    mkdir(config.model_save_path)
    mkdir(config.sample_path)
    mkdir(config.result_path)

    data_loader = get_loader(config.data_path, config.image_size,
                             config.crop_size, config.batch_size, transform=True, dataset='PascalVOC2012', mode=config.mode)

    # Solver
    solver = Solver(data_loader, vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=21)
    parser.add_argument('--lr', type=float, default=0.0001)

    # Training settings

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--pretrained_model', type=str, default=None)

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    #我们在这里修改路径
    parser.add_argument('--data_path', type=str, default='/fast/users/a1746546/data')
    parser.add_argument('--fcn_model_path', type=str, default='/fast/users/a1746546/code/model_weights/fcn8s_from_caffe.pth')
    parser.add_argument('--vgg_model_path', type=str, default='/fast/users/a1746546/code/model_weights/vgg16_from_caffe.pth')
    parser.add_argument('--log_path', type=str, default='/fast/users/a1746546/code/cspnTest/logs')
    parser.add_argument('--model_save_path', type=str, default='/fast/users/a1746546/code/cspnTest/models')
    parser.add_argument('--sample_path', type=str, default='/fast/users/a1746546/code/cspnTest/samples')
    parser.add_argument('--result_path', type=str, default='/fast/users/a1746546/code/cspnTest/results')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=70)
    parser.add_argument('--model_save_step', type=int, default=1000)
    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    main(config)