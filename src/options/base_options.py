import argparse
import os
from abc import ABC, abstractmethod
from src.utils.tools import mkdirs
import torch

# define the base options of the tasks
class BaseOptions(ABC):
    def __init__(self):
        # self.initialized = False

        # General Options: device, log
        self.name = None # Experiment name
        self.initialized = False
        # self.device = None
        # self.num_workerrs = None
        # self.seed = None

        # self.task = None

        self.log_dir = None

        # Model Options
        self.num_classes = None
        self.encoder = None
        self.decoder = None

        # Distribution Options
        # self.distributed = None
        # self.local_rank = None
        # self.world_size = None

        # Precision Options
        # self.mixed_precision = None
        # self.grad_clip = None


    @abstractmethod
    def update(self, args):
        self.name = args.name
        self.log_dir = args.log_dir
        self.num_classes = args.num_classes
        self.encoder = args.encoder
        self.decoder = args.decoder

    def initialize(self, parser):
        parser.add_argument('--log_dir', type=str, default="./log_dir")
        parser.add_argument('--num_classes', type=int, default=1)
        parser.add_argument('--encoder', type=str, default="Imagenet:resnet50")
        parser.add_argument('--decoder', type=str, default="fc")
        
        parser.add_argument('--name', type=str, default='code_debug', help='name of the experiment. It decides where to store samples and models')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.update(opt)
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.log_dir, opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()

        if print_options:
            self.print_options(opt)

        self.opt = opt
        return self.opt
