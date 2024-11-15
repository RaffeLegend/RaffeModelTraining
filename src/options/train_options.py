from src.options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):

        self.isTrain = True

        # Model Options
        self.pretrained = None
        self.arch_name = None

        # Model Saving
        self.model_save_path = None
        self.save_best_only = None
        self.early_stopping = None
        self.save_epoch_freq = None

        # Training Options
        self.epochs = None
        self.loss_function = None

        # Training strategy
        self.learning_rate = None
        self.optimizer = None
        self.momentum = None
        self.weight_decay = None
        self.schedular = None # stepLR # CosineAnnealingLR
        self.beta = None # momentum term of adam

        # Initial strategy
        self.init_type = None
        self.init_gain = None

        # show
        self.loss_freq = None

    def update(self, args):

        self.pretrained = args.pretrained
        self.arch_name = args.arch_name
        self.model_save_path = args.model_save_path
        self.save_best_only = args.save_best_only
        self.early_stopping = args.early_stopping
        self.save_epoch_freq = args.save_epoch_freq
        self.epochs = args.epochs
        self.loss_function = args.loss_function
        self.learning_rate = args.learning_rate
        self.optimizer = args.optimizer
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.schedular = args.schedular # stepLR # CosineAnnealingLR
        self.beta = args.beta # momentum term of adam
        self.init_type = args.init_type
        self.init_gain = args.init_gain
        self.loss_freq = args.show_loss_freq

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        
        parser.add_argument('--pretrained', type=bool, default=True, help='load the pretrained model or not')
        parser.add_argument('--arch_name', type=str, default='resnet50', help='see my_models/__init__.py')

        parser.add_argument('--model_save_path', type=str, default='./', help='the saving path of trained model')
        parser.add_argument('--save_best_only', type=bool, default=False, help='save the best model or all the models')
        parser.add_argument('--early_stopping', type=int, default=30, help='the epoch of earlystopping')

        parser.add_argument('--epochs', type=int, default=100, help='the epoch of training')
        parser.add_argument('--loss_function', type=str, default='bce', help='the loss function for model training')
        parser.add_argument('--learning_rate', type=float, default=0.001, help="the learning rate of model training")
        parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer [sgd, adam] of model training')
        parser.add_argument('--schedular', type=str, default='', help='the schedular of the learning rate decay')
        parser.add_argument('--beta', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--weight_decay', type=float, default=0.0, help='loss weight for l2 reg')
        parser.add_argument('--momentum', type=float, default=0.0, help='momentum of SGD optimizer')

        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')

        parser.add_argument('--show_loss_freq', type=int, default=400, help='frequency of showing loss on tensorboard')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')

        self.update(parser)

        return parser
