from .base_options import BaseOptions


class DataOptions(BaseOptions):
    def __init__(self):

        # Data augmentation
        self.augmentation = None
        self.image_height = None
        self.image_weight = None
        self.normalization = None
        self.batch_size = None
        self.shuffle = None
        self.dataset_path = None

    def update(self, args):
        self.augmentation = args.augmentation
        self.image_height = args.image_height
        self.image_weight = args.image_weight
        self.normalization = args.normalization
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.dataset_path = args.dataset_path

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--image_height', type=int, default=256, help='image height')
        parser.add_argument('--image_weight', type=int, default=256, help='image weight')        
        parser.add_argument('--augmentation', type=bool, default=True, help='train should be true, val/test should be false')        
        parser.add_argument('--normalization', type=str, default='imagenet', help='normalization params [imagenet, clip]')        
        parser.add_argument('--batch_size', type=int, default=256, help='batch size of data')
        parser.add_argument('--shuffle', type=bool, default=True, help='shuffle the data')
        parser.add_argument('--dataset_path', type=str, default='', help='the path of dataset')        

        return parser
    

class TrainDataOptions(DataOptions):
    def __init__(self):

        # Data augmentation
        self.no_flip = None

    def update(self, args):
        self.no_flip = args.no_flip

    def initialize(self, parser):
        parser = DataOptions.initialize(self, parser)

        # Data augmentation
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip')

        return parser
    

class ValDataOptions(DataOptions):
    def __init__(self):

        # Data augmentation
        self.no_flip = None

    def update(self, args):
        self.augmentation = False

    def initialize(self, parser):
        parser = DataOptions.initialize(self, parser)

        # Data augmentation
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip')

        return parser
    
class TestDataOptions(DataOptions):
    def __init__(self):

        # Data augmentation
        self.no_flip = None

    def update(self, args):
        self.augmentation = False
        self.batch_size = args.batch_size

    def initialize(self, parser):
        parser = DataOptions.initialize(self, parser)

        # Data augmentation
        parser.add_argument('--batch_size', type=int, default=1, help='batch size of testing data')

        return parser