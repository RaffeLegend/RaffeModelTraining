from .base_options import BaseOptions


class DataOptions(BaseOptions):
    def __init__(self):

        # Data augmentation
        self.augmentation = None
        self.image_height = None
        self.image_weight = None
        self.normalization_mean = None
        self.normalization_std  = None
        self.batch_size = None
        self.shuffle = None

        self.dataset_path = None

        # self.pin_memory = None
        #Testing O
        self.test_batch_size = None
        self.test_data_path = None

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--model_path')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')

        self.isTrain = False
        return parser
    

class TrainOptions(DataOptions):
    def __init__(self):

        # Data augmentation
        self.augmentation = None
        self.rz_interp = None
        self.blur_prob = None
        self.blur_sig = None
        self.resize_or_crop = None
        self.no_flip = None

        # Data loading
        self.dataset_path = None
        self.train_split = None
        self.load_size = None
        self.crop_size = None
        
        self.image_height = None
        self.image_weight = None
        self.normalization_mean = None
        self.normalization_std  = None
        self.batch_size = None
        self.shuffle = None

        self.dataset_path = None

        # self.pin_memory = None
        #Testing O
        self.test_batch_size = None
        self.test_data_path = None

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # Data augmentation
        parser.add_argument('--rz_interp', default='bilinear')
        parser.add_argument('--blur_prob', type=float, default=0.5)
        parser.add_argument('--blur_sig', default='0.0,3.0')
        parser.add_argument('--resize_or_crop', type=str, default='scale_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

        # Data loading
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--cropSize', type=int, default=224, help='then crop to this size')


        parser.add_argument('--model_path')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')

        self.isTrain = False
        return parser