from src.options.train_options import TrainOptions

class ValOptions(TrainOptions):
    def __init__(self):
        self.metrics = None #accuracy, f1_score, precision
        self.model_path = None

    def update(self, args):

        args.augmentation = False

    def initialize(self, parser):

        parser.add_argument('--model_path', type=str, default='', help="the model path need to be validated")

        self.isTrain = False
        return parser