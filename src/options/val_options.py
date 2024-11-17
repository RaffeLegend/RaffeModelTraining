from src.options.train_options import TrainOptions

class ValOptions(TrainOptions):
    def __init__(self):

        self.initialized = False
        self.metrics = None #accuracy, f1_score, precision
        self.model_path = None

    def initialize(self, parser):
        self.isTrain = False
        self.augmentation = False

        parser.option_part = "val"
        return parser