from src.options.base_options import BaseOptions


class TestOptions(BaseOptions):
    
    def __init__(self):

        self.save_predictions = None
        self.visualize_results = None
        self.result_save_path = None

        self.metrics = None #accuracy, f1_score, precision

    def initialize(self, parser):
        # parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--result_save_path', default='binary')
        parser.add_argument('--model_path')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')

        self.isTrain = False
        return parser
