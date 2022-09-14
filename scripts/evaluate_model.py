import argparse

from train_3 import Trainer
from settings import EVALUATE_MODEL_DEFAULT_SETTINGS

class ModelEvaluator:

    def __init__(self, params):

        self.params = params

    
    def load_model(self):


class ArgsParser:

    def __init__(self):
        
        self.initialize_parser()

    
    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = 'Evaluate a trained model with a validation and/or test set.',
            )


    def add_parser_args(self):

        # Verbosity and debug Parameters
        
        self.parser.add_argument(
            "--verbose", 
            action = "store_true", # TODO understand store_true vs store_false
            default = EVALUATE_MODEL_DEFAULT_SETTINGS['verbose'],
            help = "Increase output verbosity.",
            )


    def main(self):

        self.add_parser_args()
        self.arguments = self.parser.parse_args()


if __name__== "__main__":

    args_parser = ArgsParser()
    args_parser.main()
    parameters = args_parser.arguments

    model_evaluator = ModelEvaluator(parameters)
    model_evaluator.main()