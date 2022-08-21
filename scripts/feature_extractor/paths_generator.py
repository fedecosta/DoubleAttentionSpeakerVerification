import argparse

class PathsGenerator:

    def __init__(self):

        self.test = 1
        #self.initialize_training_variables()

    
    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = 'Generate audio paths file as input for the feature exctractor script. \
                It searches audio files in a directory and dumps their paths into a .lst file.',
            )

    
    def add_parser_args(self):

        self.parser.add_argument(
            'load_data_dir',
            type = str, 
            help = 'Data directory containing the audio files we want to extract features from.',
            )

        self.parser.add_argument(
            'dump_data_dir', 
            type = str, 
            default = '/scripts/feature_extractor/feature_extractor_paths.lst', 
            help = 'Data directory where we want to dump the .lst file.',
            )
        
        self.parser.add_argument(
            '--valid_audio_formats', 
            action = 'append',
            default = ['.wav', '.m4a'],
            help = 'Audio files extension to search for.',
            )

        
    def parse_args(self):

        self.params = self.parser.parse_args()
        #params.model_name = getModelName(params)

    
    def main(self):

        self.initialize_parser()
        self.add_parser_args()
        self.parse_args()

        print(self.params)
        

if __name__=="__main__":

    instance = PathsGenerator()
    instance.main()
    