import argparse
import os

from settings import PATHS_GENERATOR_DEFAULT_SETTINGS

# TODO add the usage instructions in README.md

class PathsGenerator:

    def __init__(self, params):
        self.params = params
    
    
    def search_files(self):

        self.lines_to_write = []

        print(f"Searching {self.params.valid_audio_formats} files in {self.params.load_data_folder}")
        
        for (dir_path, dir_names, file_names) in os.walk(self.params.load_data_folder):

                if self.params.verbose: print(f"Searching in {dir_path}")

                for file_name in file_names:
        
                    if file_name.split(".")[-1] in self.params.valid_audio_formats:
                        
                        path_to_write = os.path.join(dir_path, file_name)
                        self.lines_to_write.append(path_to_write)

        print(f"{len(self.lines_to_write)} files founded in {self.params.load_data_folder}")

    
    def dump_paths(self):

        if not os.path.exists(self.params.dump_file_folder):
            os.makedirs(self.params.dump_file_folder)
        dump_path = os.path.join(self.params.dump_file_folder, self.params.dump_file_name)

        print(f"Dumping files paths into {dump_path}")
        
        with open(dump_path, 'w') as file:

            for line_to_write in self.lines_to_write:
                file.write(line_to_write)
                file.write('\n')

        print(f"{len(self.lines_to_write)} files paths dumped in {dump_path}")

    
    def main(self):

        self.search_files()
        self.dump_paths()
        

class ArgsParser:

    def __init__(self):
        
        self.initialize_parser()

    
    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = 'Generate audio paths file as input for the feature exctractor script. \
                It searches audio files in a directory and dumps their paths into a .lst file.',
            )


    def add_parser_args(self):

        self.parser.add_argument(
            'load_data_folder',
            type = str, 
            help = 'Data folder containing the audio files we want to extract features from.',
            )

        self.parser.add_argument(
            '--dump_file_name', 
            type = str, 
            default = PATHS_GENERATOR_DEFAULT_SETTINGS['dump_file_name'], 
            help = 'Name of the .lst file we want to dump paths into.',
            )

        self.parser.add_argument(
            '--dump_file_folder', 
            type = str, 
            default = PATHS_GENERATOR_DEFAULT_SETTINGS['dump_file_folder'], 
            help = 'Data folder where we want to dump the .lst file.',
            )
        
        self.parser.add_argument(
            '--valid_audio_formats', 
            action = 'append',
            default = PATHS_GENERATOR_DEFAULT_SETTINGS['valid_audio_formats'],
            help = 'Audio files extension to search for.',
            )

        self.parser.add_argument(
            "--verbose", 
            action = argparse.BooleanOptionalAction,
            default = PATHS_GENERATOR_DEFAULT_SETTINGS['verbose'],
            help = "Increase output verbosity.",
            )


    def main(self):

        self.add_parser_args()
        self.arguments = self.parser.parse_args()


if __name__=="__main__":

    args_parser = ArgsParser()
    args_parser.main()
    parameters = args_parser.arguments

    paths_generator = PathsGenerator(parameters)
    paths_generator.main()
    