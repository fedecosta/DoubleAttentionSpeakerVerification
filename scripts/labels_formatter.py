import argparse
import os
import warnings


class LabelsFormatter:

    def __init__(self, input_params):

        self.input_params = input_params

    
    def format_original_labels(self):

        self.original_labels_path = os.path.join(
            self.input_params.original_labels_folder, 
            self.input_params.original_labels_file_name,
            )

        self.clients = []
        self.impostors = []

        with open(self.original_labels_path,'r') as original_labels_file:

            for num_line, line in enumerate(original_labels_file):

                # Every line has the format: label speaker_1.extension speaker_2.extension\n
                
                # Remove '\n'
                line = line.replace('\n', '')
                
                label, speaker_1_path, speaker_2_path = line.split(" ")

                # Remove the file extension
                speaker_1_path = '.'.join(speaker_1_path.split(".")[:-1])
                speaker_2_path = '.'.join(speaker_2_path.split(".")[:-1])

                dump_line = f"{speaker_1_path} {speaker_2_path}"

                if label == "0":
                    self.impostors.append(dump_line)
                elif label == "1":
                    self.clients.append(dump_line)
                else:
                    warnings.warn(f"Label not 1 or 0 at line {num_line}")
                    break

    
    def count_labels(self):

        self.total_labels = sum(1 for line in open(self.original_labels_path))
        self.total_clients = len(self.clients)
        self.total_impostors = len(self.impostors)

        print(f"Total labels: {self.total_labels}")
        print(f"{self.total_clients} clients and {self.total_impostors} impostors.")


    def dump_formatted_labels(self, lines_to_dump, dump_labels_folder, dump_labels_file_name):

        if not os.path.exists(dump_labels_folder):
            os.makedirs(dump_labels_folder)
        
        dump_path = os.path.join(dump_labels_folder, dump_labels_file_name)

        with open(dump_path, 'w') as f:
            for line in lines_to_dump:
                f.write(line)
                f.write('\n')
            f.close()

        print(f"Labels saved into {dump_path}")

    
    def dump_formatted_clients_labels(self):

        print(f"Saving {self.total_clients} clients labels...")
        
        self.dump_formatted_labels(
            lines_to_dump = self.clients, 
            dump_labels_folder = self.input_params.dump_labels_folder, 
            dump_labels_file_name = self.input_params.dump_labels_file_name_clients,
            )

        print("Clients labels saved.")

    
    def dump_formatted_impostors_labels(self):

        print(f"Saving {self.total_impostors} impostors labels...")
        
        self.dump_formatted_labels(
            lines_to_dump = self.impostors, 
            dump_labels_folder = self.input_params.dump_labels_folder, 
            dump_labels_file_name = self.input_params.dump_labels_file_name_impostors,
            )

        print("Impostors labels saved.")

    
    def main(self):
        self.format_original_labels()
        self.count_labels()
        self.dump_formatted_clients_labels()
        self.dump_formatted_impostors_labels()
        
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description = 'Format original VoxCeleb evaluation protocols into train.py readable labels',
        )

    parser.add_argument(
        '--original_labels_folder', 
        type = str, 
        ) 

    parser.add_argument(
        '--original_labels_file_name', 
        type = str, 
        )

    parser.add_argument(
        '--dump_labels_folder', 
        type = str, 
        )

    parser.add_argument(
        '--dump_labels_file_name_clients', 
        type = str, 
        default = 'clients.ndx',
        help = '',
        )

    parser.add_argument(
        '--dump_labels_file_name_impostors', 
        type = str, 
        default = 'impostors.ndx',
        help = '',
        )

    input_params = parser.parse_args()
        
    labels_formatter = LabelsFormatter(input_params)
    labels_formatter.main()