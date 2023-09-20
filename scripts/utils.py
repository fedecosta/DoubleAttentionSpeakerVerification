# Imports
# ---------------------------------------------------------------------
import os
import torch
from torch.nn import functional as F
import numpy as np
import datetime
import psutil
# ---------------------------------------------------------------------

# Utils
# ---------------------------------------------------------------------
def score(SC, th, rate):

     SC = np.array(SC)

     cond = SC >= th
     if rate == 'FAR':
         score_count = sum(cond)
     else:
         score_count = sum(cond == False)

     return round(score_count * 100 / float(len(SC)), 4)


def scoreCosineDistance(emb1, emb2):

    dist = F.cosine_similarity(emb1, emb2, dim = -1, eps = 1e-08)
    return dist


def chkptsave(opt, model, optimizer, epoch, step, start_datetime):
    ''' function to save the model and optimizer parameters '''
    if torch.cuda.device_count() > 1:
        checkpoint = {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'settings': opt,
            'epoch': epoch,
            'step':step}
    else:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'settings': opt,
            'epoch': epoch,
            'step':step}

    end_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')
    checkpoint['start_datetime'] = start_datetime
    checkpoint['end_datetime'] = end_datetime

    torch.save(checkpoint,'{}/{}_{}.chkpt'.format(opt.out_dir, opt.model_name, step))


def get_number_of_speakers(labels_file_path):

    speakers_set = set()
    with open(labels_file_path, 'r') as f:
        for line in f.readlines():
            speaker_chunk = [chunk for chunk in line.split("/") if chunk.startswith("id")]
            # Only consider directories with /id.../
            if len(speaker_chunk) > 0: 
                speaker_label = speaker_chunk[0]
            speakers_set.add(speaker_label)

    return len(speakers_set)


def generate_model_name(params, start_datetime, wandb_run_id, wandb_run_name):

    # TODO add all neccesary components

    name_components = []

    name_components.append(params.model_name_prefix)
    name_components.append(params.front_end)
    name_components.append(params.pooling_method)

    name_components = [str (component) for component in name_components]

    model_name = "_".join(name_components)

    formatted_datetime = start_datetime.replace(':', '_').replace(' ', '_').replace('-', '_')

    model_name = f"{formatted_datetime}_{model_name}_{wandb_run_id}_{wandb_run_name}"

    return model_name


def get_memory_info(cpu = True, gpu = True):

    cpu_available_pctg, gpu_free = None, None

    # CPU memory info
    if cpu:
        cpu_memory_info = dict(psutil.virtual_memory()._asdict())
        cpu_total = cpu_memory_info["total"]
        cpu_available = cpu_memory_info["available"]
        cpu_available_pctg = cpu_available * 100 / cpu_total

    # GPU memory info
    if gpu:
        if torch.cuda.is_available():
            gpu_free, gpu_occupied = torch.cuda.mem_get_info()
            gpu_free = gpu_free/1000000000
        else:
            gpu_free = None

    return cpu_available_pctg, gpu_free


def format_sc_labels(labels_path, prepend_directories = None):
    
    '''Format Speaker Classification type labels.'''

    # Expected labels line input format: /speaker/interview/file speaker_label -1
    # prepend_directories must be a list of potential directory(s) to prepend

    # Read the paths of the audios and their labels
    with open(labels_path, 'r') as data_labels_file:
        labels_lines = data_labels_file.readlines()

    # Format labels lines
    formatted_labels_lines = []
    for labels_line in labels_lines:

        # Remove end of line character, if has
        labels_line = labels_line.replace("\n", "")

        # If labels are of the form /speaker/interview/file we need to remove the first "/" to join paths
        if labels_line[0] == "/":
            labels_line = labels_line[1:]

        file_path = labels_line.split(" ")[0]
        rest_of_line = labels_line.split(" ")[1:]

        # Remove the file extension (if has) and add the pickle extension to the file path
        # HACK
        if False:
            if len(file_path.split(".")) > 1:

                file_path = '.'.join(file_path.split(".")[:-1])
                file_path = f"{file_path}.pickle"

        # Prepend optional additional directory to the labels paths (but first checks if file exists)
        if prepend_directories is not None:
            data_founded = False
            for dir in prepend_directories:
                if os.path.exists(os.path.join(dir, file_path)):
                    file_path = os.path.join(dir, file_path)
                    data_founded = True
                    break
            assert data_founded, f"{dir} {file_path} not founded."
        else:
            if not os.path.exists(file_path):
                assert data_founded, f"{file_path} not founded."

        labels_line = " ".join([file_path] + rest_of_line)
        
        formatted_labels_lines.append(labels_line)

    return formatted_labels_lines


def format_sv_labels(labels_path, prepend_directories = None):

    '''Format Speaker Verification type labels.'''

    # Expected labels line input format: /speaker_1/interview_/file /speaker_2/interview_/file

    # Read the paths of the audios and their labels
    with open(labels_path, 'r') as data_labels_file:
        labels_lines = data_labels_file.readlines()

    # Format labels lines
    formatted_labels_lines = []
    for labels_line in labels_lines:

        speaker_1 = labels_line.split(" ")[0].strip()
        speaker_2 = labels_line.split(" ")[1].strip()

        # If labels are of the form /speaker/interview/file we need to remove the first "/" to join paths
        if speaker_1[0] == "/":
            speaker_1 = speaker_1[1:]
        if speaker_2[0] == "/":
            speaker_2 = speaker_2[1:]

        # Remove the file extension (if has) and add the pickle extension to the file path
        # TODO
        if False:
            if len(speaker_1.split(".")) > 1:
                speaker_1 = '.'.join(speaker_1.split(".")[:-1]) 
            if len(speaker_2.split(".")) > 1:
                speaker_2 = '.'.join(speaker_2.split(".")[:-1]) 
        
            # Add the pickle extension
            speaker_1 = f"{speaker_1}.pickle"
            speaker_2 = f"{speaker_2}.pickle"

        # Prepend optional additional directory to the labels paths (but first checks if file exists)
        if prepend_directories is not None:
            data_founded = False
            for dir in prepend_directories:
                if os.path.exists(os.path.join(dir, speaker_1)):
                    speaker_1 = os.path.join(dir, speaker_1)
                    data_founded = True
                    break
            assert data_founded, f"{speaker_1} not founded."

            data_founded = False
            for dir in prepend_directories:
                if os.path.exists(os.path.join(dir, speaker_2)):
                    speaker_2 = os.path.join(dir, speaker_2)
                    data_founded = True
                    break
            assert data_founded, f"{speaker_2} not founded."
        else:
            if not os.path.exists(speaker_1):
                assert data_founded, f"{speaker_1} not founded."
            if not os.path.exists(speaker_2):
                assert data_founded, f"{speaker_2} not founded."

        labels_line = f"{speaker_1} {speaker_2}"
        
        formatted_labels_lines.append(labels_line)

    return formatted_labels_lines
# ---------------------------------------------------------------------

# Unnused functions
# ---------------------------------------------------------------------
#def normalize_features(self, features):
#
#    # Used when getting embeddings
#    # TODO move to the corresponding .py
#    
#    norm_features = np.transpose(features)
#    norm_features = norm_features - np.mean(norm_features, axis = 0)
#    
#    return norm_features


# def Score(SC, th, rate):
#
#     score_count = 0.0
#     for sc in SC:
#         if rate == 'FAR':
#             if float(sc) >= float(th):
#                 score_count += 1
#         elif rate == 'FRR':
#             if float(sc) < float(th):
#                 score_count += 1
#
#     return round(score_count * 100 / float(len(SC)), 4)


