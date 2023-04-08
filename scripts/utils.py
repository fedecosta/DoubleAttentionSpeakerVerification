import torch
from torch.nn import functional as F
import numpy as np
import datetime
import psutil


def Score(SC, th, rate):

    score_count = 0.0
    for sc in SC:
        if rate == 'FAR':
            if float(sc) >= float(th):
                score_count += 1
        elif rate == 'FRR':
            if float(sc) < float(th):
                score_count += 1

    return round(score_count * 100 / float(len(SC)), 4)


def Score_2(SC, th, rate):

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


def Accuracy(pred, labels):

    acc = 0.0
    num_pred = pred.size()[0]
    pred = torch.max(pred, 1)[1]
    for idx in range(num_pred):
        if pred[idx].item() == labels[idx].item():
            acc += 1

    return acc/num_pred


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


def calculate_EER(clients_similarities, impostors_similarities):

    # Given clients and impostors similarities, calculate EER

    thresholds = np.arange(-1, 1, 0.01)
    FRR, FAR = np.zeros(len(thresholds)), np.zeros(len(thresholds))
    for idx, th in enumerate(thresholds):
        FRR[idx] = Score(clients_similarities, th, 'FRR')
        FAR[idx] = Score(impostors_similarities, th, 'FAR')

    EER_Idx = np.argwhere(np.diff(np.sign(FAR - FRR)) != 0).reshape(-1)
    if len(EER_Idx) > 0:
        if len(EER_Idx) > 1:
            EER_Idx = EER_Idx[0]
        EER = round((FAR[int(EER_Idx)] + FRR[int(EER_Idx)]) / 2, 4)
    else:
        EER = 50.00

    return EER


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

    


#def normalize_features(self, features):
#
#    # Used when getting embeddings
#    # TODO move to the corresponding .py
#    
#    norm_features = np.transpose(features)
#    norm_features = norm_features - np.mean(norm_features, axis = 0)
#    
#    return norm_features


