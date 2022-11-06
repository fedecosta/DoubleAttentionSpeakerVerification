import torch
from torch.nn import functional as F
import numpy as np
import datetime


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
            speaker_label = line.split()[-2]
            speakers_set.add(speaker_label)

    return len(speakers_set)


def generate_model_name(params, start_datetime, wandb_run_id):

    # TODO add all neccesary components

    name_components = []

    name_components.append(params.model_name_prefix)
    name_components.append(params.front_end)
    name_components.append(params.pooling_method)
    
    
    #name_components.append('batch_size' + str(params.batch_size))
    #name_components.append(params.heads_number)
    #name_components.append(str(params.window_size))
    #name_components.append('lr' + params.learning_rate)
    #name_components.append('weight_decay' + params.weight_decay)
    #name_components.append('kernel' + params.kernel_size)
    #name_components.append('emb_size' + params.embedding_size)
    #name_components.append('s' + params.scaling_factor)
    #name_components.append('m' + params.margin_factor)
    

    name_components = [str (component) for component in name_components]

    model_name = "_".join(name_components)

    formatted_datetime = start_datetime.replace(':', '_').replace(' ', '_').replace('-', '_')

    model_name = f"{formatted_datetime}_{model_name}_{wandb_run_id}"

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



#def normalize_features(self, features):
#
#    # Used when getting embeddings
#    # TODO move to the corresponding .py
#    
#    norm_features = np.transpose(features)
#    norm_features = norm_features - np.mean(norm_features, axis = 0)
#    
#    return norm_features


