import torch
from torch.nn import functional as F
import numpy as np


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

    dist = F.cosine_similarity(emb1,emb2, dim=-1, eps=1e-08)
    return dist

def chkptsave(opt,model,optimizer,epoch,step):
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

    torch.save(checkpoint,'{}/{}_{}.chkpt'.format(opt.out_dir, opt.model_name,step))

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

def generate_model_name(params):

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

    return model_name
