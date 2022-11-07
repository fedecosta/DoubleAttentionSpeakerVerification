import wandb
import os
import torch

run = wandb.init(project = "speaker_verification", job_type = "test")

model_artifact = run.use_artifact('22_11_06_18_59_47_tiny_check_VGGNL_Attention_1cw3b1r3:latest')

datadir = model_artifact.download()

checkpoint = torch.load(
    os.path.join(datadir, '22_11_06_18_59_47_tiny_check_VGGNL_Attention_1cw3b1r3.chkpt'), 
    map_location = "cuda",
    )

run.summary.update({"vox_celeb_1_t_eer": 64})