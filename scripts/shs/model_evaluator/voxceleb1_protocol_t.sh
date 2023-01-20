#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH -c4
#SBATCH --mem=32G      # Max CPU Memory
#SBATCH --gres=gpu:2
#SBATCH --job-name=eval_t
python scripts/model_evaluator.py \
	'/home/usuaris/veussd/federico.costa/models/23_01_19_12_28_21_vgg_dmha_fc_aug_VGGNL_DoubleMHA_15cqf6t5/' \
	'23_01_19_12_28_21_vgg_dmha_fc_aug_VGGNL_DoubleMHA_15cqf6t5.chkpt' \
	'./labels/test/voxceleb1_t_protocol/22_12_09_13_09_15_2tkbqp0k_revived-bush-19/clients.ndx' \
	'./labels/test/voxceleb1_t_protocol/22_12_09_13_11_45_1366qjbx_avid-pyramid-20/impostors.ndx' \
	--dump_folder './models_results/' \
	--data_dir '/home/usuaris/veussd/federico.costa/datasets/voxceleb_1/test/23_01_19_10_50_02_2z4ix3k0_divine-wind-20/' \
	--evaluation_type "total_length" \
	> logs/console_output/model_evaluator/model_evaluator_t.log 2>&1