#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH -c4
#SBATCH --mem=32G      # Max CPU Memory
#SBATCH --gres=gpu:4
#SBATCH --job-name=eval_valid
python scripts/model_evaluator.py \
	'./models/23_01_02_15_09_14_vgg_mh_ap_fc_VGGNL_MultiHeadAttentionAttentionPooling_3oxqbxav/' \
	'23_01_02_15_09_14_vgg_mh_ap_fc_VGGNL_MultiHeadAttentionAttentionPooling_3oxqbxav.chkpt' \
	'./labels/valid/voxceleb_2/22_12_26_12_41_44_473354ug_dandy-darkness-45/clients.ndx' \
	'./labels/valid/voxceleb_2/22_12_26_12_43_22_3jijl4ik_celestial-flower-46/impostors.ndx' \
	--dump_folder './models_results/' \
	--data_dir '/home/usuaris/veussd/federico.costa/datasets/voxceleb_2/dev/22_12_05_21_59_21_f0yycx91_azure-pine-7/' \
	--evaluation_type 'total_length' \
	> logs/console_output/model_evaluator/model_evaluator_valid_3oxqbxav.log 2>&1