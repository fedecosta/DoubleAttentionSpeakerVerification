#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH -c4
#SBATCH --mem=32G      # Max CPU Memory
#SBATCH --gres=gpu:4
#SBATCH --job-name=eval_valid
python scripts/model_evaluator.py \
	'./models/23_04_23_11_08_32_vgg_tmh_ap_fc_VGGNL_TransformerStackedAttentionPooling_3f825m2w_revived-morning-4/' \
	'23_04_23_11_08_32_vgg_tmh_ap_fc_VGGNL_TransformerStackedAttentionPooling_3f825m2w_revived-morning-4 copy.chkpt' \
	'./labels/valid/voxceleb_2/23_04_23_00_43_37_3f137ioe_dainty-blaze-71/sv_clients_hard.ndx' \
	'./labels/valid/voxceleb_2/23_04_23_00_43_37_10utmars_royal-plasma-71/sv_impostors_hard.ndx' \
	--dump_folder './models_results/' \
	--data_dir '/home/usuaris/veussd/federico.costa/datasets/voxceleb_2/dev/22_12_05_21_59_21_f0yycx91_azure-pine-7/' \
	--evaluation_type 'total_length' \
	> logs/console_output/model_evaluator/validation_improvements/sv_hard.log 2>&1