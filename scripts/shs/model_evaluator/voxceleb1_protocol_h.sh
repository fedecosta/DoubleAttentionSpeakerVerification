#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH -c4
#SBATCH --mem=32G      # Max CPU Memory
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_h
python scripts/model_evaluator.py \
	'./models/23_04_23_11_08_32_vgg_tmh_ap_fc_VGGNL_TransformerStackedAttentionPooling_3f825m2w_revived-morning-4/' \
	'23_04_23_11_08_32_vgg_tmh_ap_fc_VGGNL_TransformerStackedAttentionPooling_3f825m2w_revived-morning-4.chkpt' \
	'./labels/test/voxceleb1_h_protocol/22_12_09_14_13_45_27ptmd8u_bright-brook-28/clients.ndx' \
	'./labels/test/voxceleb1_h_protocol/22_12_09_14_15_18_2mwdbd4n_snowy-sound-29/impostors.ndx' \
	--dump_folder './models_results/' \
	--data_dir '/home/usuaris/veussd/federico.costa/datasets/voxceleb_1/dev/23_01_19_10_54_31_hifkm2ss_soft-mountain-21/' '/home/usuaris/veussd/federico.costa/datasets/voxceleb_1/test/23_01_19_10_50_02_2z4ix3k0_divine-wind-20/' \
	--evaluation_type 'total_length' \
	--log_file_folder './logs/model_evaluator/' \
	--log_file_name 'model_evaluator_h_3f825m2w_revived-morning-4.log'