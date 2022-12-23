#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH -c1
#SBATCH --mem=64G      # Max CPU Memory
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_h
python scripts/model_evaluator.py \
	'./models/22_12_14_09_48_59_vgg_tmh_ap_fc_VGGNL_TransformerStackedAttentionPooling_c0whzsyq/' \
	'22_12_14_09_48_59_vgg_tmh_ap_fc_VGGNL_TransformerStackedAttentionPooling_c0whzsyq.chkpt' \
	'./labels/test/voxceleb1_h_protocol/22_12_09_14_13_45_27ptmd8u_bright-brook-28/clients.ndx' \
	'./labels/test/voxceleb1_h_protocol/22_12_09_14_15_18_2mwdbd4n_snowy-sound-29/impostors.ndx' \
	--dump_folder './models_results/' \
	--data_dir './datasets/voxceleb_1/dev/22_12_05_21_59_21_1tu20n33_dainty-universe-5' './datasets/voxceleb_1/test/22_12_05_21_59_21_10q28l5i_splendid-thunder-6' \
	--evaluation_type "total_length" \
	> logs/console_output/model_evaluator/model_evaluator_h_2.log 2>&1