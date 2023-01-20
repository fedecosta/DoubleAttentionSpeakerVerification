#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH -c1
#SBATCH --mem=64G      # Max CPU Memory
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_e
python scripts/model_evaluator.py \
	'./models/22_12_26_12_37_22_vgg_tmh_ap_fc_VGGNL_TransformerStackedAttentionPooling_34o9mz0j/' \
	'22_12_26_12_37_22_vgg_tmh_ap_fc_VGGNL_TransformerStackedAttentionPooling_34o9mz0j.chkpt' \
	'./labels/test/voxceleb1_e_protocol/22_12_09_14_16_34_3o9mycu1_gallant-butterfly-30/clients.ndx' \
	'./labels/test/voxceleb1_e_protocol/22_12_23_21_14_01_23g405j0_comfy-glitter-38/impostors.ndx' \
	--dump_folder './models_results/' \
	--data_dir './datasets/voxceleb_1/dev/22_12_05_21_59_21_1tu20n33_dainty-universe-5' './datasets/voxceleb_1/test/22_12_05_21_59_21_10q28l5i_splendid-thunder-6' \
	--evaluation_type "total_length" \
	> logs/console_output/model_evaluator/model_evaluator_e.log 2>&1