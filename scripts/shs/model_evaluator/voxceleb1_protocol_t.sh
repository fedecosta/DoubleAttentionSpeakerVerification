#!/bin/bash
#SBATCH --output=logs/sbatch_output/slurm-%j.out
#SBATCH -p veu             # Partition to submit to
#SBATCH -c1
#SBATCH --mem=32G      # Max CPU Memory
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval
python scripts/model_evaluator.py \
	'./models/22_12_09_19_02_40_pg_tmh_ap_fc_PatchsGenerator_TransformerStackedAttentionPooling_1w22yjpa' \
	'22_12_09_19_02_40_pg_tmh_ap_fc_PatchsGenerator_TransformerStackedAttentionPooling_1w22yjpa.chkpt' \
	'./labels/test/voxceleb1_t_protocol/22_12_09_13_09_15_2tkbqp0k_revived-bush-19/clients.ndx' \
	'./labels/test/voxceleb1_t_protocol/22_12_09_13_11_45_1366qjbx_avid-pyramid-20/impostors.ndx' \
	--dump_folder './models_results/' \
	--data_dir './datasets/voxceleb_1/test/22_12_05_21_59_21_10q28l5i_splendid-thunder-6' \
	--evaluation_type "total_length" \
	> logs/console_output/model_evaluator/3_model_evaluator.log 2>&1