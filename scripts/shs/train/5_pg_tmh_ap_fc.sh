#!/bin/bash
#SBATCH --output=logs/sbatch_output/slurm-%j.out
#SBATCH -p veu             # Partition to submit to
#SBATCH -c1
#SBATCH --mem=128G      # Max CPU Memory
#SBATCH --gres=gpu:4
#SBATCH --job-name=train_pg_tmh_ap_fc
python scripts/train.py \
	--train_labels_path './labels/train/voxceleb_2/22_12_09_15_41_00_25vauqev_cool-bee-37/train_labels.ndx' \
	--train_data_dir './datasets/voxceleb_2/dev/22_12_05_21_59_21_f0yycx91_azure-pine-7/' \
	--valid_clients_path './labels/valid/voxceleb_2/22_12_09_12_41_06_12nc5wq4_fiery-donkey-13/clients.ndx' \
	--valid_impostors_path './labels/valid/voxceleb_2/22_12_09_12_41_29_ikfavyhj_ruby-microwave-14/impostors.ndx' \
	--valid_data_dir './datasets/voxceleb_2/dev/22_12_05_21_59_21_f0yycx91_azure-pine-7/' \
	--model_output_folder './models/' \
	--max_epochs 100 \
	--batch_size 64 \
	--eval_and_save_best_model_every 8000 \
	--print_training_info_every 100 \
	--early_stopping 35 \
	--update_optimizer_every 0 \
	--normalization 'cmn' \
	--model_name_prefix 'pg_tmh_ap_fc' \
	--embedding_size 400 \
	--front_end 'PatchsGenerator' \
	--patchs_generator_patch_width 1 \
	--pooling_method 'TransformerStackedAttentionPooling' \
	--pooling_output_size 40 \
	--pooling_heads_number 8 \
	--pooling_positional_encoding \
	--transformer_n_blocks 2 \
	--transformer_expansion_coef 2 \
	--transformer_attention_type 'MultiHeadAttention' \
	--transformer_drop_out 0.0 \
	--bottleneck_drop_out 0.0 \
    --load_checkpoint \
	--checkpoint_file_folder './models/22_12_09_19_02_40_pg_tmh_ap_fc_PatchsGenerator_TransformerStackedAttentionPooling_1w22yjpa' \
	--checkpoint_file_name '22_12_09_19_02_40_pg_tmh_ap_fc_PatchsGenerator_TransformerStackedAttentionPooling_1w22yjpa.chkpt' \
	> logs/console_output/train/5_pg_tmh_ap_fc_2.log 2>&1