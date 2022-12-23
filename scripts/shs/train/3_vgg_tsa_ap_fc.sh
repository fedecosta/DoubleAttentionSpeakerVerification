#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH -c1
#SBATCH --mem=64G      # Max CPU Memory
#SBATCH --gres=gpu:4
#SBATCH --job-name=train_vgg_tsa_ap_fc
python scripts/train.py \
	--train_labels_path './labels/train/voxceleb_2/22_12_09_15_41_00_25vauqev_cool-bee-37/train_labels.ndx' \
	--train_data_dir './datasets/voxceleb_2/dev/22_12_05_21_59_21_f0yycx91_azure-pine-7/' \
	--valid_clients_path './labels/valid/voxceleb_2/22_12_09_12_41_06_12nc5wq4_fiery-donkey-13/clients.ndx' \
	--valid_impostors_path './labels/valid/voxceleb_2/22_12_09_12_41_29_ikfavyhj_ruby-microwave-14/impostors.ndx' \
	--valid_data_dir './datasets/voxceleb_2/dev/22_12_05_21_59_21_f0yycx91_azure-pine-7/' \
	--model_output_folder './models/' \
	--max_epochs 100 \
	--batch_size 128 \
	--eval_and_save_best_model_every 6000 \
	--print_training_info_every 100 \
	--early_stopping 35 \
	--update_optimizer_every 0 \
	--normalization 'cmn' \
	--model_name_prefix 'vgg_tsa_ap_fc' \
	--embedding_size 400 \
	--front_end 'VGGNL' \
	--pooling_method 'TransformerStackedAttentionPooling' \
	--pooling_output_size 400 \
	--pooling_heads_number 0 \
	--no-pooling_positional_encoding \
	--transformer_n_blocks 2 \
	--transformer_expansion_coef 2 \
	--transformer_attention_type 'SelfAttention' \
	--transformer_drop_out 0.0 \
	--bottleneck_drop_out 0.0 \
	--load_checkpoint \
	--checkpoint_file_folder './models/22_12_13_15_57_10_vgg_tsa_ap_fc_VGGNL_TransformerStackedAttentionPooling_36s20lkv/' \
	--checkpoint_file_name '22_12_13_15_57_10_vgg_tsa_ap_fc_VGGNL_TransformerStackedAttentionPooling_36s20lkv.chkpt' \
	> logs/console_output/train/3_vgg_tsa_ap_fc.log 2>&1