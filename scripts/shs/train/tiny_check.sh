#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH -c1
#SBATCH --mem=32G      # Max CPU Memory
#SBATCH --gres=gpu:2
#SBATCH --job-name=tiny_check
python scripts/train.py \
	--train_labels_path './labels/train/tiny_check/train_labels.ndx' \
	--train_data_dir './datasets/tiny/dev/22_12_02_21_20_39_3gq59pny_iconic-totem-7' \
	--valid_clients './labels/valid/tiny_check/clients.ndx' \
	--valid_data_dir './datasets/tiny/dev/22_12_02_21_20_39_3gq59pny_iconic-totem-7' \
	--valid_impostors './labels/valid/tiny_check/impostors.ndx' \
	--model_output_folder './models/' \
	--max_epochs 20 \
	--batch_size 128 \
	--eval_and_save_best_model_every 20 \
	--print_training_info_every 1 \
	--pooling_method 'TransformerStackedAttentionPooling' \
	--pooling_output_size 400 \
	--pooling_heads_number 0 \
	--no-pooling_positional_encoding \
	--transformer_n_blocks 4 \
	--transformer_expansion_coef 2 \
	--transformer_attention_type 'SelfAttention' \
	--transformer_drop_out 0.0 \
	--bottleneck_drop_out 0.0 \
	> logs/console_output/train/test.log 2>&1