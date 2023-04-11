#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH -c4
#SBATCH --mem=32G      # Max CPU Memory
#SBATCH --gres=gpu:4
#SBATCH --job-name=tiny_check
python scripts/train.py \
	--train_labels_path './labels/train/tiny_check/22_12_09_12_27_18_3jdam6kt_dry-deluge-9/train_labels.ndx' \
	--train_data_dir '/home/usuaris/veussd/federico.costa/datasets/tiny/dev/22_12_02_21_20_39_3gq59pny_iconic-totem-7' \
	--valid_clients './labels/valid/tiny_check/22_12_09_12_38_14_3v0l6av2_twilight-snowball-11/clients.ndx' \
	--valid_data_dir '/home/usuaris/veussd/federico.costa/datasets/tiny/dev/22_12_02_21_20_39_3gq59pny_iconic-totem-7' \
	--valid_impostors './labels/valid/tiny_check/22_12_09_12_39_23_32pe0eb2_atomic-wave-12/impostors.ndx' \
	--model_output_folder './models/' \
	--max_epochs 10 \
	--batch_size 128 \
	--eval_and_save_best_model_every 170 \
	--print_training_info_every 1 \
	--pooling_method 'TransformerStackedAttentionPooling' \
	--pooling_output_size 400 \
	--pooling_heads_number 0 \
	--no-pooling_positional_encoding \
	--transformer_n_blocks 1 \
	--transformer_expansion_coef 2 \
	--transformer_attention_type 'SelfAttention' \
	--transformer_drop_out 0.0 \
	--bottleneck_drop_out 0.0 \
	--evaluation_type 'random_crop' \
	--evaluation_batch_size 256 \
	> logs/console_output/train/test.log 2>&1