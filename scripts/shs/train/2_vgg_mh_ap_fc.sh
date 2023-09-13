#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G      # Max CPU Memory
#SBATCH --gres=gpu:4
#SBATCH --job-name=train_vgg_mh_ap_fc
python scripts/train.py \
	--train_labels_path './labels/train/voxceleb_2/23_04_22_21_53_30_3es1f3qk_chocolate-blaze-67/sc_labels.ndx' \
	--train_data_dir '/home/usuaris/veussd/federico.costa/datasets/voxceleb_2/dev/22_12_05_21_59_21_f0yycx91_azure-pine-7/' \
	--valid_clients_path './labels/valid/voxceleb_2/23_04_23_10_47_08_3nwi3vvk_decent-sponge-75/sv_clients_hard_reduced.ndx' \
	--valid_impostors_path './labels/valid/voxceleb_2/23_04_23_10_47_41_sk8auwvv_treasured-frost-76/sv_impostors_hard_reduced.ndx' \
	--valid_data_dir '/home/usuaris/veussd/federico.costa/datasets/voxceleb_2/dev/22_12_05_21_59_21_f0yycx91_azure-pine-7/' \
	--model_output_folder './models/' \
	--max_epochs 200 \
	--batch_size 128 \
	--eval_and_save_best_model_every 6000 \
	--print_training_info_every 100 \
	--early_stopping 0 \
	--update_optimizer_every 15 \
	--normalization 'full' \
	--model_name_prefix 'vgg_mh_ap_fc' \
	--embedding_size 400 \
	--front_end 'VGGNL' \
	--pooling_method 'MultiHeadAttentionAttentionPooling' \
	--pooling_output_size 400 \
	--pooling_heads_number 6 \
	--no-pooling_positional_encoding \
	--bottleneck_drop_out 0.0