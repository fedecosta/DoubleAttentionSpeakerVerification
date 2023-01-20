#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH -c1
#SBATCH --mem=128G      # Max CPU Memory
#SBATCH --gres=gpu:4
#SBATCH --job-name=train_vgg_sa_ap_fc
python scripts/train.py \
	--train_labels_path './labels/train/voxceleb_2/22_12_26_00_36_23_39bjucvr_gallant-dragon-44/train_labels.ndx' \
	--train_data_dir './datasets/voxceleb_2/dev/22_12_05_21_59_21_f0yycx91_azure-pine-7/' \
	--valid_clients_path './labels/valid/voxceleb_2/22_12_26_12_41_44_473354ug_dandy-darkness-45/clients.ndx' \
	--valid_impostors_path './labels/valid/voxceleb_2/22_12_26_12_43_22_3jijl4ik_celestial-flower-46/impostors.ndx' \
	--valid_data_dir './datasets/voxceleb_2/dev/22_12_05_21_59_21_f0yycx91_azure-pine-7/' \
	--model_output_folder './models/' \
	--max_epochs 200 \
	--batch_size 64 \
	--eval_and_save_best_model_every 8000 \
	--print_training_info_every 100 \
	--early_stopping 0 \
	--update_optimizer_every 0 \
	--normalization 'cmn' \
	--model_name_prefix 'vgg_sa_ap_fc' \
	--embedding_size 400 \
	--front_end 'VGGNL' \
	--pooling_method 'SelfAttentionAttentionPooling' \
	--pooling_output_size 400 \
	--pooling_heads_number 0 \
	--no-pooling_positional_encoding \
	--bottleneck_drop_out 0.0 \
	> logs/console_output/train/1_vgg_sa_ap_fc.log 2>&1