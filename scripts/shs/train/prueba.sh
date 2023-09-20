#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G      # Max CPU Memory
#SBATCH --gres=gpu:2
#SBATCH --job-name=train_vgg_dmha_fc
python scripts/train.py \
	--train_labels_path './labels/train/voxceleb_1/sc_labels_hard_reduced.ndx' \
	--train_data_dir '/home/usuaris/veussd/DATABASES/VoxCeleb/VoxCeleb1/dev/' \
	--valid_clients_path './labels/valid/voxceleb_1/sv_clients_hard_reduced.ndx' \
	--valid_impostors_path './labels/valid/voxceleb_1/sv_impostors_hard_reduced.ndx' \
	--valid_data_dir '/home/usuaris/veussd/DATABASES/VoxCeleb/VoxCeleb1/dev/' \
	--model_output_folder './models/' \
	--max_epochs 2 \
	--batch_size 32 \
	--eval_and_save_best_model_every 2000 \
	--print_training_info_every 100 \
	--normalization "full" \
	> logs/console_output/train/0_vgg_dmha_fc_lr_2.log 2>&1