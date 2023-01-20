#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu							# Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G      					# Max CPU Memory
#SBATCH --job-name=labels_generator
python scripts/labels_generator.py \
	'./datasets/voxceleb_1_augmented_4/train/23_01_17_10_11_33_1c4va9j2_glorious-flower-14' \
	--train_labels_dump_file_folder './labels/augmentation_experiments/train/voxceleb_1_augmented_4/' \
	--train_labels_dump_file_name 'train_labels.ndx' \
	--valid_impostors_labels_dump_file_folder './labels/augmentation_experiments/valid/voxceleb_1_augmented_4/' \
	--valid_impostors_labels_dump_file_name 'impostors.ndx' \
	--valid_clients_labels_dump_file_folder './labels/augmentation_experiments/valid/voxceleb_1_augmented_4/' \
	--valid_clients_labels_dump_file_name 'clients.ndx' \
	--clients_lines_max 0 \
	--impostors_lines_max 0 \
	--train_speakers_pctg 1 \
	> logs/console_output/labels_generator/console_output_4.log 2>&1