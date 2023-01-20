#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu							# Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G      					# Max CPU Memory
#SBATCH --job-name=labels_versioner
python scripts/labels_versioner.py \
	--labels_file_folder './labels/augmentation_experiments/train/voxceleb_1_augmented_4/' \
	--labels_file_name 'train_labels.ndx' \
	--prepend_directory '/home/usuaris/veussd/DATABASES/Common_Voice/cv11.0/dev_data_augmentation2_full/' \
	--dump_folder_name './labels/augmentation_experiments/train/voxceleb_1_augmented_4/' \
	--log_file_folder './logs/labels_versioner/' \
	--no-get_duration \
	> logs/console_output/labels_versioner/console_output.log 2>&1