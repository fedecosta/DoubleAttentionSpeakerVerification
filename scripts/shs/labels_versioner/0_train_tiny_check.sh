#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu							# Partition to submit to
#SBATCH -c1
#SBATCH --mem=32G      					# Max CPU Memory
#SBATCH --job-name=labels_versioner
python scripts/labels_versioner.py \
	--labels_file_folder './labels/train/voxceleb_1_augmented' \
	--labels_file_name 'train_labels.ndx' \
	--prepend_directory '/home/usuaris/veussd/DATABASES/Common_Voice/cv11.0/dev_data_augmentation/' \
	--dump_folder_name './labels/train/voxceleb_1_augmented' \
	--log_file_folder './logs/labels_versioner/'