#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu							# Partition to submit to
#SBATCH -c1
#SBATCH --mem=32G      					# Max CPU Memory
#SBATCH --job-name=labels_versioner
python scripts/labels_versioner.py \
	--labels_file_folder './labels/augmentation_experiments/valid/voxceleb_1/' \
	--labels_file_name 'impostors.ndx' \
	--prepend_directory '/home/usuaris/veussd/DATABASES/VoxCeleb/VoxCeleb1/dev/' \
	--dump_folder_name './labels/augmentation_experiments/valid/voxceleb_1/' \
	--log_file_folder './logs/labels_versioner/' \
	--get_duration \
	> logs/console_output/labels_versioner/console_output.log 2>&1