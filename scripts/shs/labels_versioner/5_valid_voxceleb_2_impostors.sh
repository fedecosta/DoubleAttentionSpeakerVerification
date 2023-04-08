#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu							# Partition to submit to
#SBATCH -c1
#SBATCH --mem=32G      					# Max CPU Memory
#SBATCH --gres=gpu:2
#SBATCH --job-name=labels_versioner_5
python scripts/labels_versioner.py \
	--labels_file_folder './labels/valid/voxceleb_2/' \
	--labels_file_name 'impostors.ndx' \
	--prepend_directory '/home/usuaris/veussd/DATABASES/VoxCeleb/VoxCeleb2/dev/' \
	--dump_folder_name './labels/valid/voxceleb_2/' \
	--log_file_folder './logs/labels_versioner/' \
	> logs/console_output/labels_versioner/5_console_output.log 2>&1