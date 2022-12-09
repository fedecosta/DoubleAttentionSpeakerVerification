#!/bin/bash
#SBATCH --output=logs/sbatch_output/slurm-%j.out
#SBATCH -p veu							# Partition to submit to
#SBATCH -c1
#SBATCH --mem=32G      					# Max CPU Memory
#SBATCH --job-name=labels_versioner
python scripts/labels_versioner.py \
	--labels_file_folder './labels/valid/voxceleb_2/' \
	--labels_file_name 'clients.ndx' \
	--prepend_directory '/home/usuaris/veussd/DATABASES/VoxCeleb/VoxCeleb2/dev/' \
	--dump_folder_name './labels/valid/voxceleb_2/' \
	--log_file_folder './logs/labels_versioner/'