#!/bin/bash
#SBATCH --output=logs/sbatch_output/slurm-%j.out
#SBATCH -p veu							# Partition to submit to
#SBATCH -c1
#SBATCH --mem=32G      					# Max CPU Memory
#SBATCH --job-name=labels_versioner
python scripts/labels_versioner.py \
	--labels_file_folder './labels/test/voxceleb1_t_protocol/' \
	--labels_file_name 'clients.ndx' \
	--prepend_directory '/home/usuaris/veussd/DATABASES/VoxCeleb/VoxCeleb1/test/' \
	--dump_folder_name './labels/test/voxceleb1_t_protocol/' \
	--log_file_folder './logs/labels_versioner/'