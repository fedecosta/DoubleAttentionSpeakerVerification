#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu							# Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G      					# Max CPU Memory
#SBATCH --job-name=labels_generator
python scripts/labels_generator.py \
	'/home/usuaris/veussd/federico.costa/datasets/tiny/dev/22_12_02_21_20_39_3gq59pny_iconic-totem-7/' \
	--train_labels_dump_file_folder './labels/improvements_tests/train/' \
	--train_labels_dump_file_name 'train_labels.ndx' \
	--valid_impostors_labels_dump_file_folder './labels/improvements_tests/valid/' \
	--valid_impostors_labels_dump_file_name 'impostors.ndx' \
	--valid_clients_labels_dump_file_folder './labels/improvements_tests/valid/' \
	--valid_clients_labels_dump_file_name 'clients.ndx' \
	--clients_lines_max 0 \
	--impostors_lines_max 0 \
	--train_speakers_pctg 1 \
	--hard_validation \
	--metadata_file_path './metadata/vox1_meta_cleaned.csv' \
	> logs/console_output/labels_generator/console_output.log 2>&1