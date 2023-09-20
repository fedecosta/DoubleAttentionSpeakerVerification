#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu							# Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G      					# Max CPU Memory
#SBATCH --job-name=labels_generator
python scripts/labels_generator.py \
	'/home/usuaris/veussd/DATABASES/VoxCeleb/VoxCeleb1/dev/' \
	--train_sc_labels_dump_file_folder './labels/train/voxceleb_1/' \
	--train_sc_labels_dump_file_name 'sc_labels_hard_reduced.ndx' \
	--valid_sc_labels_dump_file_folder './labels/valid/voxceleb_1/' \
	--valid_sc_labels_dump_file_name 'sc_labels_hard_reduced.ndx' \
	--valid_sv_impostors_labels_dump_file_folder './labels/valid/voxceleb_1/' \
	--valid_sv_impostors_labels_dump_file_name 'sv_impostors_hard_reduced.ndx' \
	--valid_sv_clients_labels_dump_file_folder './labels/valid/voxceleb_1/' \
	--valid_sv_clients_labels_dump_file_name 'sv_clients_hard_reduced.ndx' \
	--train_speakers_pctg 0.995 \
	--random_split \
	--train_sc_lines_max -1 \
	--valid_sc_lines_max -1 \
	--valid_sv_clients_lines_max 200 \
	--valid_sv_impostors_lines_max 200 \
	> logs/console_output/labels_generator/console_output_hard_reduced.log 2>&1