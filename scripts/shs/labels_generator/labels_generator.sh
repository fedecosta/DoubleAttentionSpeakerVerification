#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu							# Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G      					# Max CPU Memory
#SBATCH --job-name=labels_generator
python scripts/labels_generator.py \
	'/home/usuaris/veussd/federico.costa/datasets/tiny/dev/22_12_02_21_20_39_3gq59pny_iconic-totem-7/' \
	--train_sc_labels_dump_file_folder './labels/improvements_tests/train/' \
	--train_sc_labels_dump_file_name 'sc_labels.ndx' \
	--valid_sc_labels_dump_file_folder './labels/improvements_tests/valid/' \
	--valid_sc_labels_dump_file_name 'sc_labels.ndx' \
	--valid_sv_impostors_labels_dump_file_folder './labels/improvements_tests/valid/' \
	--valid_sv_impostors_labels_dump_file_name 'sv_impostors.ndx' \
	--valid_sv_clients_labels_dump_file_folder './labels/improvements_tests/valid/' \
	--valid_sv_clients_labels_dump_file_name 'sv_clients.ndx' \
	--train_speakers_pctg 0.1 \
	--no-random_split \
	--train_sc_lines_max -1 \
	--valid_sc_lines_max -1 \
	--valid_sv_clients_lines_max 2 \
	--valid_sv_impostors_lines_max 2 \
	--sv_hard_pairs \
	--sv_reduced_pairs \
	--metadata_file_path './metadata/vox1_meta_cleaned.csv' \
	> logs/console_output/labels_generator/console_output.log 2>&1