#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu							# Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G      					# Max CPU Memory
#SBATCH --job-name=paths_generator
python scripts/feature_extractor_paths_generator.py \
	'/home/usuaris/veussd/DATABASES/Common_Voice/cv11.0/dev_data_augmentation2_full/' \
	--dump_file_name 'voxceleb_1_dev_augmented_4_feature_extractor_paths.lst' \
	--dump_file_folder './feature_extractor/' \
    > logs/console_output/feature_extractor_paths_generator/console_output.log 2>&1