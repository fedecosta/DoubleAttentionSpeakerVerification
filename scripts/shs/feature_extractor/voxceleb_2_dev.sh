#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu							# Partition to submit to
#SBATCH -c4
#SBATCH --mem=128G      					# Max CPU Memory
#SBATCH --gres=gpu:2
#SBATCH --job-name=feature_extractor_voxceleb_2_dev
python scripts/feature_extractor.py \
    --audio_paths_file_folder './feature_extractor/' \
	--audio_paths_file_name 'voxceleb_2_dev_feature_extractor_paths.lst' \
	--prepend_directory '/home/usuaris/veussd/DATABASES/VoxCeleb/VoxCeleb2/dev/' \
	--dump_folder_name '/home/usuaris/veussd/federico.costa/datasets/voxceleb_2/dev/' \
	--sampling_rate 16000 \
	--n_fft_secs 0.025 \
	--win_length_secs 0.025 \
	--hop_length_secs 0.010 \
	--n_mels 80 \
    > logs/console_output/feature_extractor/voxceleb_2_dev.log 2>&1