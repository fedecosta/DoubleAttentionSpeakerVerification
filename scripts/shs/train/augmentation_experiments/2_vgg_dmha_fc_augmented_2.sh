#!/bin/bash
#SBATCH -o /home/usuaris/veussd/federico.costa/logs/sbatch/outputs/slurm-%j.out
#SBATCH -e /home/usuaris/veussd/federico.costa/logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G      # Max CPU Memory
#SBATCH --gres=gpu:4
#SBATCH --job-name=train_vgg_dmha_fc_aug
python scripts/train.py \
	--train_labels_path './labels/augmentation_experiments/train/voxceleb_1_augmented_2/23_01_17_15_12_30_7rtzoqej_happy-capybara-61/train_labels.ndx' \
	--train_data_dir '/home/usuaris/veussd/DATABASES/VoxCeleb/VoxCeleb1/feature_extraction/data_augmentation_experiments/voxceleb_1_augmented_2/train/23_01_17_12_41_14_28cmd1bu_efficient-firebrand-15/' \
	--valid_clients_path './labels/augmentation_experiments/valid/voxceleb_1/23_01_12_13_37_17_13monml4_lively-glade-54/clients.ndx' \
	--valid_impostors_path './labels/augmentation_experiments/valid/voxceleb_1/23_01_12_13_47_40_1pt2lxn9_trim-blaze-55/impostors.ndx' \
	--valid_data_dir '/home/usuaris/veussd/federico.costa/datasets/voxceleb_1/dev/23_01_19_10_54_31_hifkm2ss_soft-mountain-21/' \
	--model_output_folder '/home/usuaris/veussd/federico.costa/models/' \
	--max_epochs 20 \
	--batch_size 128 \
	--eval_and_save_best_model_every 1800 \
	--print_training_info_every 100 \
	--early_stopping 0 \
	--update_optimizer_every 0 \
	--normalization 'cmn' \
	--model_name_prefix 'vgg_dmha_fc_aug' \
	--embedding_size 400 \
	--front_end 'VGGNL' \
	--pooling_method 'DoubleMHA' \
	--pooling_heads_number 32 \
	--pooling_mask_prob 0.3 \
	--num_workers 2 \
	--random_crop_secs 3.0 \
	--log_file_folder '/home/usuaris/veussd/federico.costa/logs/train/' \
	> /home/usuaris/veussd/federico.costa/logs/console_output/train/augmentation_experiments/2_vgg_dmha_fc_augmented_20_4gpu.log 2>&1