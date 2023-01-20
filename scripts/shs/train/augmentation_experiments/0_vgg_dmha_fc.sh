#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G      # Max CPU Memory
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_vgg_dmha_fc
python scripts/train.py \
	--train_labels_path './labels/augmentation_experiments/train/voxceleb_1/23_01_12_12_45_49_1kuycq5o_cosmic-plasma-53/train_labels.ndx' \
	--train_data_dir './datasets/voxceleb_1/dev/22_12_05_21_59_21_1tu20n33_dainty-universe-5/' \
	--valid_clients_path './labels/augmentation_experiments/valid/voxceleb_1/23_01_12_13_37_17_13monml4_lively-glade-54/clients.ndx' \
	--valid_impostors_path './labels/augmentation_experiments/valid/voxceleb_1/23_01_12_13_47_40_1pt2lxn9_trim-blaze-55/impostors.ndx' \
	--valid_data_dir './datasets/voxceleb_1/dev/22_12_05_21_59_21_1tu20n33_dainty-universe-5/' \
	--model_output_folder './models/' \
	--max_epochs 40 \
	--batch_size 128 \
	--eval_and_save_best_model_every 800 \
	--print_training_info_every 100 \
	--early_stopping 0 \
	--update_optimizer_every 0 \
	--normalization 'cmn' \
	--model_name_prefix 'vgg_dmha_fc' \
	--embedding_size 400 \
	--front_end 'VGGNL' \
	--pooling_method 'DoubleMHA' \
	--pooling_heads_number 32 \
	--pooling_mask_prob 0.3 \
	--num_workers 2 \
	--random_crop_secs 3.0 \
	> logs/console_output/train/augmentation_experiments/0_vgg_dmha_fc_original_40.log 2>&1