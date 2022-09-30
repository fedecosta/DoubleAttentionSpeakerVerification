PATHS_GENERATOR_DEFAULT_SETTINGS = {
    'dump_file_name' : 'feature_extractor_paths.lst',
    'dump_file_folder' : './feature_extractor/',
    'valid_audio_formats': ['wav', 'm4a'],
}

FEATURE_EXTRACTOR_DEFAULT_SETTINGS = {
    'audio_paths_file_folder' : './feature_extractor/',
    'audio_paths_file_name' : 'feature_extractor_paths.lst',
    'sampling_rate' : 16000,
    'n_fft_secs': 0.023,
    'window' : 'hamming',
    'win_length_secs' : 0.023,
    'hop_length_secs' : 0.010,
    'pre_emph_coef' : 0.97,
    'n_mels' : 80,
    'overwrite' : True,
    'verbose' : False,
}

LABELS_GENERATOR_DEFAULT_SETTINGS = {
    'train_labels_dump_file_folder' : 'scripts/labels/train/',
    'train_labels_dump_file_name' : 'train_labels.ndx',
    'valid_impostors_labels_dump_file_folder' : 'scripts/labels/valid/',
    'valid_impostors_labels_dump_file_name' : 'valid_impostors_labels.ndx',
    'valid_clients_labels_dump_file_folder' : 'scripts/labels/valid/',
    'valid_clients_labels_dump_file_name' : 'valid_clients_labels.ndx',
    'train_speakers_pctg': 0.7,
    'random_split' : True,
    'clients_lines_max' : None,
    'impostors_lines_max' : None,
}

TRAIN_DEFAULT_SETTINGS = {
    'train_labels_path' : 'scripts/labels/train/train_labels.ndx',
    'train_data_dir' : '/home/usuaris/veussd/DATABASES/VoxCeleb/VoxCeleb2/dev/',
    'valid_clients' : 'scripts/labels/valid/clients.ndx',
    'valid_impostors' : 'scripts/labels/valid/impostors.ndx',
    'valid_data_dir' : '/home/usuaris/veussd/DATABASES/VoxCeleb/VoxCeleb2/dev/',
    'model_output_folder' : 'scripts/models/',
    'max_epochs' : 300,
    'batch_size' : 128,
    'eval_and_save_best_model_every' : 10,
    'print_training_info_every' : 1,
    'early_stopping' : 50,
    'update_optimizer_every' : 0,
    'load_checkpoint' : False,
    'window_size' : 3.5,
    'normalization' : 'cmn',
    'num_workers' : 2,
    'random_slicing' : False,
    'model_name_prefix' : 'cnn_pooling_fc',
    'front_end' : 'VGGNL',
    'vgg_n_blocks' : 4,
    'vgg_channels' : [128, 256, 512, 1024],
    'pooling_method' : 'Attention', # HACK changed default option for testing. Original -> 'DoubleMHA',
    'heads_number' : 32,
    'mask_prob' : 0.3,
    'embedding_size' : 400,
    'scaling_factor' : 30.0,
    'margin_factor' : 0.4,
    'optimizer' : 'adam',
    'learning_rate' : 0.0001,
    'weight_decay' : 0.001,
    'annealing' : False,
    'verbose' : False,
}

EVALUATE_MODEL_DEFAULT_SETTINGS = {
    'verbose' : False,
}