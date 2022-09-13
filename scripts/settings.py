PATHS_GENERATOR_DEFAULT_SETTINGS = {
    'dump_file_name' : 'feature_extractor_paths.lst',
    'dump_file_folder' : 'scripts/feature_extractor/',
    'valid_audio_formats': ['wav', 'm4a'],
}

FEATURE_EXTRACTOR_DEFAULT_SETTINGS = {
    'audio_paths_file_folder' : 'scripts/feature_extractor/',
    'audio_paths_file_name' : 'feature_extractor_paths.lst',
    'sampling_rate' : 22500,
    'n_fft_secs': 0.023,
    'window' : 'hamming',
    'win_length_secs' : 0.023,
    'hop_length_secs' : 0.010,
    'pre_emph_coef' : 0.97,
    'n_mels' : 80,
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
    'train_data_dir' : '', #'/home/usuaris/scratch/speaker_databases/',
    'valid_clients' : 'scripts/labels/valid/valid_clients_labels.ndx',
    'valid_impostors' : 'scripts/labels/valid/valid_impostors_labels.ndx',
    'valid_data_dir' : '', #'/home/usuaris/scratch/speaker_databases/',
    'model_output_folder' : 'scripts/models/',

    'max_epochs' : 3,
    'batch_size' : 64,
    

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