PATHS_GENERATOR_DEFAULT_SETTINGS = {
    'dump_file_name' : 'feature_extractor_paths.lst',
    'dump_file_folder' : './feature_extractor/',
    'dump_max_lines' : -1,
    'valid_audio_formats': ['wav', 'm4a'],
    'verbose' : False,
}

FEATURE_EXTRACTOR_DEFAULT_SETTINGS = {
    'audio_paths_file_folder' : './feature_extractor/',
    'audio_paths_file_name' : 'feature_extractor_paths.lst',
    'prepend_directory' : '',
    'dump_folder_name' : './datasets/v0/',
    'log_file_folder' : './logs/feature_extractor/',
    'sampling_rate' : 16000,
    'n_fft_secs': 0.032,
    'window' : 'hamming',
    'win_length_secs' : 0.025,
    'hop_length_secs' : 0.010,
    'pre_emph_coef' : 0.97,
    'n_mels' : 80,
    'overwrite' : True,
    'verbose' : False,
}

LABELS_GENERATOR_DEFAULT_SETTINGS = {
    'train_sc_labels_dump_file_folder' : './labels/train/',
    'train_sc_labels_dump_file_name' : 'sc_labels.ndx',
    'valid_sc_labels_dump_file_folder' : './labels/valid/',
    'valid_sc_labels_dump_file_name' : 'sc_labels.ndx',
    'valid_sv_impostors_labels_dump_file_folder' : './labels/valid/',
    'valid_sv_impostors_labels_dump_file_name' : 'sv_impostors.ndx',
    'valid_sv_clients_labels_dump_file_folder' : './labels/valid/',
    'valid_sv_clients_labels_dump_file_name' : 'sv_clients.ndx',
    'train_speakers_pctg': 0.98,
    'random_split' : True,
    'train_sc_lines_max' : -1,
    'valid_sc_lines_max' : -1,
    'valid_sv_clients_lines_max' : 20000,
    'valid_sv_impostors_lines_max' : 20000,
    'sv_hard_pairs' : False,
    'verbose' : False,
}

TRAIN_DEFAULT_SETTINGS = {
    'train_labels_path' : './labels/train/train_labels.ndx',
    'train_data_dir' : './datasets/v0',
    'valid_clients_path' : './labels/valid/clients.ndx',
    'valid_impostors_path' : './labels/valid/impostors.ndx',
    'valid_data_dir' : './datasets/v0',
    'model_output_folder' : './models/',
    'log_file_folder' : './logs/train/',
    'max_epochs' : 100,
    'batch_size' : 128,
    'eval_and_save_best_model_every' : 10000,
    'print_training_info_every' : 5000,
    'early_stopping' : 25,
    'update_optimizer_every' : 0,
    'load_checkpoint' : False,
    'checkpoint_file_folder' : './models/',
    'evaluation_type' : 'total_length',
    'evaluation_batch_size' : 256,
    'n_mels' : 80,
    'random_crop_secs' : 3.5,
    'normalization' : 'cmn',
    'num_workers' : 2,
    'model_name_prefix' : 'cnn_pooling_fc',
    'front_end' : 'VGGNL',
    'vgg_n_blocks' : 4,
    'vgg_channels' : [128, 256, 512, 1024],
    'pooling_method' : 'SelfAttentionAttentionPooling',
    'pooling_output_size' : 400,
    'bottleneck_drop_out' : 0.0,
    'embedding_size' : 400,
    'scaling_factor' : 30.0,
    'margin_factor' : 0.4,
    'optimizer' : 'adam',
    'learning_rate' : 0.0001,
    'weight_decay' : 0.001,
    'verbose' : False,
}

MODEL_EVALUATOR_DEFAULT_SETTINGS = {
    'data_dir' : [''],
    'dump_folder' : './models_results',
    'log_file_folder' : './logs/model_evaluator/',
    'log_file_name' : 'model_evaluator.log',
    'normalization' : 'cmn',
    'evaluation_type' : "total_length",
    'batch_size' : 64,
    'random_crop_secs' : 3.5,
}