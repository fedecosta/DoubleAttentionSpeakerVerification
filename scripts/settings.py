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