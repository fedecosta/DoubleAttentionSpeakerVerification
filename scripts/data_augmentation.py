import os
import logging

_LOGGER = logging.getLogger(__name__)

class DataAugmentator:
    
    EFFECTS = ["apply_reverb", "add_background_noise"]
    SPEEDS = ["0.9", "1.", "1.1"]
    SNR_NOISE_RANGE = [0, 15]
    SNR_SPEECH_RANGE = [10, 30]
    SNR_MUSIC_RANGE = [5, 15]

    def __init__(
        self,
        augmentation_directory,
        augmentation_labels_path,
        rirs_directory,
        rirs_labels_path,
    ):
        
        self.augmentation_directory = augmentation_directory
        self.augmentation_labels_path = augmentation_labels_path
        self.rirs_directory = rirs_directory
        self.rirs_labels_path = rirs_labels_path
        self.create_augmentation_list(augmentation_labels_path)
        self.create_rir_list(rirs_labels_path)
        
    
    def create_augmentation_list(self, augmentation_labels_path):
        
        with open(augmentation_labels_path) as handle:
            self.augmentation_list = handle.readlines()
        
        self.augmentation_list = [os.path.join(dir, file_path) for file_path in self.augmentation_list]

    
    def create_rir_list(self, rirs_labels_path):
        
        with open(rirs_labels_path) as handle:
            self.rirs_list = handle.readlines()
        
        self.rirs_list = [os.path.join(dir, file_path) for file_path in self.rirs_list]
            
    
    def get_SNR_bounds(self, background_audio_type):
        if background_audio_type == "noise":
            return self.SNR_NOISE_RANGE
        elif background_audio_type == "speech":
            return self.SNR_SPEECH_RANGE
        elif background_audio_type == "music":
            return self.SNR_MUSIC_RANGE
        else:
            return self.SNR_NOISE_RANGE

    def sample_random_SNR(self, background_audio_type):
        snr_bounds = self.get_SNR_bounds(background_audio_type)
        return random.uniform(snr_bounds[0], snr_bounds[1])
        
    
    def add_background_noise(self, audio, sample_rate):
        
        background_audio_line = random.choice(self.augmentation_list).strip()
        try:
            background_audio_name = background_audio_line.split(" ")[0]
            background_audio_type = background_audio_line.split(" ")[1]
        except:
            background_audio_type = "noise"
        
        
        noise, noise_sample_rate = torchaudio.load(
            self.augmentation_directory + "/" + background_audio_name# + ".wav"
        )
        
        if noise.shape[1] > audio.shape[1]:
            # mejor random slice
            noise = noise[:, :audio.shape[1]]
        else:
            # repetir el sonido durante la duracion del audio
            repeat_times = math.ceil(audio.shape[1] / noise.shape[1])
            noise = noise.repeat(1, repeat_times)
            noise = noise[:, :audio.shape[1]]
        
        audio_SNR = torch.tensor(
            self.sample_random_SNR(background_audio_type)
        ).unsqueeze(0)
        
        noisy_audio = F.add_noise(audio, noise, audio_SNR)
        
        return noisy_audio

    
    
    def apply_reverb(self, audio, sample_rate):
        
        rir_wav, rir_sample_rate = torchaudio.load(
            self.rirs_directory + "/" + random.choice(self.rirs_list).strip()# + ".wav"
        )

        rir_wav = rir_wav[:, int(rir_sample_rate * 0.01) : int(rir_sample_rate * 1.3)]
        rir_wav = rir_wav / torch.norm(rir_wav, p=2)
        #rir_wav = torch.flip(rir_wav, [1])

        augmented_audio_wav = F.fftconvolve(audio, rir_wav)        
        
        return augmented_audio_wav
    
    
    def apply_speed_perturbation(self, audio, sample_rate):
        
        speed = random.choice(self.SPEEDS)
        
        print(f"Speed perturbation: {speed}")
        
        effects = [
            ["speed", speed],
            ["rate", f"{sample_rate}"],
        ]
        
        return torchaudio.sox_effects.apply_effects_tensor(
            audio, sample_rate, effects
        )[0]
    
    
    def augment(self, audio, sample_rate):
        
        audio = self.apply_speed_perturbation(audio, sample_rate)
        effect = random.choice(self.EFFECTS)
        
        print(f"Effect augmentation: {effect}")
        
        
        return getattr(self, effect)(audio, sample_rate)