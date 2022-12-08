import sys
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class VGGNL(torch.nn.Module):

    def __init__(self, vgg_n_blocks, vgg_channels):
        super().__init__()

        self.vgg_n_blocks = vgg_n_blocks
        self.vgg_channels = vgg_channels
        self.generate_conv_blocks(
            vgg_n_blocks = self.vgg_n_blocks, 
            vgg_channels = self.vgg_channels,
            )


    # Method used only at model.py
    def get_hidden_states_dimension(self, input_dimension):

        # Compute the front-end hidden state output's dimension
        # The front-end inputs a (frames, freq_bins) spectrogram \
        # and outputs (frames / (2 ^ vgg_n_blocks)) hidden states of size (freq_bins / (2 ^ vgg_n_blocks)) * vgg_end_channels

        # Each convolutional block reduces dimension by /2
        hidden_states_dimension = input_dimension
        for num_block in range(self.vgg_n_blocks):
            hidden_states_dimension = np.ceil(np.array(hidden_states_dimension, dtype = np.float32) / 2)

        hidden_states_dimension = int(hidden_states_dimension) * self.vgg_end_channels

        return hidden_states_dimension
    

    def generate_conv_block(self, start_block_channels, end_block_channels):

        # Create one convolutional block
        
        conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels = start_block_channels, 
                out_channels = end_block_channels, 
                kernel_size = 3, 
                stride = 1, 
                padding = 1,
                ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = end_block_channels, 
                out_channels = end_block_channels, 
                kernel_size = 3, 
                stride = 1, 
                padding = 1,
                ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = 2, 
                stride = 2, 
                padding = 0, 
                ceil_mode = True,
                )
            )

        return conv_block


    def generate_conv_blocks(self, vgg_n_blocks, vgg_channels):

        # Generate a nn list of vgg_n_blocks convolutional blocks
        
        self.conv_blocks = nn.Sequential() # A Python list will fail with torch
        
        start_block_channels = 1 # The first block starts with the input spectrogram, which has 1 channel
        end_block_channels = vgg_channels[0] # The first block ends with vgg_channels[0] channels

        for num_block in range(1, vgg_n_blocks + 1):
            
            conv_block_name = f"convolutional_block_{num_block}"
            conv_block = self.generate_conv_block(
                start_block_channels = start_block_channels, 
                end_block_channels = end_block_channels,
                )
            
            self.conv_blocks.add_module(conv_block_name, conv_block)

            # Update start_block_channels and end_block_channels for the next block
            if num_block < vgg_n_blocks: # If num_block = vgg_n_blocks, start_block_channels and end_block_channels must not get updated
                start_block_channels = end_block_channels # The next block will start with end_block_channels channels
                end_block_channels = vgg_channels[num_block] 
        
        # VGG ends with the end_block_channels of the last block
        self.vgg_end_channels = end_block_channels


    def forward(self, input_tensor):

        # input_tensor dimensions are:
        # input_tensor.size(0) = number of batches
        # input_tensor.size(1) = number of frames of the spectrogram
        # input_tensor.size(2) = number of frequency bins of the spectrogram

        # We need to add a new dimension corresponding to the channels
        # This channel dimension will be 1 because the spectrogram has only 1 channel
        input_tensor =  input_tensor.view( 
            input_tensor.size(0),  
            input_tensor.size(1), 
            1, 
            input_tensor.size(2),
            )
            
        # We need to put the channel dimension first because nn.Conv2d need it that way
        input_tensor = input_tensor.transpose(1, 2)

        # Pass the tensor through the convolutional blocks 
        encoded_tensor = self.conv_blocks(input_tensor)
        
        # We want to flatten the output
        # For each batch, we will have encoded_tensor.size(1) hidden state vectors \
        # of size encoded_tensor.size(2) * encoded_tensor.size(3)
        output_tensor = encoded_tensor.transpose(1, 2)

        output_tensor = output_tensor.contiguous().view(
            output_tensor.size(0), 
            output_tensor.size(1), 
            output_tensor.size(2) * output_tensor.size(3)
            )

        return output_tensor


class PatchsGenerator(torch.nn.Module):

    def __init__(self, patch_width):

        super().__init__()

        self.patch_width = patch_width


    def spectrogram_to_tokens(self, spectrogram, patch_width):

        # We are going to take the spectrogram and generate patch tokens.
        # In this case, patches will be of dimension patch_width x mels.
        # Each token will be flatten to dimension 1 x (patch_width * mels).
        # Note that we might need to do some padding 

        batch_size, frames, bands = spectrogram.size()

        frames_right_pad = (patch_width - (frames % patch_width)) % patch_width
        pad = (0, 0, 0, frames_right_pad)
        tokens = F.pad(spectrogram, pad, "constant", 0)

        batch_size, padded_frames, bands = tokens.size()

        tokens = tokens.view(batch_size, padded_frames // patch_width, bands * patch_width)

        return tokens


    def forward(self, x):

        tokens = self.spectrogram_to_tokens(x, patch_width = self.patch_width)

        return tokens

