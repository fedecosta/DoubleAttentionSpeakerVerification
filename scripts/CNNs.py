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
    def get_vgg_output_dimension(self, input_dimension):

        # TODO check: Compute the front-end component output's dimension ?

        # Each convolutional block reduces x and y dimension by /2
        output_dimension = input_dimension
        for num_block in range(self.vgg_n_blocks):
            output_dimension = np.ceil(np.array(output_dimension, dtype = np.float32) / 2)

        # TODO check: only x dimension matters? why not x * y * outputChannel?
        output_dimension = int(output_dimension) * self.vgg_end_channels

        return output_dimension
    

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
        
        self.conv_blocks = nn.ModuleList() # A Python list will fail with torch
        
        start_block_channels = 1 # The first block starts with the input spectrogram, which has 1 channel
        end_block_channels = vgg_channels[0] # The first block ends with vgg_channels[0] channels

        for num_block in range(1, vgg_n_blocks + 1):
            
            conv_block = self.generate_conv_block(
                start_block_channels = start_block_channels, 
                end_block_channels = end_block_channels,
                )
            self.conv_blocks.append(conv_block)

            
            # Update start_block_channels and end_block_channels for the next block
            if num_block < vgg_n_blocks: # If num_block = vgg_n_blocks, start_block_channels and end_block_channels must not get updated
                start_block_channels = end_block_channels # The next block will start with end_block_channels channels
                end_block_channels = vgg_channels[num_block] 
        
        # VGG ends with the end_block_channels of the last block
        self.vgg_end_channels = end_block_channels


    def forward(self, padded_input_tensor):

        # TODO why is this?
        padded_input_tensor =  padded_input_tensor.view( 
            padded_input_tensor.size(0),  
            padded_input_tensor.size(1), 
            1, 
            padded_input_tensor.size(2)
            ).transpose(1, 2)

        encoded_tensor = padded_input_tensor

        # Loop over the convolutional blocks
        for num_block in range(self.vgg_n_blocks):
            encoded_tensor = self.conv_blocks[num_block](encoded_tensor)

        # TODO why is this?
        output_tensor = encoded_tensor.transpose(1, 2)

        # TODO why is this?
        output_tensor = output_tensor.contiguous().view(
            output_tensor.size(0), 
            output_tensor.size(1), 
            output_tensor.size(2) * output_tensor.size(3)
            )

        return output_tensor