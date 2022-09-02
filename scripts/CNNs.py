import sys
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

def getVGG3LOutputDimension(inputDimension, outputChannel = 128):

    # Compute the component output's dimension

    # Each convolutional block reduces x and y dimension by /2
    outputDimension = np.ceil(np.array(inputDimension, dtype=np.float32) / 2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32) / 2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32) / 2)

    # TODO check: only x dimension matters? why not x * y * outputChannel?
    outputDimension = int(outputDimension) * outputChannel

    return outputDimension

def getVGG4LOutputDimension(inputDimension, outputChannel=128):

    # Compute the component output's dimension

    # Each convolutional block reduces x and y dimension by /2
    outputDimension = np.ceil(np.array(inputDimension, dtype=np.float32)/2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32)/2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32)/2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32)/2)
    
    outputDimension = int(outputDimension) * outputChannel

    return outputDimension

# TODO make a more general VGG 'N' L class!
class VGG3L(torch.nn.Module):

    def __init__(self, vgg_last_channels):
        super().__init__()

        # The component will have 3 convolutional blocks
        # Each block will have 2 conv layers followed by a max pooling layer
 
        # Block 1 conv layers
        self.conv11 = torch.nn.Conv2d(
            in_channels = 1, # the spectrogram has 1 channel
            out_channels = int(vgg_last_channels / 4), 
            kernel_size = 3, 
            stride = 1, 
            padding = 1,
            )

        # This layer will apply a convolution over volume, and it's ok to be a Conv2d, it's not a Conv3d.
        self.conv12 = torch.nn.Conv2d(
            int(vgg_last_channels/4), 
            int(vgg_last_channels/4), 
            3, 
            stride=1, 
            padding=1,
            )


        self.conv21 = torch.nn.Conv2d(int(vgg_last_channels/4), int(vgg_last_channels/2), 3, stride=1, padding=1)
        self.conv22 = torch.nn.Conv2d(int(vgg_last_channels/2), int(vgg_last_channels/2), 3, stride=1, padding=1)
        self.conv31 = torch.nn.Conv2d(int(vgg_last_channels/2), int(vgg_last_channels), 3, stride=1, padding=1)
        self.conv32 = torch.nn.Conv2d(int(vgg_last_channels), int(vgg_last_channels), 3, stride=1, padding=1)
        
    def forward(self, paddedInputTensor):

        paddedInputTensor =  paddedInputTensor.view( 
            paddedInputTensor.size(0),  
            paddedInputTensor.size(1), 
            1, 
            paddedInputTensor.size(2)
            ).transpose(1, 2)

        encodedTensorLayer1 = F.relu(self.conv11(paddedInputTensor))
        encodedTensorLayer1 = F.relu(self.conv12(encodedTensorLayer1))
        encodedTensorLayer1 = F.max_pool2d(encodedTensorLayer1, 2, stride=2, ceil_mode=True)

        encodedTensorLayer2 = F.relu(self.conv21(encodedTensorLayer1))
        encodedTensorLayer2 = F.relu(self.conv22(encodedTensorLayer2))
        encodedTensorLayer2 = F.max_pool2d(encodedTensorLayer2, 2, stride=2, ceil_mode=True)

        encodedTensorLayer3 = F.relu(self.conv31(encodedTensorLayer2))
        encodedTensorLayer3 = F.relu(self.conv32(encodedTensorLayer3))
        encodedTensorLayer3 = F.max_pool2d(encodedTensorLayer3, 2, stride=2, ceil_mode=True)

        outputTensor = encodedTensorLayer3.transpose(1, 2)

        outputTensor = outputTensor.contiguous().view(
            outputTensor.size(0), 
            outputTensor.size(1), 
            outputTensor.size(2) * outputTensor.size(3)
            )

        return outputTensor

class VGG4L(torch.nn.Module):

    def __init__(self, vgg_last_channels):
        super(VGG4L, self).__init__()

        self.conv11 = torch.nn.Conv2d(1, int(vgg_last_channels/8), 3, stride=1, padding=1)
        self.conv12 = torch.nn.Conv2d(int(vgg_last_channels/8), int(vgg_last_channels/8), 3, stride=1, padding=1)
        self.conv21 = torch.nn.Conv2d(int(vgg_last_channels/8), int(vgg_last_channels/4), 3, stride=1, padding=1)
        self.conv22 = torch.nn.Conv2d(int(vgg_last_channels/4), int(vgg_last_channels/4), 3, stride=1, padding=1)
        self.conv31 = torch.nn.Conv2d(int(vgg_last_channels/4), int(vgg_last_channels/2), 3, stride=1, padding=1)
        self.conv32 = torch.nn.Conv2d(int(vgg_last_channels/2), int(vgg_last_channels/2), 3, stride=1, padding=1)
        self.conv41 = torch.nn.Conv2d(int(vgg_last_channels/2), int(vgg_last_channels), 3, stride=1, padding=1)
        self.conv42 = torch.nn.Conv2d(int(vgg_last_channels), int(vgg_last_channels), 3, stride=1, padding=1)
        
    def forward(self, paddedInputTensor):

        paddedInputTensor =  paddedInputTensor.view( paddedInputTensor.size(0),  paddedInputTensor.size(1), 1, paddedInputTensor.size(2)).transpose(1, 2)

        encodedTensorLayer1 = F.relu(self.conv11(paddedInputTensor))
        encodedTensorLayer1 = F.relu(self.conv12(encodedTensorLayer1))
        encodedTensorLayer1 = F.max_pool2d(encodedTensorLayer1, 2, stride=2, ceil_mode=True)

        encodedTensorLayer2 = F.relu(self.conv21(encodedTensorLayer1))
        encodedTensorLayer2 = F.relu(self.conv22(encodedTensorLayer2))
        encodedTensorLayer2 = F.max_pool2d(encodedTensorLayer2, 2, stride=2, ceil_mode=True)

        encodedTensorLayer3 = F.relu(self.conv31(encodedTensorLayer2))
        encodedTensorLayer3 = F.relu(self.conv32(encodedTensorLayer3)) 
        encodedTensorLayer3 = F.max_pool2d(encodedTensorLayer3, 2, stride=2, ceil_mode=True)

        encodedTensorLayer4 = F.relu(self.conv41(encodedTensorLayer3))
        encodedTensorLayer4 = F.relu(self.conv42(encodedTensorLayer4))
        encodedTensorLayer4 = F.max_pool2d(encodedTensorLayer4, 2, stride=2, ceil_mode=True)

        outputTensor = encodedTensorLayer4.transpose(1, 2)
        outputTensor = outputTensor.contiguous().view(outputTensor.size(0), outputTensor.size(1), outputTensor.size(2) * outputTensor.size(3))

        return outputTensor


class VGGNL(torch.nn.Module):

    def __init__(self, vgg_n_blocks, vgg_last_channels):
        super().__init__()

        self.vgg_n_blocks = vgg_n_blocks
        self.vgg_last_channels = vgg_last_channels
        self.generate_conv_blocks(n_blocks = self.vgg_n_blocks, vgg_last_channels = self.vgg_last_channels)


    def generate_conv_block(self, block_channels):

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = block_channels, kernel_size = 3, stride = 1, padding = 1,),
            nn.ReLU(),
            nn.Conv2d(in_channels = block_channels, out_channels = block_channels, kernel_size = 3, stride = 1, padding = 1,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride=2, padding=0, ceil_mode=True)
            )

    def generate_conv_blocks(self, n_blocks, vgg_last_channels):
        
        self.conv_blocks = []

        for num_block in range(n_blocks):

            conv_block_channels = vgg_last_channels / (2 ** (n_blocks - num_block ))
            conv_block = self.generate_conv_block(block_channels = conv_block_channels)
            self.conv_blocks.append(conv_block)


    def forward(self, paddedInputTensor):
        
        paddedInputTensor =  paddedInputTensor.view( 
            paddedInputTensor.size(0),  
            paddedInputTensor.size(1), 
            1, 
            paddedInputTensor.size(2)
            ).transpose(1, 2)

        encodedTensor = paddedInputTensor
        for num_block in range(self.vgg_n_blocks):
            encodedTensor = self.conv_blocks[num_block](encodedTensor)

        outputTensor = encodedTensor.transpose(1, 2)

        outputTensor = outputTensor.contiguous().view(
            outputTensor.size(0), 
            outputTensor.size(1), 
            outputTensor.size(2) * outputTensor.size(3)
            )

        return outputTensor