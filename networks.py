import torch
import torch.nn as nn
import math
import random
from components import Layer

def get_network(_name):
    return {
        'color': ColorNet,
        'seg': SegNet
    }[_name]

class ColorNet(nn.Module):
    def __init__(self, nInputs, nOutputs):
        super(ColorNet, self).__init__()
        
        print('Init Network')
        _dims = [nInputs, 32, 64, 128, 256, 512, 512]

        ## Encoder 
        for i in range(6):
            setattr(self, 'enc{}'.format(i+1), Layer(_dims[i], _dims[i+1], kernel_size=3,stride=1,padding=1,sampling=None if i == 0 else 'down'))
            setattr(self, 'ins{}'.format(i+1), nn.InstanceNorm2d(num_features=_dims[i+1]))

        ## Decoder
        _dims.reverse()
        _dims = [k*2 for k in _dims[:-1]]

        for i in range(5):
            setattr(self, 'dec{}'.format(i+1), Layer(_dims[i], _dims[i+1], kernel_size=3,stride=1,padding=1,sampling='up'))

        self.final_conv = nn.Conv2d(_dims[-1],nOutputs,3,1,1,bias=False)
        self.final_relu = nn.LeakyReLU(0.1,True)

    def forward(self, I, R):
        sources = []

        ## Encoder
        for i in range(6):
            I = getattr(self, 'enc{:d}'.format(i + 1))(I)
            R = getattr(self, 'enc{:d}'.format(i + 1))(R)
            sources.append(torch.cat([getattr(self, 'ins{:d}'.format(i + 1))(I), R], 1))

        ## Decoder
        X = sources.pop()
        for i in range(5):
            X = getattr(self, 'dec{:d}'.format(i + 1))(X, sources[-i-1])

        return self.final_relu(self.final_conv(X))

"""
    This (GridNet) network is borrowed from https://github.com/Hv0nnus/GridNet/blob/master/Python_Files/GridNet_structure.py
"""

class firstConv(nn.Module):
    def __init__(self, nInputs, nOutputs):
        super(firstConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=nInputs, out_channels=nOutputs,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               dilation=1,
                               groups=1,
                               bias=False)

        self.batch1 = nn.BatchNorm2d(num_features=nOutputs,
                                     eps=1e-05,
                                     momentum=0.1,
                                     affine=True)

        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=nOutputs, out_channels=nOutputs,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               dilation=1,
                               groups=1,
                               bias=False)
        
        self.batch2 = nn.BatchNorm2d(num_features=nOutputs,
                                     eps=1e-05,
                                     momentum=0.1,
                                     affine=True)

        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)

        return x

class convSequence(nn.Module):
    def __init__(self, nInputs, nOutputs, dropFactor):
        super(convSequence, self).__init__()

        self.dropFactor = dropFactor

        self.batch1 = nn.BatchNorm2d(num_features=nInputs, eps=1e-05, momentum=0.1, affine=True)

        self.relu1 = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=nInputs, out_channels=nOutputs,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               dilation=1,
                               groups=1,
                               bias=False)

        self.batch2 = nn.BatchNorm2d(num_features=nOutputs,
                                     eps=1e-05,
                                     momentum=0.1,
                                     affine=True)

        self.relu2 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=nOutputs, out_channels=nOutputs,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               dilation=1,
                               groups=1,
                               bias=False)

    def forward(self, x_init):
        x = self.batch1(x_init)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        # *1 is a small trick that transform boolean into integer
        x = ((random.random() > self.dropFactor) * 1) * x
        x += x_init
        return x

class subSamplingSequence(nn.Module):
    def __init__(self, nInputs, nOutputs):
        """
        :param nInputs: number of features map for the input
        :param nOutputs: number of features map for the output
        This class represente a bloc that reduce the resolution of each feature map (factor2)
        """
        super(subSamplingSequence, self).__init__()

        self.batch1 = nn.BatchNorm2d(num_features=nInputs,
                                     eps=1e-05,
                                     momentum=0.1,
                                     affine=True)

        self.relu1 = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=nInputs, out_channels=nOutputs,
                               kernel_size=(3, 3),
                               stride=(2, 2),
                               padding=(1, 1),
                               dilation=1,
                               groups=1,
                               bias=False)

        self.batch2 = nn.BatchNorm2d(num_features=nOutputs,
                                     eps=1e-05,
                                     momentum=0.1,
                                     affine=True)

        self.relu2 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=nOutputs, out_channels=nOutputs,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               dilation=1,
                               groups=1,
                               bias=False)

    def forward(self, x):
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        return x


class upSamplingSequence(nn.Module):
    def __init__(self, nInputs, nOutputs):
        """
        :param nInputs: number of features map for the input
        :param nOutputs: number of features map for the output
        This class represente a bloc that increase the resolution of each feature map(factor2)
        """

        super(upSamplingSequence, self).__init__()

        self.batch1 = nn.BatchNorm2d(num_features=nInputs,
                                     eps=1e-05,
                                     momentum=0.1,
                                     affine=True)

        self.relu1 = nn.ReLU()

        # assume Hin = 25
        # Hout = (Hin-1)*stride[0]-2*pad[0]+kernel[0]+output_pad[0]
        #      = (25-1)*2-2*1+3
        #      = 49 -> wrong
        self.convTranspose1 = nn.ConvTranspose2d(in_channels=nInputs, out_channels=nOutputs,
                                                 kernel_size=(3, 3),
                                                 stride=(2, 2),
                                                 padding=(1, 1),
                                                 dilation=1,
                                                 groups=1,
                                                 output_padding=1,
                                                 bias=False)

        self.batch2 = nn.BatchNorm2d(num_features=nOutputs,
                                     eps=1e-05,
                                     momentum=0.1,
                                     affine=True)

        self.relu2 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=nOutputs, out_channels=nOutputs,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               dilation=1,
                               groups=1,
                               bias=False)

    def forward(self, x):
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.convTranspose1(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        return x


class lastConv(nn.Module):
    def __init__(self, nInputs, nOutputs):
        """
        :param nInputs: number of features map for the input
        :param nOutputs: number of features map for the output
        This class represente the last Convolution of the network before the prediction
        """

        super(lastConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=nInputs, out_channels=nOutputs,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               dilation=1,
                               groups=1,
                               bias=False)

        self.batch1 = nn.BatchNorm2d(num_features=nOutputs,
                                     eps=1e-05,
                                     momentum=0.1,
                                     affine=True)

        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=nOutputs, out_channels=nOutputs,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               dilation=1,
                               groups=1,
                               bias=False)

        self.batch2 = nn.BatchNorm2d(num_features=nOutputs,
                                     eps=1e-05,
                                     momentum=0.1,
                                     affine=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        return x

class SegNet(nn.Module):
    def __init__(self, nInputs, nOutputs, nColumns, nFeatMaps, dropFactor):
        super(SegNet, self).__init__()

        # Define some parameters as an attribut of the class
        len_nfeatureMaps = len(nFeatMaps)
        self.nColumns = nColumns
        self.nFeatMaps = nFeatMaps
        self.len_nfeatureMaps = len_nfeatureMaps

        # A normalisation before any computation
        self.batchNormInitial = nn.BatchNorm2d(num_features=nInputs,
                                               eps=1e-05,
                                               momentum=0.1,
                                               affine=True)

        # The first convolution before entering into the grid.
        self.firstConv = firstConv(nInputs=nInputs, nOutputs=nFeatMaps[0])

        # We create the Grid. We will creat conv and sub/up sequences with different name.
        # The name is : "sequenceName" + starting position of the sequence(i,j) + "to" + ending position (k,l)
        for i in range(len(nFeatMaps)):

            for j in range(nColumns):

                # We don t creat a residual bloc on the last column
                if j < (nColumns - 1):
                    setattr(self, "convSequence" + str(i) + "_" + str(j) + "to" + str(i) + "_" + str(j + 1),
                            convSequence(nFeatMaps[i], nFeatMaps[i], dropFactor))

                # We creat subSampling only on half of the grid and not in the last row
                if j < (nColumns // 2) and i < (len(nFeatMaps) - 1):
                    setattr(self, "subSamplingSequence" + str(i) + "_" + str(j) + "to" + str(i + 1) + "_" + str(j),
                            subSamplingSequence(nFeatMaps[i], nFeatMaps[i + 1]))

                # Welook a the other half but not the first row
                if j >= (nColumns // 2) and i > 0:
                    setattr(self, "upSamplingSequence" + str(i) + "_" + str(j) + "to" + str(i - 1) + "_" + str(j),
                            upSamplingSequence(nFeatMaps[i], nFeatMaps[i - 1]))

        # The last convolution before the result.
        self.lastConv = lastConv(nInputs=nFeatMaps[0],
                                 nOutputs=nOutputs)


    def addTransform(self, X_i_j, SamplingSequence):
        """
        :param X_i_j: The value on the grid a the position (i,j)
        :param SamplingSequence: The sampling that should be added to the point (i,j)
        :return: The fusion of the actual value on (i,j) and the new data which come from the sampling
        """
        return X_i_j + SamplingSequence

    def forward(self, x):

        # A normalisation before any computation
        x = self.batchNormInitial(x)

        # The first convolution before entering into the grid.
        x = self.firstConv(x)

        # X is the matrix that represente the values of the features maps at the point (i,j) in the grid.
        X = [[0 for i in range(self.nColumns)] for j in range(self.len_nfeatureMaps)]
        # The input of the grid is on (0,0)
        X[0][0] = x

        # Looking on half of the grid, with sumsampling and convolution sequence
        for j in range(self.nColumns // 2):

            for i in range(self.len_nfeatureMaps):

                # For the first column, there is only subsampling
                if j > 0:
                    # This syntaxe call self.conSequencei_(j-1)toi_j(X[i][j-1])
                    X[i][j] = getattr(self, "convSequence"
                                      + str(i) + "_" + str(j - 1) + "to" + str(i) + "_" + str(j))(X[i][j - 1])

                    # For the first row, there is only ConvSequence (residual bloc)
                    if i > 0:
                        X[i][j] = self.addTransform(X[i][j], getattr(self, "subSamplingSequence"
                                                                     + str(i - 1) + "_" + str(j) + "to" + str(i) +
                                                                     "_" + str(j))(X[i - 1][j]))
                else:
                    # For the first row, there is only ConvSequence (residual bloc)
                    if i > 0:
                        X[i][j] = getattr(self, "subSamplingSequence"
                                          + str(i - 1) + "_" + str(j) + "to" + str(i) +
                                          "_" + str(j))(X[i - 1][j])

        # Looking on the other half of the grid
        for i in range(self.len_nfeatureMaps - 1, -1, -1):
            for j in range(self.nColumns // 2, self.nColumns):

                X[i][j] = getattr(self, "convSequence" +
                                  str(i) + "_" + str(j - 1) + "to" + str(i) + "_" + str(j))(X[i][j - 1])

                # There is no upSampling on the last row
                if i < (self.len_nfeatureMaps - 1):
                    X[i][j] = self.addTransform(X[i][j], getattr(self, "upSamplingSequence"
                                                                 + str(i + 1) + "_" + str(j) + "to" + str(i) +
                                                                 "_" + str(j))(X[i + 1][j]))

        x_final = self.lastConv(X[0][self.nColumns - 1])

        return x_final