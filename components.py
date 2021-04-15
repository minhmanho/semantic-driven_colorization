import torch.nn as nn

def get_convsampler(_name, nInputs, nOutputs):
    return {
        'down': nn.Conv2d(nInputs, nOutputs, kernel_size=3, stride=2, padding=1, bias=False),
        'up': nn.ConvTranspose2d(nInputs, nOutputs, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
    }[_name]

class Layer(nn.Module):
    def __init__(self, nInputs, nOutputs, kernel_size=3, stride=1, padding=1, sampling=None):
        super(Layer, self).__init__()
        self.sampling = False if sampling is None else True
        if self.sampling:
            self.conv1 = get_convsampler(sampling, nInputs, nOutputs)
        else:
            self.conv1 = nn.Conv2d(nInputs, nOutputs, kernel_size, stride, padding, bias=False)
        self.conv2 = nn.Conv2d(nOutputs , nOutputs, kernel_size, stride, padding, bias=False)
        self.relu = nn.LeakyReLU(0.1,True)

    def forward(self, _input, skip_feat=None):
        X = self.relu(self.conv1(_input))
        if skip_feat is not None:
            X = X + skip_feat
        X = self.relu(self.conv2(X))
        return X
