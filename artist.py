import networks as net
import torch
import numpy as np
import time
import os
def normalize(_tensor):
    return (_tensor - 0.5)/0.5

def unnormalize(_tensor):
    return (_tensor*0.5) + 0.5

class Artist:
    def __init__(self, c_dir, s_dir=None, use_cuda=True):
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.load_ckpts(c_dir, s_dir)

    def load_ckpts(self, c_dir, s_dir):
        self.color = net.get_network('color')(1, 2).to(self.device)
        self.color.load_state_dict(torch.load(c_dir)['G'])
        self.color.eval()

        if s_dir is not None:
            self.seg = net.get_network('seg')(1,60,6,[16, 32, 64, 128, 256], 0.0).to(self.device)
            self.seg.load_state_dict(torch.load(s_dir)['G'])
            self.seg.eval()
        else:
            self.seg = None

    def colorize(self, gray_img, seg_map=None):
        gray_tensor = torch.FloatTensor(gray_img.transpose((2, 0, 1)).astype(np.float32)  / 255.0).unsqueeze(0).to(self.device)

        if seg_map is not None:
            seg_tensor = torch.FloatTensor(seg_map[..., np.newaxis].transpose((2, 0, 1)).astype(np.float32) / 59.0).unsqueeze(0).to(self.device)
        elif self.seg is not None:
            seg_gray_out = self.seg(gray_tensor)
            seg_tensor = torch.unsqueeze(seg_gray_out.data.max(1)[1], 1).float() / 59.0
            print('Semantic Map Detected')
        else:
            seg_tensor = torch.zeros_like(gray_tensor)
            seg_tensor = seg_tensor.to(self.device)

        seg_tensor = normalize(seg_tensor)
        gray_tensor = normalize(gray_tensor)

        color_out = self.color(gray_tensor, seg_tensor)
        color_tensor = torch.cat((torch.unsqueeze(gray_tensor[:, 0, :, :], 1), color_out), 1)

        color_out = unnormalize(color_tensor).cpu()[0,:, :, :].detach().numpy().transpose(1, 2, 0) * 255.0
        seg_out = unnormalize(seg_tensor).cpu()[0,0,:,:].detach().numpy() * 59.0
        print('Colorized')
        return color_out.astype(np.uint8), seg_out.astype(np.uint8)
