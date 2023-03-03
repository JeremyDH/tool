import torch
import torch.nn as nn
import BackBone
import base_block

class Yolox(nn.Module):
    def __init__(self):
        super(Yolox, self).__init__()