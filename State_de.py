import numpy as np
import sys
import cv2
from utils import batch_psnr, normalize, init_logger_ipol, \
				variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb
import torch

import os
from models import FFDNet
from torch.autograd import Variable
import matplotlib.image as mpimg
from PIL import Image, ImageStat

def relu(x):
        x_ = x.copy()
        x_[x_<0] = 0
        return x_

class State_de():
    def __init__(self, size, move_range, model):
        self.image = np.zeros(size,dtype=np.float32)
        # (1, 3, 70, 70)
        self.move_range = move_range
        self.net = model

    
    
    def reset(self, x):
        self.image = x
        self.raw = x * 255
        self.raw[np.where(self.raw <= 2)] = 3

    def step_el(self, act):
        neutral = 6
        move = act.astype(np.float32)
        moves = (move - neutral) / 20
        moved_image = np.zeros(self.image.shape, dtype=np.float32)
        # de = move[:, 3:, :, :]
        # RGB channel deconvolution
        r = self.image[:, 0, :, :]
        g = self.image[:, 1, :, :]
        b = self.image[:, 2, :, :]
        

        moved_image[:, 0, :, :] = r + (moves[:, 0, :, :]) * r * (1 - r)
        moved_image[:, 1, :, :] = g + (0.1 * moves[:, 1, :, :] + 0.9 * moves[:, 0, :, :]) * g * (1 - g)
        moved_image[:, 2, :, :] = b + (0.1 * moves[:, 2, :, :] + 0.9 * moves[:, 0, :, :]) * b * (1 - b)
        self.image = 0.8 * moved_image + 0.2 * self.image

    def step_co(self, act):
        bgr_t = np.transpose(self.image, (0,2,3,1))
        temp3 = np.zeros(bgr_t.shape, bgr_t.dtype)
        # de = move[:, 3:, :, :]
        # BGR channel deconvolution
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        for i in range(0, 1):
                temp = cv2.cvtColor(bgr_t[i], cv2.COLOR_RGB2HSV)

                v_channel = (temp[...,2]*255).astype(np.uint8)
                v_channel = clahe.apply(v_channel)

                s_channel = (temp[...,1]*255).astype(np.uint8)
                s_channel = clahe.apply(s_channel)


                temp[...,2] = (v_channel/255).astype(np.float32)
                temp[...,1] = (s_channel/255).astype(np.float32)
                # (v_channel).astype(np.float32)
                temp3[i] = cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)
                print(temp3[i])
        bgr3 = np.transpose(temp3, (0,3,1,2))
        self.image = bgr3

    def step_de(self, act_b):
        pix_num = act_b.shape[1]*act_b.shape[2]
        threshold = pix_num
        checker = act_b.sum(1)
        checker = checker.sum(1)
        for i in range(len(checker)):
            # if checker[i] < threshold:
            #     self.image[i] = self.image[i]
            # else:
            sh_im = self.image.shape
            imorig = np.expand_dims(self.image[i], 0)
            imorig_float = imorig * 255
            lowimg = np.expand_dims(self.raw[i], 0)
            if sh_im[2] % 2 == 1:
                expanded_h = True
                imorig = np.concatenate((imorig, \
                                         imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

            if sh_im[3] % 2 == 1:
                expanded_w = True
                imorig = np.concatenate((imorig, \
                                         imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)

            # imorig = normalize(imorig_float)
            imorig = torch.Tensor(imorig)

            # Sets data type according to CPU or GPU modes
            dtype = torch.cuda.FloatTensor

            # noise level map
            nsigma = (imorig_float - lowimg) / lowimg
            nsigma = 0 + (np.max(nsigma)*2 - 0) * (nsigma - np.min(nsigma)) / (np.max(nsigma) - np.min(nsigma))
            nsigma[np.where(nsigma < 0)] = 0
            nsigma = nsigma / 255
            nsigma.astype('int')
            nsigma = nsigma[:, :, ::2, ::2]

            # Test mode
            with torch.no_grad():  # PyTorch v0.4.0
                imorig = Variable(imorig.type(dtype))
                nsigma = Variable(
                    torch.FloatTensor(nsigma).type(dtype))
            # Estimate noise and subtract it to the input image
            im_noise_estim = self.net(imorig, nsigma)
            outim = torch.clamp(imorig - im_noise_estim, 0., 1.)
            self.image[i] = outim.cpu().detach().numpy()