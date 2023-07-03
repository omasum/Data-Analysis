import os

import cv2
import numpy as np
import torch
import math

def fft(img):
    # fourier transform
    f_img = torch.fft.fft2(img) # complex64
    # shift
    fshift_img = torch.fft.fftshift(f_img)
    t_magnitude = torch.abs(fshift_img)
    # t_magnitude = 20*torch.log(torch.abs(fshift_img))
    t_magnitude = torch.log(1 + t_magnitude)
    t_magnitude = t_magnitude/t_magnitude.max()*255.0
    t_phase = torch.angle(fshift_img) #-pi~pi
    t_phase = (t_phase/math.pi +1)/2*255.0
    t_phase = torch.clip(t_phase, min=0.0, max=255.0)
    t_magnitude = torch.clip(t_magnitude, min=0.0, max=255.0)
    return fshift_img, t_magnitude, t_phase

def dfft(m_img):
    ishift = torch.fft.ifftshift(m_img)
    ifft = torch.fft.ifft2(ishift)
    iimg = torch.abs(ifft)
    iimg = torch.clip(iimg, min=0.0, max=255.0)
    return iimg

# path = r"D:\cjm\dataset\testBlended"
path = r"ParallelRenderTestblended/testBlended"
result = r"DataAnalysis/fourier_parallel"

for file in os.listdir(path):
    oriimg = cv2.imread(os.path.join(path, file)) #[h,w,c]
    t1_oriimg = torch.tensor(oriimg.transpose(2, 0, 1)) #[c,h,w]
    t_oriimg = 1/3*t1_oriimg[0,:,:] + 1/3*t1_oriimg[1,:,:] + 1/3*t1_oriimg[2,:,:] #[h,w]
    imgconsole = torch.split(t_oriimg, split_size_or_sections=int(t_oriimg.shape[1]/5), dim=1)

    line2 = []
    line3 = []
    # line4 = []
    for i in range(0,5):
        img = imgconsole[i]
        f_img, magnitude, phase = fft(img)
        # r_img = dfft(f_img)
        line2.append(magnitude)
        line3.append(phase)
        # line4.append(r_img)
    concate2 = torch.cat(line2, dim=1)
    concate3 = torch.cat(line3, dim=1)
    # concate4 = torch.cat(line4, dim=1)

    concate = torch.cat([t_oriimg, concate2, concate3], dim=0).unsqueeze(dim=0).repeat(3,1,1)
    concate = torch.cat([t1_oriimg, concate],dim=1)
    concate = concate.numpy().transpose(1,2,0).astype(np.uint8)
    cv2.imwrite(os.path.join(result, file), concate)
