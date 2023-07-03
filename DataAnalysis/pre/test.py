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

path = "DataAnalysis/test_data/0000001;PolishedMarbleFloor_01Xmetal_bumpy_squares;1X1.png"
oriimg = cv2.imread(path) #[h,w,c]
t_oriimg = torch.tensor(oriimg.transpose(2, 0, 1)) #[c,h,w]
imgconsole = torch.split(t_oriimg, split_size_or_sections=int(t_oriimg.shape[2]/5), dim=2)
input = imgconsole[0]
f_img, magnitude, phase = fft(input)

r,g,b = torch.split(input, split_size_or_sections=1, dim=0)
f_imgr, magnituder, phaser = fft(r)
f_imgg, magnitudeg, phaseg = fft(g)
f_imgb, magnitudeb, phaseb = fft(b)

f_img2 = torch.cat([f_imgr, f_imgg, f_imgb], dim = 0)
magnitude2 = torch.cat([magnituder, magnitudeg, magnitudeb], dim = 0)
phase2 = torch.cat([phaser, phaseg, phaseb], dim = 0)

print(abs(f_img-f_img2).mean(), "\n", abs(magnitude-magnitude2).mean(), "\n", abs(phase-phase2).mean())