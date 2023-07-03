import numpy as np
import cv2
import os
import torch
from pytorch_wavelets import DWTForward, DWTInverse

level = 4
dwt = DWTForward(J=level, wave='haar', mode='zero')
idwt = DWTInverse(wave='haar', mode='zero')

def Split(img):
    '''
        compose subband-images to 1 image
        Args:
        img: tensor[B, C*4, H/2**j, W/2**j]
        Return:
        image: tensor[B, C, H/2**(j-1), W/2**(j-1)]
    '''

    result = torch.split(img, split_size_or_sections=int(img.shape[1]/4),dim=1)
    imglin1 = torch.cat([result[0], result[1]], dim=3)
    imglin2 = torch.cat([result[2], result[3]], dim=3)
    img = torch.cat([imglin1, imglin2], dim=2)
    return img

original_path = r"ParallelRenderTestblended/testBlended"

target = r"DataAnalysis/waveletL4_parallel_haar"

for file in os.listdir(original_path):

    original1 = cv2.imread(os.path.join(original_path, file))
    original = cv2.cvtColor(original1, cv2.COLOR_BGR2RGB) # [256, 256*5, 3]

    original = torch.tensor(original.transpose(2, 0, 1)) # torch.tensor(3, 256, 256*5)
    original = torch.unsqueeze(1/3*original[0,:,:] + 1/3*original[1,:,:] + 1/3*original[2,:,:], dim=0) #[1,h,w]
    imgs = torch.split(original, int(original.shape[2]/5), dim=2)

    line2 = []
    # line4 = [] # save idwt
    line3 = [] # middle value
    for i in range(0,5):
        input = imgs[i].unsqueeze(0) # torch.tensor(1, C, 256, 256)

        f_inputl, f_inputh = dwt(input) # f_inputl: tensor(B,C,H/2**level,W/2**level), f_inputh[i]: tensor(B,C,3,H/2**(i+1),W/2**(i+1))

        r_input = idwt((f_inputl, f_inputh)) # idwt from dwt image
        # line4.append(r_input)

        level_inx = int(len(f_inputh))-1 # lastest level index
        input_l = f_inputl

        for j in range(level_inx, -1, -1): # j-level
            lh = f_inputh[j][:, :, 0, :, :] # (B, C, H/2**(j+1), W/2**(j+1))
            hl = f_inputh[j][:, :, 1, :, :]
            hh = f_inputh[j][:, :, 2, :, :]

            f_input = Split(torch.cat([input_l, lh, hl, hh], dim = 1)) # (B, C*4, H/2**(j+1), W/2**(j+1)) -> (B, C, H/2**j, W/2**j)
            input_l = f_input # (B, C, H/2**j, W/2**j)

            # set j-level coarse coefficient
            middle_l = input_l #(B, C, 3, H/2**(j+1), W/2**(j+1))
            
        line2.append(f_input)

    for L in range(level, 0, -1):
        mdwt = DWTForward(J=L, wave='haar', mode='zero')
        m_inputl, m_inputh = mdwt(imgs[0].unsqueeze(0))
        m_inputhnew = []
        for t in range(0, L):
            m_inputhnew.append(torch.zeros_like(m_inputh[t]))
        m_input = idwt((m_inputl, m_inputhnew))
        line3.append(m_input) # from coarse to fine, shape[H, 1, H, W]
    if level<5:
        sup = torch.zeros_like(imgs[0].unsqueeze(0))
        for s in range(level+1, 5+1):
            line3.append(sup)

    con = torch.cat(line2, dim = 3).squeeze(0)
    con2 = torch.cat(line3, dim= 3).squeeze(0)
    # con3 = torch.cat(line4, dim = 3).squeeze(0)
    # result = torch.cat([original, con, con2, con3], dim=1)
    result = torch.cat([original, con, con2], dim=1)
    result = result.repeat(3,1,1)    
    result = result.numpy().transpose(1, 2, 0).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(target, file), result)