import numpy as np
import cv2


def prepareImage(img):
    img = img / 2**16
    preparedImage =  np.array(
            np.dstack([
                    img[:,:,3],
                    img[:,:,7],
                    img[:,:,16],
                    img[:,:,20],
                    img[:,:,3] - img[:,:,16], # dif red
                    img[:,:,16] - img[:,:,3], # dif for transitionChange
                    img[:,:,8] - img[:,:,21],
                    img[:,:,20] - img[:,:,7],
                    img[:,:,12],
                    img[:,:,25],
                    img[:,:,12] - img[:,:,25],
                    img[:,:,25] - img[:,:,12], # dif for transitionChange
                    img[:,:,11], # dif for transitionChange
                    img[:,:,24], # dif for transitionChange
                    img[:,:,11] - img[:,:,24], # dif for transitionChange
                    img[:,:,24] - img[:,:,11],
                ])).astype('float32')
    return preparedImage

def prepareMask(mask):
    
    mask = (mask*65536).astype(int)
    mask[mask>1]=1
    return mask[:,:,0]


