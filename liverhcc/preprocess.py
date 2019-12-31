import numpy as np
import nibabel as nib
import settings

# cut off img intensities
# img : npy array
# lb  : desired lower bound
# ub  : desired upper bound
def window(img, lb, ub):
    too_low  = img <= lb
    too_high = img >= ub
    img[too_low]  = lb
    img[too_high] = ub
    return img

# rescale img to [0,1]
# img : npy array
# lb  : known lower bound in image
# ub  : known upper bound in image
def rescale(img, lb, ub):
    rs = img.astype(settings.FLOAT_DTYPE)
    rs = 2.*(rs - lb)/(ub-lb)   - 1.0
    rs = rs.astype(settings.FLOAT_DTYPE)
    return rs

