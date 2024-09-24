import numpy as np
import sys, os
from importlib import reload

from . import legacy

NC_FREQ = 514 # must be 2**N + 2
NR_FREQ = 512 # must be 2**N
NC_SPAT = NC_FREQ
NR_SPAT = NR_FREQ # must be 2**N
NSUB = legacy.NSUB

# for testing purpose
reload(legacy)

def set_padding_size(NC, NR):
    global NC_FREQ, NR_FREQ, NC_SPAT, NR_SPAT
    NC_FREQ = NC
    NR_FREQ = NR
    NC_SPAT = NC_FREQ
    NR_SPAT = NR_FREQ

def set_dither_size(factor):
    global NSUB
    NSUB = factor

def fft(data): 
    b = np.zeros((NC_SPAT, NR_SPAT))
    b[:data.shape[0], :data.shape[1]] = data
    # b[100:200, 100:200] = data_large[100:200, 100:200]
    a = np.zeros((NC_FREQ, NR_FREQ))
    isign = 1
    work = np.zeros((2, NR_SPAT))
    data_hat = legacy.real2dfft(a, NR_FREQ, NC_FREQ, b, NR_SPAT, NC_SPAT, isign, work, onedim=False)
    # plt.imshow(norm(orig_hat))
    # plt.colorbar()
    return data_hat

def ifft(data_hat): 
    b = np.zeros((NC_FREQ, NR_FREQ))
    b[:data_hat.shape[0], :data_hat.shape[1]] = data_hat
    # b[100:200, 100:200] = data_large[100:200, 100:200]
    a = np.zeros((NC_SPAT, NR_SPAT))
    isign = -1
    work = np.zeros((2, NR_FREQ))
    data = legacy.real2dfft(a, NR_SPAT, NC_SPAT, b, NR_FREQ, NC_FREQ, isign, work, onedim=False)
    # plt.imshow(norm(orig_hat))
    # plt.colorbar()
    return data

def phase_shift(A, offsets, n, verbose=True):
    DR = 0
    DC = 0
    shift = False
    if not verbose: 
        sys.stdout = open(os.devnull, 'w')
    Aphased = legacy.phase(A, NR_FREQ, NC_FREQ, shift, DR, DC, offsets, n)
    if not verbose: 
        sys.stdout = sys.__stdout__
    return Aphased
