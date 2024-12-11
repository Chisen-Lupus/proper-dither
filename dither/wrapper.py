"""
This module provides a wrapper for the ``legacy`` module, offering higher-level
functions for working with Fourier transforms and phase adjustments. It simplifies
the usage of functions from ``legacy`` by providing default parameters and
convenient interfaces.

Key Features:
-------------
- Padding and configuration settings for Fourier transform dimensions.
- High-level FFT and IFFT functions for 2D data.
- Phase shifting function for advanced image processing.

Dependencies:
-------------
- numpy
- legacy (translated Fortran77 code from Luer's (1999) paper).

Example Usage:
--------------
.. code-block:: python

    from wrapper import set_padding_size, fft, ifft, phase_shift

    # Set padding size
    set_padding_size(NC=1024, NR=1024)

    # Perform FFT
    data_hat = fft(data)

    # Perform IFFT
    reconstructed_data = ifft(data_hat)

    # Apply phase shift
    A_phased = phase_shift(A, offsets=[[0, 0, 1]], n=1)
"""

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
    """
    Set the padding size for FFT operations.

    Parameters
    ----------
    NC : int
        Number of columns for the Fourier transform.
    NR : int
        Number of rows for the Fourier transform.

    Notes
    -----
    Both NC and NR must satisfy the constraints imposed by the Fourier transform:
    NC must be ``2**N + 2``, and NR must be ``2**N`` for some integer N.
    """
    global NC_FREQ, NR_FREQ, NC_SPAT, NR_SPAT
    NC_FREQ = NC
    NR_FREQ = NR
    NC_SPAT = NC_FREQ
    NR_SPAT = NR_FREQ

def set_dither_size(factor):
    """
    Set the dither factor used in the phase shifting process.

    Parameters
    ----------
    factor : int
        Dither factor (number of sub-samples per pixel).
    """
    global NSUB
    NSUB = factor

def fft(data): 
    """
    Perform a 2D forward real-to-complex Fourier transform.

    Parameters
    ----------
    data : ndarray
        Input 2D array for transformation.

    Returns
    -------
    ndarray
        Transformed data in the Fourier domain.

    Notes
    -----
    The input data is padded to the size specified by ``NC_SPAT`` and ``NR_SPAT``
    before performing the FFT.
    """
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
    """
    Perform a 2D inverse complex-to-real Fourier transform.

    Parameters
    ----------
    data_hat : ndarray
        Input 2D array in the Fourier domain.

    Returns
    -------
    ndarray
        Reconstructed data in the spatial domain.

    Notes
    -----
    The input data is padded to the size specified by ``NC_FREQ`` and ``NR_FREQ``
    before performing the inverse FFT.
    """
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
    """
    Apply phase shifts to the input data.

    Parameters
    ----------
    A : ndarray
        Input 2D array representing Fourier-transformed data.
    offsets : list of lists
        Array of offsets and weights, with shape (n, 3), where the columns are
        x-offset, y-offset, and weight.
    n : int
        Index of the position for which the coefficients are computed.
    verbose : bool, optional
        Whether to suppress internal print statements. Default is True.

    Returns
    -------
    ndarray
        Modified 2D array after phase adjustment.

    Notes
    -----
    This function uses the ``phase`` function from the ``legacy`` module to
    perform the adjustments.
    """
    DR = 0
    DC = 0
    shift = False
    if not verbose: 
        stdout_backup = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try: 
            Aphased = legacy.phase(A, NR_FREQ, NC_FREQ, shift, DR, DC, offsets, n)
        except Exception as e: 
            print(repr(e))
        finally:
            sys.stdout = stdout_backup
    else: 
        Aphased = legacy.phase(A, NR_FREQ, NC_FREQ, shift, DR, DC, offsets, n)
    return Aphased
