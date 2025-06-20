
import os, sys
import numpy as np
import scipy
import copy
from typing import Callable, Any, Tuple, List, Optional
from numpy.typing import NDArray

import matplotlib.pyplot as plt

def combine_image(
    normalized_atlas: List[NDArray[np.float64]], 
    centroids: List[Tuple[float, float]], 
    wts: Optional[List[float]] = None, 
    oversample: int = 2, 
    # for tesing purpose:
    return_full_array: bool = False, 
    overpadding: int = 0
) -> NDArray[np.float64]:
    """
    Apply phase shifts to the input data.

    Parameters
    ----------
    normalized_atlas : list-like container of arrays
        Input 2D array TODO
    centroids : list 
        TODO
    wts : int
        TODO
    oversample : int
        TODO

    Returns
    -------
    ndarray
        TODO
    """

    # REGULARIZE INPUT

    centroids = np.array(centroids)
    if wts is None:
        wts = np.ones(len(normalized_atlas))

    # ASSERTATION

    assert len(normalized_atlas)==len(centroids)
    assert len(centroids)==len(wts)
    assert len(set([im.shape for im in normalized_atlas]))==1

    # SOME GLOBAL FACTORS

    NSUB = oversample
    NPP = len(normalized_atlas)
    NX, NY = normalized_atlas[0].shape
    NX_LARGE = NX*NSUB
    NY_LARGE = NY*NSUB
    NC_FREQ = int(2**np.ceil(np.log2(NX_LARGE))) # find the next 2^N+2 e.g. 514
    NR_FREQ = int(2**np.ceil(np.log2(NY_LARGE))) # 4x of NX_LARGE can effectlively dissipate the noise
    NC_FREQ *= 2**overpadding
    NR_FREQ *= 2**overpadding


    A_total = np.zeros((NC_FREQ//2+1, NR_FREQ), dtype=np.complex128)

    for npos in range(len(normalized_atlas)): 

        data = normalized_atlas[npos]
        data_large = np.zeros((NC_FREQ, NR_FREQ))
        data_large[:NX*NSUB:NSUB, :NY*NSUB:NSUB] = data
        coef = np.zeros((NSUB, NSUB), dtype=np.complex128)

        dx = centroids[:, 1]
        dy = centroids[:, 0]
        phix = NSUB*np.pi*dx
        phiy = NSUB*np.pi*dy

        # BEGIN COEFFICIENT COMPUTATION

        # NOTE: Only half of the coefficients calculated here are used for now.
        for iy in range(NSUB): 
            for ix in range(NSUB): 

                # Precompute normalized phase shifts
                px = -2 * phix / NSUB
                py = -2 * phiy / NSUB

                # Compute base indices and initial phases
                nuin = ix - (NSUB - 1) // 2
                nvin = iy
                pxi = nuin * px
                pyi = -nvin * py

                # Generate sub-grid indices
                isatx, isaty = np.meshgrid(np.arange(NSUB), np.arange(NSUB), indexing='xy')
                isatx = isatx.flatten()
                isaty = isaty.flatten()

                # Calculate total phase using broadcasting
                phit = np.outer(isatx, px) + pxi + np.outer(isaty, py) + pyi

                # Compute complex phases and normalize
                phases = (np.cos(phit) + 1j * np.sin(phit)) / NSUB**2

                # Pivot the fundamental component to the first row
                nfund = NSUB * nvin - nuin
                phases[[0, nfund], :] = phases[[nfund, 0], :]

                # Add weighting factor
                if NPP==NSUB**2:
                    phasem = phases
                else: 
                    phasem = phases @ np.diag(wts) @ np.conj(phases).T

                vec = np.linalg.inv(phasem)

                # For NSUB2 images, we are done
                if NPP==NSUB**2:
                    coef[iy, ix] = vec[npos, 0]
                # Otherwise, we need to do a little more work. Here we just solve for the fundamental image.
                else: 
                    coef[iy, ix] = 0
                    for i in range(NSUB**2):
                        coef[iy, ix] += vec[i, 0]*np.conj(phases[i, npos])

                    # XXX: Moving it to the else branch means totally ignore wts for NSUB**2 images
                    # Add weighting factor
                    coef[iy, ix] *= wts[npos]

                # print(f'Image {npos}, power {coef[isec]*np.conj(coef[isec])}, sector {isec}')

        # print('---')

        # END COEFFICIENT COMPUTATION

        # BEGIN FFT2

        # We only need half of the transformed array since we are doing real transform
        A_hat = np.conj(scipy.fft.rfft2(data_large, axes=(1, 0)))

        # END FFT2

        # BEGIN PHASE SHIFT APPLICATION

        for iy in range(NSUB):
            for ix in range(NSUB):

                # process columns

                # Starting and ending points of this sector
                nu = NC_FREQ//NSUB
                isu = min(nu*ix, NC_FREQ//2+1)
                ieu = min(nu*(ix+1), NC_FREQ//2+1)
                if isu==ieu: 
                    break

                # Compute the normalized column positions (U)
                cols = np.arange(isu, ieu)
                U = cols / NC_FREQ  # Multiply back by 2 to match original scale

                # Compute the column phase shift (as a complex exponential)
                cphase = np.exp(-2j * phix[npos] * U)

                # process rows

                nv = NR_FREQ//NSUB 
                isv = NR_FREQ//2 - nv*iy
                iev = NR_FREQ//2 - nv*(iy+1) if iy<NSUB-1 else NR_FREQ//2 - NR_FREQ

                # Extract the complex coefficient
                coef_complex = coef[iy, ix]

                # Compute the normalized row positions (V)
                rows = np.arange(isv-1, iev-1, -1)
                V = np.where(rows >= NR_FREQ // 2, (rows - NR_FREQ) / NR_FREQ, rows / NR_FREQ)

                # Compute the row phase shift (as a complex exponential)
                rphase = np.exp(-2j * phiy[npos] * V)

                # apply shift

                # Compute the overall phase shift (outer product for broadcasting)
                phase_shift = coef_complex * np.outer(cphase, rphase)

                # Apply the phase shift to A
                A_hat[np.ix_(cols, rows)] *= phase_shift  # No need for cols // 2

        A_total += A_hat

    # END PHASE SHIFT APPLICATION

    # BEGIN IFFT2
    
    data_rec = scipy.fft.irfft2(np.conj(A_total), s=(NC_FREQ, NR_FREQ), axes=(1, 0))
    data_real = data_rec.real

    # END IFFT2

    combined_image = data_real[:NX_LARGE, :NY_LARGE]

    if return_full_array:
        return data_real
    else:
        return combined_image
