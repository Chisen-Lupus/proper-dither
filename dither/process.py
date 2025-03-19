
import os, sys
import numpy as np
import scipy
import copy
from typing import Any



def combine_image(normalized_atlas, centroids, wt=None, oversample=2) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Apply phase shifts to the input data.

    Parameters
    ----------
    normalized_atlas : list-like container of arrays
        Input 2D array TODO
    centroids : list 
        TODO
    wt : int
        TODO
    oversample : int
        TODO

    Returns
    -------
    ndarray
        TODO
    """

    # GENERATE POSSIBLY MISSING INPUT

    if wt is None:
        wt = np.ones(len(normalized_atlas))

    # ASSERTATION

    assert len(normalized_atlas)==len(centroids)
    assert len(centroids)==len(wt)
    assert all(im.size==normalized_atlas[0].size for im in normalized_atlas)

    # SOME GLOBAL FACTORS

    NSUB = oversample
    NPP = len(normalized_atlas)
    NX, NY = normalized_atlas[0].shape
    NX_LARGE = NX*NSUB
    NY_LARGE = NY*NSUB
    N =int(np.ceil(np.log2(np.max([NX_LARGE, NY_LARGE]))))
    NC_FREQ = 2**N + 2
    NR_FREQ = 2**N

    Atotal = np.zeros((NC_FREQ//2, NR_FREQ), dtype=np.complex128)
    F = np.zeros((NR_FREQ, NR_FREQ), dtype=np.complex128)

    for npos in range(len(normalized_atlas)): 

        data = normalized_atlas[npos]
        data_large = np.zeros((NR_FREQ, NR_FREQ))
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
                isatx, isaty = np.meshgrid(np.arange(NSUB), np.arange(NSUB))
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
                if NPP>NSUB**2: 
                    phasem = phases @ np.diag(wt) @ np.conj(phases).T
                else: 
                    phasem = phases

                vec = np.linalg.inv(phasem)

                # For NSUB2 images, we are done
                if NPP==NSUB**2:
                    coef[iy, ix] = vec[npos, 0]
                # Otherwise, we need to do a little more work. Here we just solve for the fundamental image.
                else: 
                    coef[iy, ix] = 0
                    for i in range(NSUB**2):
                        coef[iy, ix] += vec[i, 0]*np.conj(phases[i, npos])

                # Add weighting factor
                coef[iy, ix] *= wt[npos]

                # print(f'Image {npos}, power {coef[isec]*np.conj(coef[isec])}, sector {isec}')

        # print('---')

        # END COEFFICIENT COMPUTATION

        # BEGIN FFT2

        # We only need half of the transformed array since we are doing real transform
        A_hat = scipy.fft.fft2(data_large) # data_large must be (2^N, 2^N) for now
        A_unique = A_hat[:NC_FREQ//2, :]  # shape (NC_FREQ//2, NR_FREQ)
        A_complex = np.conj(A_unique)
        # A_complex = np.conj(A_hat)

        # END FFT2

        # BEGIN PHASE SHIFT APPLICATION

        for iy in range(NSUB):
            for ix in range(NSUB):

                # Starting and ending points of this sector
                nu = NC_FREQ//NSUB
                isu = min(nu*ix, NC_FREQ//2)
                ieu = min(nu*(ix+1), NC_FREQ//2)
                if isu==ieu: 
                    break

                nv = NR_FREQ//NSUB
                isv = NR_FREQ//2 - nv*iy
                iev = NR_FREQ//2 - nv*(iy+1) if iy<NSUB-1 else NR_FREQ//2 - NR_FREQ

                # Extract the complex coefficient
                coef_complex = coef[iy, ix]

                # Compute the normalized row positions (V)
                # print('ix', ix, 'iy', iy)
                # print('isu', isu, 'ieu', ieu, 'isv', isv, 'iev', iev)
                rows = np.arange(isv-1, iev-1, -1)
                # rows = np.where(rows >= 0, rows, NR_FREQ + rows) # numpy array can take negative index
                V = np.where(rows >= NR_FREQ // 2, (rows - NR_FREQ) / NR_FREQ, rows / NR_FREQ)

                # Compute the row phase shift (as a complex exponential)
                rphase = np.exp(-2j * phiy[npos] * V)

                # Compute the normalized column positions (U)
                cols = np.arange(isu, ieu)
                U = cols / (NC_FREQ - 2)   # Multiply back by 2 to match original scale

                # Compute the column phase shift (as a complex exponential)
                cphase = np.exp(-2j * phix[npos] * U)

                # Compute the overall phase shift (outer product for broadcasting)
                phase_shift = coef_complex * np.outer(cphase, rphase)

                # Apply the phase shift to A
                # print(U, V)
                # print('cols', cols[[0, -1]], cols.shape)
                # print('rows', rows[[0, -1]], rows.shape)
                A_complex[np.ix_(cols, rows)] *= phase_shift  # No need for cols // 2

        Atotal += np.conj(A_complex)
        # F += np.conj(A_complex)

        # print('------')

    # END PHASE SHIFT APPLICATION

    # BEGIN IFFT2
    
    F[:NC_FREQ//2, :] = Atotal
    F[NC_FREQ//2:, 0] = np.conj(Atotal[1:NR_FREQ//2])[::-1, 0]
    F[NC_FREQ//2:, 1:] = np.conj(Atotal[1:NR_FREQ//2])[::-1, :0:-1]
    data_rec = scipy.fft.ifft2(F)
    data_real = data_rec.real

    # END IFFT2

    combined_image = data_real[:NX_LARGE, :NY_LARGE]

    return combined_image
