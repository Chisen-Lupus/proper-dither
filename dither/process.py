
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

    for npos in range(len(normalized_atlas)): 

        # BEGIN PHASE

        data = normalized_atlas[npos]
        data_large = np.zeros((NR_FREQ, NR_FREQ))
        data_large[:NX*NSUB:NSUB, :NY*NSUB:NSUB] = data
        coef = np.zeros((NSUB, NSUB), dtype=np.complex128)
        
        # LINE 247 - read offsets (totally different from the original code)

        dx = centroids[:, 1]
        dy = centroids[:, 0]
        phix = NSUB*np.pi*dx
        phiy = NSUB*np.pi*dy

        # LINE 289 - Calculate the coefficients for each image. 

        nsy = NSUB # 2
        nsx = (NSUB - 1)//2 + 1 # 1
        isy = -((NSUB-1)//2) + 1 # 0, NOTE: added a bracket to regulate different handling of integer division
        isy = 0

        # BEGIN COEFFICIENT COMPUTATION

        for iy in range(isy, isy+nsy): 
            for ix in range(0, nsx): 

                # LINE 304

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

                # LINE 355 - This loads an identity matrix, which will be used to invert the phase matrix.

                # LINE 371 - The weighting factor is used at this point.

                if NPP>NSUB**2: 
                    phasem = phases @ np.diag(wt) @ np.conj(phases).T
                else: 
                    phasem = phases

                vec = np.linalg.inv(phasem)

                # LINE 490 - For NSUB2 images, we are done

                if NPP==NSUB**2:
                    coef[iy, ix] = vec[npos, 0]

                # LINE 495 - Otherwise, we need to do a little more work. Here we just solve for the fundamental image.

                else: 
                    coef[iy, ix] = 0
                    for i in range(NSUB**2):
                        coef[iy, ix] += vec[i, 0]*np.conj(phases[i, npos])

                # LINE 505 - Addin weighting factor

                coef[iy, ix] *= wt[npos]

                # print(f'Image {npos}, power {coef[isec]*np.conj(coef[isec])}, sector {isec}')
                # print(f'Image {npos}, power {coef[isec]}, sector {isec}')

        print('---')
        # END COEFFICIENT COMPUTATION

        # LINE 516 - apply the complex scale factor to the transform
        
        # BEGIN FFT2

        A_hat = scipy.fft.fft2(data_large) # data_large must be (2^N, 2^N) for now
        A_unique = A_hat[:NC_FREQ//2, :]  # shape (NC_FREQ//2, NR_FREQ)
        A_complex = np.conj(A_unique)

        # END FFT2

        # BEGIN PHASE SHIFT APPLICATION

        isv = NR_FREQ//2
        iev = isv - NR_FREQ//NSUB + 1 
        for iy in range(isy, isy+nsy):
            ieu = NC_FREQ - (NSUB - 1)*(nsx - 1)*NC_FREQ//NSUB
            # print(ieu) # TODO: verify that ieu==NC_FREQ if nsx = 1
            isu = 0
            for ix in range(0, nsx):
                # print('nsx', nsx)
                # nu = NC_FREQ//NSUB
                # print(NC_FREQ, NR_FREQ, nu)
                # isu = ix*nu
                # ieu = (ix+1)*nu

                # Extract the complex coefficient
                coef_complex = coef[iy, ix]

                # Compute the normalized row positions (V)
                # Define rows
                print('ix', ix, 'iy', iy)
                print('isu', isu, 'ieu', ieu, 'isv', isv, 'iev', iev)
                rows = np.arange(isv - 1, iev - 2, -1)  
                # rows = np.where(rows >= 0, rows, NR_FREQ + rows) # numpy array can take negative index
                V = np.where(rows >= NR_FREQ // 2, (rows - NR_FREQ) / NR_FREQ, rows / NR_FREQ)
                # print(V)

                # Compute the row phase shift (as a complex exponential)
                rphase = np.exp(-2j * phiy[npos] * V)

                # Compute the normalized column positions (U)
                # cols = np.arange(isu, ieu + 1, 2)
                cols = np.arange(isu, ieu, 2)
                U = cols/(NC_FREQ - 2)/2

                # Compute the column phase shift (as a complex exponential)
                cphase = np.exp(-2j * phix[npos] * U)

                # Compute the overall phase shift (outer product for broadcasting)
                phase_shift = coef_complex * np.outer(cphase, rphase)

                # Apply the phase shift to A
                A_complex[np.ix_(cols // 2, rows)] *= phase_shift
                print('cols', (cols // 2)[[0, -1]], cols.shape)
                print('rows', rows[[0, -1]], rows.shape)
                
                isu = ieu + 1
                ieu = NC_FREQ - (nsx - 2 - ix)*NC_FREQ//NSUB # TODO: check values
                # print(isu, ieu)
                
            isv = iev - 1
            iev = isv - NR_FREQ//NSUB + 1 # TODO: check values
            if iy==(isy + nsy - 2): 
                iev = -(NR_FREQ//2) + 1 # NOTE: add a bracket to change negative sign to minus sign

        Atotal += np.conj(A_complex)
        
        print('------')
    
    # END PHASE SHIFT APPLICATION

    # BEGIN IFFT2

    F = np.zeros((NR_FREQ, NR_FREQ), dtype=np.complex128)
    F[:NC_FREQ//2, :] = Atotal
    F[NC_FREQ//2:, 0] = np.conj(Atotal[1:NR_FREQ//2])[::-1, 0]
    F[NC_FREQ//2:, 1:] = np.conj(Atotal[1:NR_FREQ//2])[::-1, :0:-1]
    data_rec = scipy.fft.ifft2(F)
    data_real = data_rec.real

    # END IFFT2

    combined_image = data_real[:NX_LARGE, :NY_LARGE]

    return combined_image
