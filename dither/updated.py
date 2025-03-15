"""
This module implements algorithms for Fourier transform operations, phase adjustments,
and related computations using NumPy's FFT methods. It provides optimized versions of
the routines found in ``wrapper.py``, significantly improving computation speed while
maintaining compatibility with the existing workflow.

Key Features:
-------------
- **Fourier Transforms**: Fast 1D and 2D real-to-complex and complex-to-real transforms.
- **Phase Adjustments**: Efficient phase shifting for advanced image processing.
- **Enhanced Performance**: Leveraging NumPy's optimized FFT implementations.

Dependencies:
-------------
- numpy

Example Usage:
--------------
.. code-block:: python

    from updated import real2dfft_forward, real2dfft_backward, phase_shift

    # Perform a forward 2D real FFT
    data_hat = real2dfft_forward(data)

    # Perform an inverse 2D real FFT
    reconstructed_data = real2dfft_backward(data_hat)

    # Apply phase shifting
    A_phased = phase_shift(A, offsets=[[0, 0, 1]], n=1)
"""

import os, sys
import numpy as np
import scipy

# NOTE: If you need some fourier transform in old days...

def four1_forward(a): 
    """
    Perform a 1D forward complex-to-complex FFT using NumPy.

    Parameters
    ----------
    a : ndarray
        Input array alternating real and imaginary components.

    Returns
    -------
    ndarray
        Fourier-transformed output array alternating real and imaginary components.
    """
    a_complex = a[::2]+a[1::2]*1j
    a_hat_complex = np.fft.fft(a_complex)
    a_hat_real = a_hat_complex.real
    a_hat_imag = a_hat_complex.imag
    a_hat = np.zeros(len(a))
    a_hat[0] = a_hat_real[0]
    a_hat[1] = a_hat_imag[0]
    a_hat[2::2] = a_hat_real[-1:0:-1]
    a_hat[3::2] = a_hat_imag[-1:0:-1]
    return a_hat

def four1_backward(a_hat): 
    """
    Perform a 1D backward complex-to-complex FFT using NumPy.

    Parameters
    ----------
    a_hat : ndarray
        Fourier-transformed input array alternating real and imaginary components.

    Returns
    -------
    ndarray
        Inverse Fourier-transformed output array alternating real and imaginary components.
    """
    a_hat_real = np.zeros(len(a_hat)//2)
    a_hat_imag = np.zeros(len(a_hat)//2)
    a_hat_real[-1:0:-1] = a_hat[2::2]
    a_hat_imag[-1:0:-1] = a_hat[3::2]
    a_hat_real[0] = a_hat[0]
    a_hat_imag[0] = a_hat[1]
    a_hat_complex = a_hat_real + a_hat_imag*1j
    a_complex = np.fft.ifft(a_hat_complex)
    a = np.zeros(len(a_hat))
    a[::2] = a_complex.real
    a[1::2] = a_complex.imag
    a *= len(a_hat)//2
    return a

def realft_forward(a): 
    """
    Perform a 1D forward real-to-complex FFT using NumPy.

    Parameters
    ----------
    a : ndarray
        Input real array.

    Returns
    -------
    ndarray
        Fourier-transformed output array alternating real and imaginary components.
    """
    n = len(a)
    nn = (len(a) - 2)//2
    a_hat_complex = np.fft.fft(a[:nn*2])
    a_hat_real = a_hat_complex.real
    a_hat_imag = a_hat_complex.imag
    a_hat = np.zeros(n)
    a_hat[::2] = a_hat_real[:nn+1]
    a_hat[1::2] = -a_hat_imag[:nn+1]
    return a_hat

def realft_backward(a_hat): 
    """
    Perform a 1D backward complex-to-real FFT using NumPy.

    Parameters
    ----------
    a_hat : ndarray
        Fourier-transformed input array alternating real and imaginary components.

    Returns
    -------
    ndarray
        Inverse Fourier-transformed real array.
    """
    nn = (len(a_hat) - 2)//2
    a_hat_real = np.zeros(nn*2)
    a_hat_real[:nn+1] = a_hat[0::2]
    a_hat_real[nn+1:] = a_hat[2:-2:2][::-1]
    a_hat_imag = np.zeros(nn*2)
    a_hat_imag[:nn+1] = -a_hat[1::2]
    a_hat_imag[nn+1:] = a_hat[3:-1:2][::-1]
    a_complex = a_hat_real + a_hat_imag*1j
    a = np.fft.ifft(a_complex)
    a = np.concatenate((a, [0]*2))
    a = a.real
    a *= nn
    return a

def real2dfft_forward(data):
    """
    Perform a 2D forward real-to-complex FFT using NumPy.

    Parameters
    ----------
    data : ndarray
        Input 2D real array.

    Returns
    -------
    ndarray
        Fourier-transformed 2D array.
    """
    nc, nr = data.shape
    data_hat = data.copy()
    for ir in range(nr):
        data_hat[:, ir] = realft_forward(data_hat[:, ir])
    for ic in range(0, nc, 2):
        row = np.zeros(nr*2)
        row[0::2] = data_hat[ic, :]
        row[1::2] = data_hat[ic+1, :]
        row_hat = four1_forward(row)
        data_hat[ic, :] = row_hat[0::2]
        data_hat[ic+1, :] = row_hat[1::2]
    return data_hat

def real2dfft_backward(data):
    """
    Perform a 2D backward complex-to-real FFT using NumPy.

    Parameters
    ----------
    data : ndarray
        Fourier-transformed 2D array.

    Returns
    -------
    ndarray
        Reconstructed 2D real array.
    """
    nc, nr = data.shape
    data_hat = data.copy()
    for ic in range(0, nc, 2):
        row = np.zeros(nr*2)
        row[0::2] = data_hat[ic, :]
        row[1::2] = data_hat[ic+1, :]
        row_hat = four1_backward(row)
        data_hat[ic, :] = row_hat[0::2]
        data_hat[ic+1, :] = row_hat[1::2]
    for ir in range(nr):
        data_hat[:, ir] = realft_backward(data_hat[:, ir])
    data_hat /= nr*(nc - 2)//2
    return data_hat


PI = np.pi
RADIOAN = PI/180
NSUB = 2
NDIV = NSUB
NC_FREQ = 514 # must be 2**N + 2
NR_FREQ = 512 # must be 2**N
NC_SPAT = NC_FREQ
NR_SPAT = NR_FREQ # must be 2**N

def phase(A_in, offsets, npos):
    """
    Apply phase shifts and calculate coefficients for Fourier-based image transformations.

    Parameters
    ----------
    A_in : ndarray
        Input 2D array representing the Fourier-transformed data.
    offsets : ndarray
        Array of offsets and weights, with shape (n, 3), where the columns are x-offset, y-offset, and weight.
    npos : int
        Index of the position for which the coefficients are computed.

    Returns
    -------
    ndarraygit 
        Modified 2D array after phase adjustment and transformation.
    """
    A = A_in.copy()
    NST = len(offsets)
    phix = np.zeros(NST)
    phasem = np.zeros((NDIV**2, NDIV**2), dtype=np.complex128)
    vec = np.zeros((NDIV**2, NDIV**2), dtype=np.complex128)
    coef = np.zeros((NDIV**2), dtype=np.complex128)
    phases = np.zeros((NDIV**2, NST), dtype=np.complex128)
    key = np.zeros(NDIV**2)
    
    # LINE 247 - read offsets (totally different from the original code)

    offsets = np.array(offsets)
    xcr = offsets[0, 0]
    ycr = offsets[0, 1]
    dx = offsets[:, 0] - xcr
    dy = offsets[:, 1] - ycr
    phix = NSUB*PI*dx
    phiy = NSUB*PI*dy
    wt = offsets[:, 2]
    npp = len(offsets)

    # LINE 289 - Calculate the coefficients for each image. 

    nsy = NSUB # 2
    nsx = (NSUB - 1)//2 + 1 # 1
    isy = -((NSUB-1)//2) + 1 # 0, NOTE: added a bracket to regulate different handling of integer division
    isy = 0
    isec = 0

    for iy in range(isy, isy+nsy): 
        for ix in range(0, nsx): 

            isec += 1

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

            if npp>NSUB**2: 
                phasem = phases @ np.diag(wt) @ np.conj(phases).T
            else: 
                phasem = phases
                
            vec = np.linalg.inv(phasem)

            # LINE 490 - For NSUB2 images, we are done
            
            if npp==NSUB**2:
                coef[isec-1] = vec[npos-1, 0]

            # LINE 495 - Otherwise, we need to do a little more work.  Here we just solve for the fundamental image.
            
            else: 
                # print('npp!=NSUB**2')
                coef[isec-1] = 0
                for i in range(1, NSUB**2+1):
                    coef[isec-1] += vec[i-1, 0]*np.conj(phases[i-1, npos-1])

            # LINE 505 - Addin weighting factor

            coef[isec-1] *= wt[npos-1]

    # LINE 516 - apply the complex scale factor to the transform

    isec = 0
    isv = NR_FREQ//2
    iev = isv - NR_FREQ//NSUB + 1 
    for iy in range(isy, isy+nsy):
        ieu = NC_FREQ - (NSUB - 1)*(nsx - 1)*NC_FREQ//NSUB
        isu = 1
        for ix in range(0, nsx):
            isec += 1
            A_complex = A[::2, :] + 1j * A[1::2, :]

            # Extract the complex coefficient
            coef_complex = coef[isec-1]

            # Compute the normalized row positions (V)
            rows = np.arange(isv, iev-1, -1)
            rows = np.where(rows > 0, rows, NR_FREQ + rows)
            V = np.where(rows > NR_FREQ // 2, (rows - NR_FREQ - 1) / NR_FREQ, (rows - 1) / NR_FREQ)

            # Compute the row phase shift (as a complex exponential)
            rphase = np.exp(-2j * phiy[npos-1] * V)

            # Compute the normalized column positions (U)
            cols = np.arange(isu, ieu + 1, 2)
            U = (cols - 1) / (NC_FREQ - 2) / 2

            # Compute the column phase shift (as a complex exponential)
            cphase = np.exp(-2j * phix[npos-1] * U)

            # Compute the overall phase shift (outer product for broadcasting)
            phase_shift = coef_complex * np.outer(cphase, rphase)

            # Apply the phase shift to A
            A_complex[np.ix_(cols // 2, rows - 1)] *= phase_shift

            # If needed, convert back to separate real and imaginary parts
            A[::2, :] = A_complex.real
            A[1::2, :] = A_complex.imag
            
            isu = ieu + 1
            ieu = NC_FREQ - (nsx - 2 - ix)*NC_FREQ//NSUB # TODO: check values
        isv = iev - 1
        iev = isv - NR_FREQ//NSUB + 1 # TODO: check values
        if iy==(isy + nsy - 2): 
            iev = -(NR_FREQ//2) + 1 # NOTE: add a bracket to change negative sign to minus sign

    return A

N514 = NC_FREQ
N512 = NR_FREQ
N257 = N514//2
N256 = N512//2


np.set_printoptions(linewidth=200)

def legacy_fft2(data_large):
    """
    data_large: real array of shape (N514, 512)
    Only the first 512 rows (i.e. 2*nfft with nfft=N256) are used in the FFT.
    The FFT output is then packed into a (N514, 512) array:
    - Even-indexed rows hold the real parts of the unique frequencies.
    - Odd-indexed rows hold the negative imaginary parts.
    """
    # Use only the first 512 rows for the FFT (2 * N256)
    fft_input = data_large[:N512, :]
    
    # Compute the full 2D FFT; shape is (N512, N512)
    A_complex = scipy.fft.fft2(fft_input)
    # print('A_complex\n', A_complex)
    
    # For a real signal of length N512, the unique FFT coefficients are indices 0 to N256 (inclusive)
    # This gives us N257 rows of unique data.
    A_unique = A_complex[:N257, :]  # shape (N257, N512)
    
    # Pack the FFT output into a half-complex real array of shape (N514, N512)
    A = np.zeros((N514, N512), dtype=data_large.dtype)
    A[0::2, :] = A_unique.real    # even-indexed rows
    A[1::2, :] = -A_unique.imag   # odd-indexed rows
    
    return A#, A_complex

def legacy_ifft2(A):
    """
    Inverse of legacy_fft2.

    Parameters
    ----------
    A : np.array, shape (N514, N512)
        Half-complex FFT output as produced by legacy_fft2,
        where even-indexed rows hold the real parts of the unique coefficients,
        and odd-indexed rows hold the negative imaginary parts.

    Returns
    -------
    data_reconstructed : np.array, shape (N512, N512)
        The inverse FFT computed from the unique coefficients.
    """
    # print('A\n', A)
    U = A[0::2, :] - 1j * A[1::2, :]  # U.shape = (257, 512)
    # print('U\n', U)

    # Reconstruct the full FFT array (shape 512×512).
    # The unique coefficients U are for indices 0 to 256. For k=1,...,255, the
    # full FFT satisfies F[512-k] = conj(F[k]). (Note: index 256 is the Nyquist.)
    F = np.zeros((N512, N512), dtype=np.complex128)
    # print('F\n', F)
    F[:N257, :] = U
    F[N257:, 0] = np.conj(U[1:N256])[::-1, 0]
    F[N257:, 1:] = np.conj(U[1:N256])[::-1, :0:-1]


    # print('F\n', F.shape)
    # F = A[0::2] - 1j * A[1::2]

    # For 1 <= k <= N//2 - 1, recover the negative frequencies.
    # That is, for each k, set F[N-k, :] = conj( U[k, (-j mod M)] ).
    # The column reordering is done by reversing U[k, :] and then rolling right by 1.
    # print('F\n', F)
    
    # Compute the inverse 2D FFT.
    # Note: scipy.fft.ifft2 scales by 1/(N512*N512), while our routines effectively use a factor of 1/(N512*N256).
    # Multiplying by 2 adjusts for this normalization difference.
    data_rec = scipy.fft.ifft2(F)
    
    # Transpose back to recover the original ordering.
    # data_rec = data_rec.T
    # print(data_rec.imag)
    
    # Place the reconstructed data into a (N514, N512) array.
    data_return = np.zeros((N514, N512))
    data_return[:N512] = data_rec.real
    return data_return#, F


def combine_image(normalized_atlas, centroids, wt):

    # [y, x, wt]
    offsets = np.hstack((centroids[:, ::-1], wt[:, np.newaxis]))
    # print(offsets)
    Atotal = np.zeros((NC_FREQ, NR_FREQ))
    for i in range(len(normalized_atlas)): 
        data = normalized_atlas[i]
        nx, ny = data.shape
        data_large = np.zeros((NC_FREQ, NR_FREQ))
        data_large[:nx*NSUB:NSUB, :ny*NSUB:NSUB] = data


        # BEGIN FFT2

        A = legacy_fft2(data_large)

        # END FFT2

        # BEGIN PHASE

        npos = i + 1
        NST = len(offsets)
        phix = np.zeros(NST)
        # spr, spi, rpr, rpi, cpr, cpi, ypr, ypi, tpr, tpi = 0
        # fr, fi = 0
        phasem = np.zeros((NDIV**2, NDIV**2), dtype=np.complex128)
        vec = np.zeros((NDIV**2, NDIV**2), dtype=np.complex128)
        coef = np.zeros((NDIV**2), dtype=np.complex128)
        phases = np.zeros((NDIV**2, NST), dtype=np.complex128)
        # row, col, vrow = 0
        key = np.zeros(NDIV**2)
        
        # LINE 247 - read offsets (totally different from the original code)

        offsets = np.array(offsets)
        xcr = offsets[0, 0]
        ycr = offsets[0, 1]
        dx = offsets[:, 0] - xcr
        dy = offsets[:, 1] - ycr
        phix = NSUB*PI*dx
        phiy = NSUB*PI*dy
        wt = offsets[:, 2]
        npp = len(offsets)

        # LINE 289 - Calculate the coefficients for each image. 

        nsy = NSUB # 2
        nsx = (NSUB - 1)//2 + 1 # 1
        isy = -((NSUB-1)//2) + 1 # 0, NOTE: added a bracket to regulate different handling of integer division
        isy = 0
        isec = 0

        # print('iy', list(range(isy, isy+nsy)))
        # print('ix', list(range(0, nsx)))
        for iy in range(isy, isy+nsy): 
            for ix in range(0, nsx): 

                isec += 1

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

                if npp>NSUB**2: 
                    phasem = phases @ np.diag(wt) @ np.conj(phases).T
                else: 
                    phasem = phases
                    
                vec = np.linalg.inv(phasem)

                # # LINE 490 - For NSUB2 images, we are done
                
                if npp==NSUB**2:
                    # print('npp==NSUB**2')
                    # print('coef', coef)
                    # print('vec', vec[:, 0], npos-1)
                    coef[isec-1] = vec[npos-1, 0]

                # LINE 495 - Otherwise, we need to do a little more work.  Here we just solve for the fundamental image.
                
                else: 
                    # print('npp!=NSUB**2')
                    coef[isec-1] = 0
                    for i in range(1, NSUB**2+1):
                        coef[isec-1] += vec[i-1, 0]*np.conj(phases[i-1, npos-1])
                # print(isec, npos, coef[isec], coef)

                # LINE 505 - Addin weighting factor

                coef[isec-1] *= wt[npos-1]

                # print(f'Image {npos}, power {coef[isec-1]*np.conj(coef[isec-1])}, sector {isec}')
                # print(f'Image {npos}, power {coef[isec-1]}, sector {isec}')
            #     break
            # break

        # print('coef', coef)
        # COEFS.append(coef)

        # LINE 516 - apply the complex scale factor to the transform

        isec = 0
        isv = NR_FREQ//2
        iev = isv - NR_FREQ//NSUB + 1 
        # print(A)
        # print('phix', phix)
        # print('phiy', phiy)
        # print(isv, iev)
        # print(nsy)
        # print(nsx)
        # print(isv-(iev-1))
        for iy in range(isy, isy+nsy):
            ieu = NC_FREQ - (NSUB - 1)*(nsx - 1)*NC_FREQ//NSUB
            # print(ieu) # TODO: verify that ieu==NC_FREQ if nsx = 1
            isu = 1
            for ix in range(0, nsx):
                isec += 1
                A_complex = A[::2, :] + 1j * A[1::2, :]

                # Extract the complex coefficient
                coef_complex = coef[isec-1]

                # Compute the normalized row positions (V)
                rows = np.arange(isv, iev-1, -1)
                rows = np.where(rows > 0, rows, NR_FREQ + rows)
                V = np.where(rows > NR_FREQ // 2, (rows - NR_FREQ - 1) / NR_FREQ, (rows - 1) / NR_FREQ)

                # Compute the row phase shift (as a complex exponential)
                rphase = np.exp(-2j * phiy[npos-1] * V)

                # Compute the normalized column positions (U)
                cols = np.arange(isu, ieu + 1, 2)
                U = (cols - 1) / (NC_FREQ - 2) / 2

                # Compute the column phase shift (as a complex exponential)
                cphase = np.exp(-2j * phix[npos-1] * U)

                # Compute the overall phase shift (outer product for broadcasting)
                phase_shift = coef_complex * np.outer(cphase, rphase)

                # Apply the phase shift to A
                A_complex[np.ix_(cols // 2, rows - 1)] *= phase_shift

                # If needed, convert back to separate real and imaginary parts
                A[::2, :] = A_complex.real
                A[1::2, :] = A_complex.imag
                
                isu = ieu + 1
                ieu = NC_FREQ - (nsx - 2 - ix)*NC_FREQ//NSUB # TODO: check values
            isv = iev - 1
            iev = isv - NR_FREQ//NSUB + 1 # TODO: check values
            if iy==(isy + nsy - 2): 
                iev = -(NR_FREQ//2) + 1 # NOTE: add a bracket to change negative sign to minus sign

        # print(vec)
        # print(phasem)
        # plt.imshow(vec.real)
        # plt.colorbar()
        # plt.show()
        # plt.imshow(phasem.real)
        # plt.colorbar()
        # plt.show()
        # plt.imshow((vec@phasem).real)
        # plt.colorbar()
        # plt.show()
        Aphased = A


        Atotal += Aphased
    # print(Atotal)
    # END PHASE

    # BEGIN IFFT2

    f = legacy_ifft2(Atotal)

    
    nx, ny = normalized_atlas[0].shape
    nx_large = nx*NSUB
    ny_large = ny*NSUB
    combined_image = f[:nx_large, :ny_large]
    return combined_image

np.fft.fft2