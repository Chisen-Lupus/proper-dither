"""
This module contains Python translations of functions from the Fortran77 code
described in Lauer's (1999) paper. These functions implement advanced Fourier
transform techniques and phase calculations for astronomical image processing.

Reference
---------
Lauer, T. R. (1999). Combining Undersampled Dithered Images.

Key Features
------------
- Phase adjustment and coefficient calculations using the ``phase`` function.
- 1D complex-to-complex FFT using ``four1``.
- Real-to-complex and complex-to-real FFT using ``realft``.
- 2D FFT transformations using ``real2dfft``.

Assumptions and Constraints
---------------------------
- The code assumes compatibility with NumPy arrays for numerical computations.
- Some algorithmic logic and constants (e.g., ``NSTMAX``, ``NDIV``) are directly
  ported from the Fortran77 code.

Example Usage
-------------
.. code-block:: python

    import legacy

    # Example: Perform phase adjustment
    input_array = ...  # Your 2D numpy array
    adjusted_array = legacy.phase(
        input_array, nrow=256, ncol=256, shift=True, DR=0.1, DC=0.2, 
        offsets=[[0, 0, 1]], npos=1
    )

    # Example: 1D FFT
    fft_result = legacy.four1(data, nn=128, isign=1)
"""

import numpy as np

PI = np.pi
RADIOAN = PI/180
NSTMAX = 140
NDIV = 3

NSUB = 2

# TODO: change the comments
# TODO: add well docmentation like the original files

def phase(A_in, nrow, ncol, shift, DR, DC, offsets, npos):
    """
    Apply phase shifts and calculate coefficients for Fourier-based image transformations.

    Parameters
    ----------
    A_in : ndarray
        Input 2D array representing the Fourier-transformed data.
    nrow : int
        Number of rows in the data.
    ncol : int
        Number of columns in the data.
    shift : bool
        Whether to apply a global phase shift based on `DR` and `DC`.
    DR : float
        Row shift factor.
    DC : float
        Column shift factor.
    offsets : ndarray
        Array of offsets and weights, with shape (n, 3), where the columns are x-offset, y-offset, and weight.
    npos : int
        Index of the position for which the coefficients are computed.

    Returns
    -------
    ndarray
        Modified 2D array after phase adjustment and transformation.
    """
    # TODO: check if they are all in use
    A = A_in.copy()
    phix = np.zeros(NSTMAX)
    # spr, spi, rpr, rpi, cpr, cpi, ypr, ypi, tpr, tpi = 0
    # fr, fi = 0
    phasem = np.zeros((NDIV**2, NDIV**2), dtype=np.complex128)
    vec = np.zeros((NDIV**2, NDIV**2), dtype=np.complex128)
    coef = np.zeros((NDIV**2), dtype=np.complex128)
    phases = np.zeros((NDIV**2, NSTMAX), dtype=np.complex128)
    # row, col, vrow = 0
    key = np.zeros(NDIV**2)

    # LINE 165
    
    if shift: 
        for row in range(1, nrow+1): 
            # U = 0
            if row<=nrow//2: 
                U = row/nrow
            else: 
                U = (row - nrow - 1)/nrow
            rphase = 2*PI*DR*U
            rpr = np.cos(rphase)
            rpi = np.sin(rphase)
            for col in range(1, ncol+1, 2):
                V = (col - 1)/ncol/2
                cphase = 2*PI*DC*V
                cpr = np.cos(cphase)
                cpi = np.sin(cphase)
                fr = A[col-1, row-1]
                fi = A[col+1-1, row-1]
                tpr = rpr*cpr - rpi*cpi
                tpi = rpi*cpr + rpr*cpi
                # print(tpr, tpi)
                # print(rphase, cphase, row)
                A[col-1, row-1] = fr*tpr - fi*tpi
                A[col+1-1, row-1] = fi*tpr + fr*tpi
                # print(A[col, row] - A_in[col, row])
                # print(A[col+1, row] - A_in[col+1, row])
        return A
    
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

            for nim in range(1, npp+1):
                px = -phix[nim-1]*2/NSUB
                py = -phiy[nim-1]*2/NSUB
                nuin = ix - (NSUB - 1)//2
                nvin = iy
                pxi = nuin*px
                pyi = -nvin*py
                # print('pxi', pxi, 'pyi', pyi)

                isat = 0
                for isaty in range(0, NSUB):
                    for isatx in range(0, NSUB): 
                        isat += 1
                        phit = isatx*px + pxi + isaty*py + pyi
                        # print(isat-1, nim-1, phit)
                        phases[isat-1, nim-1] = (np.cos(phit) + np.sin(phit)*1j)/NSUB**2
                    #     break
                    # break

                # LINE 344 - Pivot if required so that the fundamental is always in column 1

                nfund = 1 + NSUB*nvin - nuin
                print('Fundamental', nfund)
                temp = phases[0, nim-1]
                # print(nfund, nim, nvin, nuin)
                phases[0, nim-1] = phases[nfund-1, nim-1]
                phases[nfund-1, nim-1] = temp

            # print(iy, ix, phases[:4, :4].real)
            # vec[:4, :4] = la.inv(phases[:4, :4])
            # print(vec.shape, phases.shape)
            # plt.imshow(phases[:9, :9].real)

            # LINE 355 - This loads an identity matrix, which will be used to invert the phase matrix.
            
            for i in range(1, NSUB**2+1): 
                for j in range(1, NSUB**2+1):
                    vec[j-1, i-1] = 0+0j
                vec[i-1, i-1] = 1+0j
                key[i-1] = i
            
            # vec[:4, :4] = la.inv(phases[:4, :4])

            ### BEGIN MATRIX INVERSION

            # LINE 367 - If N>nsub2, then the problem is over determined

            # LINE 371 - The weighting factor is used at this point.

            if npp>NSUB**2: 
                # print('npp>NSUB**2')
                for i in range(1, NSUB**2+1): 
                    for j in range(1, NSUB**2+1): 
                        phasem[j-1, i-1] = 0+0j
                        for k in range(1, npp+1): 
                            phasem[j-1, i-1] += np.conj(phases[i-1, k-1])*wt[k-1]*phases[j-1, k-1]
            else: 
                for i in range(1, NSUB**2+1): 
                    for j in range(1, NSUB**2+1): 
                        phasem[j-1, i-1] = phases[j-1, i-1]
            # print(phasem==0)
            # print(phasem.shape, phases.shape, vec.shape)

            # LINE 391 - solve for the data vector phases

            for i in range(1, NSUB**2-1+1): 
                for j in range(i+1, NSUB**2+1): 

                    # LINE 396 - Check for zero division and pivot if required.

                    # print(phasem)#, la.inv(phasem))
                    # print(i, phasem[i, i]*np.conj(phasem[i, i])==0)
                    if (phasem[i-1, i-1]*np.conj(phasem[i-1, i-1])==0): 
                        pivot = False
                        k = i+1
                        while (not pivot) and (k<=NSUB**2): 
                            if (phasem[i-1, k-1]*np.conj(phasem[i-1, k-1])!=0): 
                                
                                pivot = True
                                itemp = key[i-1]
                                key[i-1] = key[k-1]
                                key[k-1] = itemp
                                for kk in range(1, NSUB**2+1): 
                                    temp = phasem[kk-1, i-1]
                                    phasem[kk-1, i-1] = phasem[kk-1, k-1]
                                    phasem[kk-1, k-1] = temp
                                    temp = vec[kk-1, i-1]
                                    vec[kk-1, i-1] = vec[kk-1, k-1]
                                    vec[kk-1, k-1] = temp
                            else: 
                                k += 1

                        if not pivot: 
                            raise ValueError('singular phase matrix')
                    
                    # if phasem[i-1, i-1]==0: return

                    # LINE 436 - Any pivoting required is now completed

                    # print(i, j, 'phasem', phasem[i-1, i-1])
                    # print(phasem[:4, :4])
                    rat = phasem[i-1, j-1]/phasem[i-1, i-1]
                    for k in range(i, NSUB**2+1): 
                        phasem[k-1, j-1] -= rat*phasem[k-1, i-1]
                    for k in range(1, NSUB**2+1): 
                        vec[k-1, j-1] -= rat*vec[k-1, i-1]
                    # print(i, phasem[i, i])
                    # print(phasem)
                # return
            
            for i in range(NSUB**2, 1, -1):
                rat = phasem[i-1, i-1]
                for j in range(1, NSUB**2+1):
                    # print('rat', rat, i, phasem[i-1, i-1])
                    # print(phasem)
                    # print(phases[:4, :4]==0)
                    vec[j-1, i-1] /= rat
                for j in range(i-1, 0, -1):
                    rat = phasem[i-1, j-1]
                    for k in range(1, NSUB**2+1):
                        vec[k-1, j-1] -= rat*vec[k-1, i-1]
            for j in range(1, NSUB**2+1):
                vec[j-1, 0] /= phasem[0, 0]

            # LINE 467 - The vec array now holds the inverse of the original phasem array.

            ### END MATRIX INVERSION

            # LINE 469 - If any pivoting has been done, undo it.

            for i in range(1, NSUB**2+1):
                if key[i-1]!=i: 
                    k = i+1
                    # print(key)
                    while (key[k-1]!=i) and (k<NSUB**2): 
                        k += 1
                    for kk in range(1, NSUB**2+1):
                        temp = vec[kk-1, i-1]
                        vec[kk-1, i-1] = vec[kk-1, k-1]
                        vec[kk-1, k-1] = temp
                    key[k-1] = key[i-1]

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
            print(f'Image {npos}, power {coef[isec-1]}, sector {isec}')
        #     break
        # break

    # print('coef', coef)
    # COEFS.append(coef)

    # LINE 516 - apply the complex scale factor to the transform

    isec = 0
    isv = nrow//2
    iev = isv - nrow//NSUB + 1 
    # print(A)
    # print('phix', phix)
    # print('phiy', phiy)
    # print(isv, iev)
    # print(nsy)
    # print(nsx)
    # print(isv-(iev-1))
    for iy in range(isy, isy+nsy):
        ieu = ncol - (NSUB - 1)*(nsx - 1)*ncol//NSUB
        # print(ieu) # TODO: verify that ieu==ncol if nsx = 1
        isu = 1
        for ix in range(0, nsx):
            isec += 1
            # print(isec, coef[isec-1])
            spr = coef[isec-1].real # NOTE Fortran casts Complex to Real directly
            spi = coef[isec-1].imag
            # print('this coef', coef[isec-1])
            # COEFS.append(coef[isec-1])
            for vrow in range(isv, iev-1, -1): 
                # print('vrow', vrow, 'isv', isv, 'iev', iev)
                if vrow>0:
                    row = vrow
                else: 
                    row = nrow + vrow
                if row>nrow//2: 
                    V = (row - nrow - 1)/nrow
                else: 
                    V = (row - 1)/nrow
                rphase = -2*phiy[npos-1]*V
                # rphase = rphase/2-1
                # print(rphase)
                rpr = np.cos(rphase)
                rpi = np.sin(rphase)
                ypr = rpr*spr - rpi*spi
                ypi = rpi*spr + rpr*spi
                # print(rphase, rpr, rpi, spr, spi, ypr, ypi, rpr*spr)
                # print((ieu+1-isu)//2)
                for col in range(isu, ieu+1, 2):
                    # print(row, isv, iev, col, isu, ieu)
                    U = (col - 1)/(ncol - 2)/2
                    cphase = -2*phix[npos-1]*U
                    # print(U, V)
                    # print(rphase, cphase, phiy[npos-1], phiy[npos-1])
                    # print(' ', cphase)
                    # print(rphase, cphase)
                    cpr = np.cos(cphase)
                    cpi = np.sin(cphase)
                    tpr = ypr*cpr - ypi*cpi
                    tpi = ypi*cpr + ypr*cpi
                    # print(vrow, row, col, isu, ieu+2)
                    fr = A[col-1, row-1]
                    fi = A[col+1-1, row-1]
                    # print('ypr', ypr, cpr, ypr*cpr, ypi, cpi, ypr*cpr - ypi*cpi, tpr,tpi)
                    # print()
                    A[col-1, row-1] = fr*tpr - fi*tpi
                    A[col+1-1, row-1] = fi*tpr + fr*tpi
            isu = ieu + 1
            ieu = ncol - (nsx - 2 - ix)*ncol//NSUB # TODO: check values
        isv = iev - 1
        iev = isv - nrow//NSUB + 1 # TODO: check values
        if iy==(isy + nsy - 2): 
            iev = -(nrow//2) + 1 # NOTE: add a bracket to change negative sign to minus sign

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
    return A

def four1(data, nn, isign):
    """
    Perform a 1D complex-to-complex Fast Fourier Transform (FFT).

    Parameters
    ----------
    data : ndarray
        Input array of length ``2 * nn``, alternating real and imaginary components.
    nn : int
        Number of complex elements in the data.
    isign : int
        Sign of the exponent in the FFT; ``1`` for forward transform, ``-1`` for inverse transform.

    Returns
    -------
    ndarray
        Transformed data array of the same shape as the input.

    References
    ----------
    W.H. Press et al., "Numerical Recipes" (JJGG).
    """
    # print('four1', data.size, nn, data)
    n = 2*nn
    data = data.copy()
    # data = np.concatenate((data, np.zeros(10)))
    j = 1
    for i in range(1, n+1, 2):
        if j > i:
            data[j-1], data[i-1] = data[i-1], data[j-1]
            data[j+1-1], data[i+1-1] = data[i+1-1], data[j+1-1]
        
        m = n//2
        while m>=2 and j>m:
            j -= m
            m //= 2
        j += m
    
    # FFT computation
    mmax = 2
    while n>mmax:
        istep = 2*mmax
        theta = 2*np.pi/(isign*mmax)
        wpr = -2.0*np.sin(0.5*theta)**2
        wpi = np.sin(theta)
        wr = 1.0
        wi = 0.0
        for m in range(1, mmax+1, 2):
            for i in range(m, n+1, istep):
                j = i + mmax
                # print(i, j, mmax, nn, n, istep, len(data))
                tempr = wr*data[j-1] - wi*data[j]
                tempi = wr*data[j] + wi*data[j-1]
                data[j-1]  = data[i-1] - tempr
                data[j] = data[i] - tempi
                data[i-1] += tempr
                data[i] += tempi
            wtemp = wr
            wr = wr*wpr - wi*wpi + wr
            wi = wi*wpr + wtemp*wpi + wi
        
        mmax = istep
    # return data[:-10]
    return data

def realft(data, n, isign):
    """
    Perform a real-to-complex or complex-to-real FFT transformation.

    Parameters
    ----------
    data : ndarray
        Input array of length `2 * n`.
    n : int
        Number of real elements in the data.
    isign : int
        Sign of the exponent in the FFT; `1` for forward transform, `-1` for inverse transform.

    Returns
    -------
    ndarray
        Transformed data array of the same shape as the input.

    References
    ----------
    W.H. Press et al., "Numerical Recipes" (JJGG).
    """
    # print('realft', data.size, n)
    theta = 2*np.pi/(2*n)
    data = data.copy()
    wr = 1.0
    wi = 0.0
    c1 = 0.5
    if isign == 1:
        c2 = -0.5
        # print(data, n)
        data = four1(data, n, 1)
        # print(data)
        data[2*n] = data[0]
        data[2*n+1] = data[1]
    else:
        c2 = 0.5
        theta = -theta
    wpr = -2.0*np.sin(0.5*theta)**2
    wpi = np.sin(theta)
    n2p3 = 2*n + 3
    for i in range(1, n//2+2):
        i1 = 2*i - 1
        i2 = i1 + 1
        i3 = n2p3 - i2
        i4 = i3 + 1
        wrs = wr
        wis = wi
        h1r = c1*(data[i1-1] + data[i3-1])
        h1i = c1*(data[i2-1] - data[i4-1])
        h2r = -c2*(data[i2-1] + data[i4-1])
        h2i = c2*(data[i1-1] - data[i3-1])
        data[i1-1] = h1r + wrs*h2r - wis*h2i
        data[i2-1] = h1i + wrs*h2i + wis*h2r
        data[i3-1] = h1r - wrs*h2r + wis*h2i
        data[i4-1] = -h1i + wrs*h2i + wis*h2r
        wtemp = wr
        wr = wr*wpr - wi*wpi + wr
        wi = wi*wpr + wtemp*wpi + wi
    if isign == -1:
        data = four1(data, n, -1)
    return data

def real2dfft(a, nra, nca, b, nrb, ncb, isign, work, onedim=False):
    """
    Perform a 2D real-to-complex or complex-to-real FFT transformation.

    Parameters
    ----------
    a : ndarray
        Input/output array for the transformation.
    nra : int
        Number of rows in the input array.
    nca : int
        Number of columns in the input array.
    b : ndarray
        Auxiliary array for intermediate transformations.
    nrb : int
        Number of rows in the auxiliary array.
    ncb : int
        Number of columns in the auxiliary array.
    isign : int
        Sign of the exponent in the FFT; `1` for forward transform, `-1` for inverse transform.
    work : ndarray
        Temporary work array for intermediate calculations.
    onedim : bool, optional
        Whether to perform a 1D FFT along the first dimension only. Default is False.

    Returns
    -------
    ndarray
        Transformed data array.

    References
    ----------
    W.H. Press et al., "Numerical Recipes" (JJGG).
    """
    a = a.copy()
    b = b.copy()
    work = np.zeros((2, max(nra, nrb)))
    if isign >= 0:
        # print(a.shape, b.shape, nrb, ncb, a[:6, :nrb].shape)
        a[:ncb, :nrb] = b
        a[ncb:, :] = 0
        nhc = (nca - 2)//2
        # print(nhc)
        for ir in range(nrb):
            a[:, ir] = realft(a[:, ir], nhc, isign)
        # print(a)
        # print(a[0, :].size, nhc)
        # a[0, :] = realft(a[0, :], nhc, isign)
        if onedim:
            return a
        for ic in range(1, nca+1, 2):
            # print('ic', ic, a[ic+1-1, :nrb])
            # print(a)
            work[0, :nrb] = a[ic-1, :nrb]
            work[1, :nrb] = a[ic+1-1, :nrb]
            work[:, nrb:] = 0
            # print('work', 'nra', nra)
            work = four1(work.T.flatten(), nra, isign).reshape(-1, 2).T
            # print(work)
            a[ic-1, :] = work[0, :]
            a[ic+1-1, :] = work[1, :]
    else:
        if not onedim:
            for ic in range(1, ncb+1, 2):
                work[0, :nrb] = b[ic-1, :nrb]
                work[1, :nrb] = b[ic+1-1, :nrb]
                # print('work2', 'nra', nra)
                work = four1(work.T.flatten(), nra, isign).reshape(-1, 2).T
                b[ic-1, :] = work[0, :]
                b[ic+1-1, :] = work[1, :]
        nhc = (ncb - 2)//2
        for ir in range(nra):
            b[:, ir] = realft(b[:, ir], nhc, isign)
        if onedim:
            tmp = 1.0/nhc
        else:
            tmp = 1.0/(nrb*nhc)
        a[:nca, :nra] = b[:nca, :nra]*tmp
    return a