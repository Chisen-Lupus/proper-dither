import os, sys
import numpy as np

def four1_forward(a): 
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


### NOTE: TEST CODE

PI = np.pi
RADIOAN = PI/180
NSTMAX = 140
NDIV = 3

from .wrapper import NR_FREQ, NC_FREQ
from .wrapper import set_padding_size
NSUB = 2

NC, NR = 2**9+2, 2**9
set_padding_size(NC, NR)

def phase_updated(A_in, offsets, npos):
    A = A_in.copy()
    phix = np.zeros(NSTMAX)
    phasem = np.zeros((NDIV**2, NDIV**2), dtype=np.complex128)
    vec = np.zeros((NDIV**2, NDIV**2), dtype=np.complex128)
    coef = np.zeros((NDIV**2), dtype=np.complex128)
    phases = np.zeros((NDIV**2, NSTMAX), dtype=np.complex128)
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

            for nim in range(npp):
                px = -phix[nim]*2/NSUB
                py = -phiy[nim]*2/NSUB
                nuin = ix - (NSUB - 1)//2
                nvin = iy
                pxi = nuin*px
                pyi = -nvin*py

                isat = 0
                for isaty in range(0, NSUB):
                    for isatx in range(0, NSUB): 
                        isat += 1
                        phit = isatx*px + pxi + isaty*py + pyi
                        phases[isat-1, nim] = (np.cos(phit) + np.sin(phit)*1j)/NSUB**2

                # LINE 344 - Pivot if required so that the fundamental is always in column 1

                nfund = 1 + NSUB*nvin - nuin
                print('Fundamental', nfund)
                temp = phases[0, nim]
                phases[0, nim] = phases[nfund-1, nim]
                phases[nfund-1, nim] = temp

            # LINE 355 - This loads an identity matrix, which will be used to invert the phase matrix.
            
            vec = np.eye(NDIV**2, dtype=np.complex128)
            key = np.arange(NSUB**2) + 1
            
            # print(phases)
            # vec[:9, :4] = la.pinv(phases[:4, :9])

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

            # LINE 391 - solve for the data vector phases

            for i in range(1, NSUB**2-1+1): 
                for j in range(i+1, NSUB**2+1): 

                    # LINE 396 - Check for zero division and pivot if required.

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

                    # LINE 436 - Any pivoting required is now completed

                    rat = phasem[i-1, j-1]/phasem[i-1, i-1]
                    for k in range(i, NSUB**2+1): 
                        phasem[k-1, j-1] -= rat*phasem[k-1, i-1]
                    for k in range(1, NSUB**2+1): 
                        vec[k-1, j-1] -= rat*vec[k-1, i-1]
            
            for i in range(NSUB**2, 1, -1):
                rat = phasem[i-1, i-1]
                for j in range(1, NSUB**2+1):
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
                    while (key[k-1]!=i) and (k<NSUB**2): 
                        k += 1
                    for kk in range(1, NSUB**2+1):
                        temp = vec[kk-1, i-1]
                        vec[kk-1, i-1] = vec[kk-1, k-1]
                        vec[kk-1, k-1] = temp
                    key[k-1] = key[i-1]

            # LINE 490 - For NSUB2 images, we are done
            
            if npp==NSUB**2:
                coef[isec-1] = vec[npos-1, 0]

            # LINE 495 - Otherwise, we need to do a little more work.  Here we just solve for the fundamental image.
            
            else: 
                coef[isec-1] = 0
                for i in range(1, NSUB**2+1):
                    coef[isec-1] += vec[i-1, 0]*np.conj(phases[i-1, npos-1])

            # LINE 505 - Addin weighting factor

            coef[isec-1] *= wt[npos-1]

            print(f'Image {npos}, power {coef[isec-1]}, sector {isec}')

    # LINE 516 - apply the complex scale factor to the transform
    # plt.imshow(dutils.norm(A))
    isec = 0
    isv = NR_FREQ//2
    iev = isv - NR_FREQ//NSUB + 1 

    for iy in range(isy, isy+nsy): # 2 times
        ieu = NC_FREQ - (NSUB - 1)*(nsx - 1)*NC_FREQ//NSUB
        isu = 1
        for ix in range(0, nsx): # 1 times
            isec += 1
            coefficient = coef[isec-1]
            ROW = np.arange(iev-1, isv)
            V = ROW/NR_FREQ
            if isv<=0: 
                ROW += NR_FREQ
            print(iev-1, isv)
            rphase = -2*phiy[npos-1]*V
            COL = np.arange(isu-1, ieu, 2)
            U = COL/(NC_FREQ - 2)/2
            cphase = -2*phix[npos-1]*U
            fr = A[COL, ROW[0]:ROW[-1]+1]
            fi = A[COL+1, ROW[0]:ROW[-1]+1]
            f = fr + fi*1j
            RPHASE, CPHASE = np.meshgrid(rphase, cphase)
            factor = coefficient*np.exp(1j*(RPHASE + CPHASE))
            f_updated = f*factor
            A[COL, ROW[0]:ROW[-1]+1] = f_updated.real
            A[COL+1, ROW[0]:ROW[-1]+1] = f_updated.imag
            isu = ieu + 1
            ieu = NC_FREQ - (nsx - 2 - ix)*NC_FREQ//NSUB 
        isv = iev - 1
        iev = isv - NR_FREQ//NSUB + 1
        if iy==(isy + nsy - 2): 
            iev = -(NR_FREQ//2) + 1

    return A

def phase_shift(A, offsets, n, verbose=False):
    if not verbose: 
        stdout_backup = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try: 
            Aphased = phase_updated(A, offsets, n)
        except Exception as e: 
            print(repr(e))
        finally:
            sys.stdout = stdout_backup
    else: 
        Aphased = phase_updated(A, offsets, n)
    return Aphased
