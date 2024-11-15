import numpy as np

from astropy.wcs import WCS
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve, Box2DKernel

from scipy.ndimage import label

from photutils.detection import DAOStarFinder
from photutils.centroids import centroid_com

# TODO: naming convention of the functions
# TODO: delete comments and add documentations
# TODO: regulate variable names
# TODO: 80 columns rule

def norm(z): 
    norm = np.sqrt(z.real**2 + z.imag**2)
    return norm

def get_power_spectrum(fft_result):
    fft_shifted = np.fft.fftshift(fft_result)
    power_spectrum = np.abs(fft_shifted)**2
    return power_spectrum

def get_pixel_center_coordinate(fits_path):
    with fits.open(fits_path) as hdul:
        wcs = WCS(hdul['SCI'].header)
        ny, nx = hdul['SCI'].data.shape
        center_x = (nx - 1) / 2
        center_y = (ny - 1) / 2
        sky_coord = wcs.pixel_to_world(center_x, center_y)
        return center_x, center_y, sky_coord

def get_pixel_center_from_array(data):
    nx, ny = data.shape
    center_x = nx//2 + 1
    center_y = ny//2 + 1
    return center_x, center_y

def get_power_spectrum_from_realfft2d(Atotal):
    nx, ny = Atotal.shape
    a1 = Atotal[:, 0:nx//2-1:]
    a2 = Atotal[:, -1:nx//2-2:-1]
    a = a1**2+a2**2
    r1 = a[::2][::-1]
    r2 = a[1::2]
    r = np.vstack((r1, r2))
    rr = np.hstack((r[:, ::-1], r))
    return rr

def create_cutout_fits(fits_path, cutout_path, coord, radius):
    '''
    Parameters: 
        fits_path:      string
        output_path:    string
        coord:          SkyCoord
        radius:         float (in arcsec)
    '''
    with fits.open(fits_path) as hdul:
        # read file
        header = hdul['SCI'].header
        # wcs.sip = None

        wcs = WCS(header)
        data = hdul['SCI'].data
        pixel_scale = wcs.proj_plane_pixel_scales()[0].value # degrees per pixel
        radius_pix = int(radius/3600/pixel_scale)
        size = (radius_pix*2, radius_pix*2)

        # Create the cutout
        cutout = Cutout2D(data, coord, size, wcs=wcs)
        hdul['SCI'].header.update(cutout.wcs.to_header())

        # Create new FITS HDUs
        new_hdul = fits.HDUList([fits.PrimaryHDU(header=hdul[0].header)])
        for hdu_name in ['SCI', 'ERR', 'DQ', 'AREA', 'VAR_POISSON', 'VAR_RNOISE', 'VAR_FLAT']:
            if hdu_name in hdul:
                hdu_data = Cutout2D(hdul[hdu_name].data, coord, size, wcs=wcs)
                new_hdu = fits.ImageHDU(data=hdu_data.data, header=hdul[hdu_name].header, name=hdu_name)
                new_hdu.header.update(hdu_data.wcs.to_header())
                new_hdul.append(new_hdu)

        # Write the new cutout to a FITS file
        new_hdul.writeto(cutout_path, overwrite=False)
        print(f"Cutout saved to {cutout_path}")

def get_centroids_using_DAOStarFinder(data, center_x, center_y):
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=3.0, threshold=20.0*std)
    sources = daofind(data - median)
    # print(len(sources))
    distances_to_center = (sources['xcentroid'] - center_x)**2 + (sources['ycentroid'] - center_y)**2
    source = sources[distances_to_center==np.min(distances_to_center)][0]
    x = source['xcentroid']
    y = source['ycentroid']
    return x, y

def get_centroids_using_centroid_com(data, center_x, center_y, r):
    centroid = centroid_com(data[center_x-1-r:center_x+r, center_y-1-r:center_y+r])
    x = centroid[0] + center_x - 1 - r
    y = centroid[1] + center_y - 1 - r
    return x, y

def get_brightest_pixel(data, xmin=None, xmax=None, ymin=None, ymax=None): 
    if not xmin: xmin=0
    if not ymin: ymin=0
    if not xmax: xmax=data.shape[0]

    if not ymax: ymax=data.shape[1]
    data_slice = data[xmin:xmax, ymin:ymax]
    y, x = np.where(data_slice==np.max(data_slice))
    return [x[0]+xmin, y[0]+ymin]

def calculate_padding_radius(centroids_int):
    centerx_max = np.max(centroids_int[:, 0])
    centerx_min = np.min(centroids_int[:, 0])
    centery_max = np.max(centroids_int[:, 1])
    centery_min = np.min(centroids_int[:, 1])
    pad = np.max([centerx_max-centerx_min, centery_max-centery_min]) 
    return pad

def pad_image_with_centroid(data, pad, centroid_int):
    nx, ny = data.shape
    nx_aligned = nx + pad
    ny_aligned = ny + pad
    aligned_data = np.zeros((nx_aligned, ny_aligned))
    dx = nx_aligned//2 - centroid_int[1] - 1
    dy = nx_aligned//2 - centroid_int[0] - 1
    aligned_data[dx:dx+nx, dy:dy+ny] = data
    return aligned_data

def get_cosmic_ray_mask_without_AGN(image_data, kernel_size=3, sigma_threshold=5, max_connected_pixels=12):
    """
    Detect cosmic rays in an image based on pixel differences from a smoothed version,
    while ignoring regions connected by more than a certain number of pixels.

    Parameters:
    -----------
    image_data : 2D numpy array
        The input image data.
    kernel_size : int, optional
        Size of the box kernel for smoothing (default is 3x3).
    sigma_threshold : float, optional
        Threshold for detecting cosmic rays in units of standard deviation (default is 5).
    max_connected_pixels : int, optional
        Maximum number of connected pixels allowed for a region to be considered a cosmic ray (default is 10).

    Returns:
    --------
    cosmic_ray_mask : 2D numpy array
        A boolean mask where cosmic rays are detected (True means cosmic ray).
    """

    mean, median, sigma = sigma_clipped_stats(image_data, sigma=3)
    # Define a smoothing kernel
    kernel = Box2DKernel(kernel_size)
    
    # Convolve the image with the kernel to create a smoothed version
    smoothed_image = convolve(image_data, kernel)
    
    # Calculate the difference between the original and smoothed image
    diff = np.abs(image_data - smoothed_image)
    
    # Define the threshold for cosmic ray detection
    # print(mean, median, sigma, np.std(image_data))
    threshold = sigma_threshold * sigma# np.std(image_data)
    
    # Create a mask where the difference exceeds the threshold
    cosmic_ray_mask = diff > threshold
    
    # Label connected components in the mask
    labeled_array, num_features = label(cosmic_ray_mask)
    
    # Iterate over each connected component and check the number of pixels
    for label_num in range(1, num_features + 1):
        region_size = np.sum(labeled_array == label_num)
        if region_size > max_connected_pixels:
            # If a connected region has more than max_connected_pixels, remove it from the mask
            cosmic_ray_mask[labeled_array == label_num] = False
    
    return cosmic_ray_mask