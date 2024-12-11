"""
This module provides utility functions for data processing and other modules in the package.
The utilities include power spectrum calculations, centroid determination, FITS file manipulation,
and image padding.

Key Features:
-------------
- Power spectrum computation from FFT results.
- Centroid determination using DAOStarFinder and centroid_com.
- FITS file handling for creating cutouts with WCS adjustments.
- Cosmic ray detection using a sigma-clipped approach.
- Image extraction and padding utilities.

Dependencies:
-------------
- numpy
- astropy
- photutils
- scipy

Example Usage:
--------------
.. code-block:: python

    from utils import get_power_spectrum, create_cutout_fits, get_centroids_using_DAOStarFinder

    # Compute power spectrum
    power_spectrum = get_power_spectrum(fft_result)

    # Create a FITS cutout
    create_cutout_fits('input.fits', 'cutout.fits', coord, radius=30.0)

    # Find centroids using DAOStarFinder
    x, y = get_centroids_using_DAOStarFinder(data, center_x, center_y)
"""

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
    """
    Compute the magnitude of a complex array.

    Parameters
    ----------
    z : ndarray
        Input complex array.

    Returns
    -------
    ndarray
        Magnitude of the complex array.
    """
    norm = np.sqrt(z.real**2 + z.imag**2)
    return norm

def get_power_spectrum(fft_result):
    """
    Compute the power spectrum from a Fourier-transformed array.

    Parameters
    ----------
    fft_result : ndarray
        Fourier-transformed input array.

    Returns
    -------
    ndarray
        Power spectrum of the input array.
    """
    fft_shifted = np.fft.fftshift(fft_result)
    power_spectrum = np.abs(fft_shifted)**2
    return power_spectrum

def get_pixel_center_coordinate(fits_path):
    """
    Get the pixel and sky coordinates of the center of a FITS image.

    Parameters
    ----------
    fits_path : str
        Path to the FITS file.

    Returns
    -------
    tuple
        Center x and y pixel coordinates, and sky coordinates.
    """
    with fits.open(fits_path) as hdul:
        wcs = WCS(hdul['SCI'].header)
        ny, nx = hdul['SCI'].data.shape
        center_x = (nx - 1) / 2
        center_y = (ny - 1) / 2
        sky_coord = wcs.pixel_to_world(center_x, center_y)
        return center_x, center_y, sky_coord

def get_pixel_center_from_array(data):
    """
    Calculate the center pixel coordinates of a 2D array.

    Parameters
    ----------
    data : ndarray
        Input 2D array.

    Returns
    -------
    tuple
        Center x and y pixel coordinates.
    """
    nx, ny = data.shape
    center_x = nx//2 + 1
    center_y = ny//2 + 1
    return center_x, center_y

def get_power_spectrum_from_realfft2d(Atotal):
    """
    Compute the power spectrum from a 2D real FFT result.

    Parameters
    ----------
    Atotal : ndarray
        Input 2D array from a real FFT.

    Returns
    -------
    ndarray
        Power spectrum of the input array.
    """
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
    """
    Create a cutout from a FITS file and save it as a new FITS file.

    Parameters
    ----------
    fits_path : str
        Path to the input FITS file.
    cutout_path : str
        Path to save the cutout FITS file.
    coord : SkyCoord
        Center coordinate of the cutout.
    radius : float
        Radius of the cutout in arcseconds.

    Returns
    -------
    None
    """
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
    """
    Find centroids of stars using DAOStarFinder near a specified center.

    Parameters
    ----------
    data : ndarray
        Input 2D image data.
    center_x : float
        X-coordinate of the center.
    center_y : float
        Y-coordinate of the center.

    Returns
    -------
    tuple
        X and Y coordinates of the centroid closest to the specified center.
    """
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
    """
    Compute the centroid using the center-of-mass method within a specified radius.

    Parameters
    ----------
    data : ndarray
        Input 2D image data.
    center_x : int
        X-coordinate of the center.
    center_y : int
        Y-coordinate of the center.
    r : int
        Radius for the centroid calculation.

    Returns
    -------
    tuple
        X and Y coordinates of the centroid.
    """
    centroid = centroid_com(data[center_x-1-r:center_x+r, center_y-1-r:center_y+r])
    x = centroid[0] + center_x - 1 - r
    y = centroid[1] + center_y - 1 - r
    return x, y

def get_brightest_pixel(data, xmin=None, xmax=None, ymin=None, ymax=None): 
    """
    Find the brightest pixel in a specified region of the image.

    Parameters
    ----------
    data : ndarray
        Input 2D image data.
    xmin : int, optional
        Minimum x-coordinate of the region. Default is 0.
    xmax : int, optional
        Maximum x-coordinate of the region. Default is the image width.
    ymin : int, optional
        Minimum y-coordinate of the region. Default is 0.
    ymax : int, optional
        Maximum y-coordinate of the region. Default is the image height.

    Returns
    -------
    list
        X and Y coordinates of the brightest pixel.
    """
    if not xmin: xmin=0
    if not ymin: ymin=0
    if not xmax: xmax=data.shape[0]
    if not ymax: ymax=data.shape[1]
    data_slice = data[xmin:xmax, ymin:ymax]
    y, x = np.where(data_slice==np.max(data_slice))
    return [x[0]+xmin, y[0]+ymin]

def calculate_padding_radius(centroids_int):
    """
    Calculate the padding radius based on centroid coordinates.

    Parameters
    ----------
    centroids_int : ndarray
        Array of centroid coordinates.

    Returns
    -------
    float
        Padding radius.
    """
    centerx_max = np.max(centroids_int[:, 0])
    centerx_min = np.min(centroids_int[:, 0])
    centery_max = np.max(centroids_int[:, 1])
    centery_min = np.min(centroids_int[:, 1])
    pad = np.max([centerx_max-centerx_min, centery_max-centery_min]) 
    return pad

def pad_image_with_centroid(data, pad, centroid_int):
    """
    Pad an image around a specified centroid.

    Parameters
    ----------
    data : ndarray
        Input 2D image data.
    pad : int
        Padding size.
    centroid_int : tuple
        Integer centroid coordinates (x, y).

    Returns
    -------
    ndarray
        Padded image.
    """
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

    Parameters
    ----------
    image_data : ndarray
        The input image data.
    kernel_size : int, optional
        Size of the box kernel for smoothing (default is 3x3).
    sigma_threshold : float, optional
        Threshold for detecting cosmic rays in units of standard deviation (default is 5).
    max_connected_pixels : int, optional
        Maximum number of connected pixels allowed for a region to be considered a cosmic ray (default is 12).

    Returns
    -------
    ndarray
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

def extract_central_region(image, fraction=0.1, radius=None, center=None):
    """
    Extract a central region from an image.

    Parameters
    ----------
    image : ndarray
        Input 2D image data.
    fraction : float, optional
        Fraction of the image size to extract (default is 0.1).
    radius : int, optional
        Radius of the region to extract. Overrides ``fraction`` if provided.
    center : tuple, optional
        Center of the region to extract. Default is the image center.

    Returns
    -------
    ndarray
        Extracted central region with NaN padding for out-of-bound areas.
    """
    # Determine the shape and center of the image
    image_shape = np.array(image.shape)
    if center is None:
        center = image_shape // 2
    center_x, center_y = int(center[0]), int(center[1])

    # Determine the half-widths based on radius or fraction
    if radius is not None:
        half_width_x, half_width_y = int(radius), int(radius)
    else:
        half_width_x, half_width_y = (image_shape * fraction // 2).astype(int)

    # Define the slice ranges, padding with NaN if needed
    start_x, end_x = center_x - half_width_x, center_x + half_width_x + 1
    start_y, end_y = center_y - half_width_y, center_y + half_width_y + 1

    # Initialize the output array filled with NaN
    output_shape = (2 * half_width_x + 1, 2 * half_width_y + 1)
    cutout = np.full(output_shape, np.nan, dtype=image.dtype)

    # Determine the intersection between the image and the cutout
    src_start_x = max(0, start_x)
    src_end_x = min(image.shape[0], end_x)
    src_start_y = max(0, start_y)
    src_end_y = min(image.shape[1], end_y)

    dest_start_x = max(0, -start_x)
    dest_end_x = dest_start_x + (src_end_x - src_start_x)
    dest_start_y = max(0, -start_y)
    dest_end_y = dest_start_y + (src_end_y - src_start_y)

    # Copy the valid region of the image into the cutout
    cutout[
        dest_start_x:dest_end_x,
        dest_start_y:dest_end_y
    ] = image[
        src_start_x:src_end_x,
        src_start_y:src_end_y
    ]

    return cutout
