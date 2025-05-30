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

import copy

import numpy as np

from astropy.wcs import WCS
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve, Box2DKernel
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.modeling.models import Shift
from astropy.modeling import models
import astropy.units as u

from scipy.ndimage import label, zoom, gaussian_filter, shift
from scipy.optimize import minimize
from scipy.signal import convolve2d

from photutils.detection import DAOStarFinder
from photutils.centroids import centroid_com

from jwst.datamodels import ImageModel

from webbpsf import NIRCam

from typing import Any

import gwcs

from .process import combine_image

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
    center_x = (nx - 1) // 2  # left center for even, center for odd
    center_y = (ny - 1) // 2
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

def get_centroids_using_DAOStarFinder(data, center_x, center_y, radius=30, snr=20, 
                                      return_flux=False):
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
    radius : float, optional
        Searching radius (default is 30 pixels).
    snr : float, optional
        Minimum SNR needed for the star finder.

    Returns
    -------
    tuple
        X and Y coordinates of the brightest centroid within the specified radius.
    """
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=3.0, threshold=snr*std)
    # sources = daofind(data - median)
    sources = daofind(data)
    if len(sources) == 0:
        raise ValueError("No sources found within the specified SNR.") 
    # Compute distances to the center
    distances = np.sqrt((sources['xcentroid'] - center_x)**2 + (sources['ycentroid'] - center_y)**2)
    # Filter sources within the given radius
    mask = distances<=radius
    filtered_sources = sources[mask]
    # print(filtered_sources)
    if len(filtered_sources) == 0:
        raise ValueError("No sources found within the specified radius.")  
    # Find the brightest source within the radius
    brightest_source = filtered_sources[np.argmax(filtered_sources['flux'])]
    if return_flux: 
        return np.array([brightest_source['xcentroid'], brightest_source['ycentroid']]), sources, brightest_source['flux']
    else:  
        return np.array([brightest_source['xcentroid'], brightest_source['ycentroid']]), sources


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
    x, y = np.where(data_slice==np.nanmax(data_slice))
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

def crop_image_with_centroid(data, radius, centroid_int):
    """
    Crop an image around a specified centroid.

    Parameters
    ----------
    data : ndarray
        Input 2D image data.
    radius : int
        Cropping radius.
    centroid_int : tuple
        Integer centroid coordinates (x, y).

    Returns
    -------
    ndarray
        Cropped image.
    """
    x, y = centroid_int
    x_min = max(0, x - radius)
    x_max = min(data.shape[0], x + radius)
    y_min = max(0, y - radius)
    y_max = min(data.shape[1], y + radius)
    
    return data[x_min:x_max, y_min:y_max]

def get_cosmic_ray_mask_without_AGN(image_data, kernel_size=3, sigma_threshold=5, max_connected_pixels=12, avoid=None):
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

    # avoid region
    xmin, xmax, ymin, ymax = avoid
    cosmic_ray_mask[xmin:xmax, ymin:ymax] = False
    
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


def create_jwst_cutout_fits(fits_path, cutout_path, coord, radius):
    """
    Create a cutout from a JWST ImageModel FITS file and save it as a new FITS file.

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
    with ImageModel(fits_path) as model:
        # Extract WCS from JWST ImageModel
        original_wcs = model.meta.wcs  

        # Convert coordinate to pixel values
        x, y = original_wcs.world_to_array_index(coord)

        # Get pixel scale and compute cutout size
        pixel_scale = 0.063 if model.meta.instrument.channel=='LONG' else 0.031
        radius_pix = int(radius / pixel_scale)
        size = (radius_pix * 2, radius_pix * 2)    
        cutout_x, cutout_y = x, y  # Cutout center
        cutout_size_x, cutout_size_y = size  # Cutout size

        # Get the original bounding box (pixel range)
        (x_min, x_max), (y_min, y_max) = original_wcs.bounding_box
        # print((x_min, x_max), (y_min, y_max))
        # print(x, y)



        # Define new bounding box for the cutout
        new_x_min = max(x_min, cutout_x - cutout_size_x // 2 - 0.5)
        new_x_max = min(x_max, cutout_x + cutout_size_x // 2 - 0.5)
        new_y_min = max(y_min, cutout_y - cutout_size_y // 2 - 0.5)
        new_y_max = min(y_max, cutout_y + cutout_size_y // 2 - 0.5)
        # print([int(new_y_min+0.5),int(new_y_max+0.5), int(new_x_min+0.5),int(new_x_max+0.5)])

        # Create the new WCS    
        detector_frame = original_wcs.input_frame  # Detector frame
        sky_frame = original_wcs.output_frame  # Sky frame
        forward_transform = original_wcs.forward_transform  # Original transform

        # Compute the reference pixel shift
        x_shift = new_y_min  # x_min, y_min are the lower-left pixel coordinates of the cutout in original image
        y_shift = new_x_min

        # Apply shift in the detector frame
        shifted_transform = (Shift(x_shift) & Shift(y_shift)) | forward_transform

        # Create a new GWCS using the updated transform
        cutout_wcs = gwcs.WCS(forward_transform=shifted_transform,
                        input_frame=detector_frame,
                        output_frame=sky_frame)

        # Assign the new GWCS to the cutout model
        cutout_wcs.bounding_box = ((new_x_min, new_x_max), (new_y_min, new_y_max))
        new_transform = (Shift(-new_x_min) & Shift(-new_y_min)) | original_wcs.forward_transform
        # cutout_wcs._pipeline = [('shift', new_transform)]
        

        # Create a new cutout model
        cutout_model = copy.deepcopy(model)
        cutout_model.meta.wcs = cutout_wcs  # Assign updated GWCS
        # cutout_model.meta.instrument = model.meta.instrument  # Preserve metadata
        # NOTE: change both gwcs and wcs
        cutout_model.meta.wcsinfo.crpix1 -= new_y_min
        cutout_model.meta.wcsinfo.crpix2 -= new_x_min
        

        # Copy data layers
        for hdu_name in ['data', 'err', 'dq', 'area', 'var_poisson', 'var_rnoise', 'var_flat']:
            if hasattr(model, hdu_name) and getattr(model, hdu_name) is not None:
                cutout_data = getattr(model, hdu_name)[
                    int(new_x_min+0.5):int(new_x_max+0.5),
                    int(new_y_min+0.5):int(new_y_max+0.5)
                ]
                setattr(cutout_model, hdu_name, cutout_data)

        # Save the cutout as a JWST-compatible FITS file
        cutout_model.save(cutout_path, overwrite=True)
        print(f"Cutout saved to {cutout_path}")


def zoom_with_nan_handling(array, zoom_factor):
    """Zoom an array while handling NaN values properly."""
    nan_mask = np.isnan(array)

    # Fill NaNs with interpolated values (Gaussian smoothing)
    array_filled = np.copy(array)
    array_filled[nan_mask] = 0  # Temporary fill
    array_filled = gaussian_filter(array_filled, sigma=1, mode='nearest')

    # Apply zoom
    zoomed_array = zoom(array_filled, zoom_factor, order=1) / (zoom_factor**2)  # Keep flux scaling

    # Apply zoom to NaN mask and restore NaNs
    zoomed_mask = zoom(nan_mask.astype(float), zoom_factor, order=0) > 0.5  # Nearest neighbor
    zoomed_array[zoomed_mask] = np.nan
    
    return zoomed_array

def shift_with_nan_handling(array, shift_values, order=3):
    """Shift an array while preserving NaN values."""
    nan_mask = np.isnan(array)

    # Fill NaNs with interpolated values
    array_filled = np.copy(array)
    array_filled[nan_mask] = 0  # Temporary fill
    array_filled = gaussian_filter(array_filled, sigma=1, mode='nearest')

    # Apply shift
    shifted_array = shift(array_filled, shift=shift_values, order=order, mode='nearest')

    # Apply shift to NaN mask and restore NaNs
    shifted_mask = shift(nan_mask.astype(float), shift=shift_values, order=0, mode='nearest') > 0.5
    shifted_array[shifted_mask] = np.nan

    return shifted_array


def save_JWST_fits(dither_path, combined_image, combined_err=None, 
                   cutout_path_base=None, base_crop_dx=None, base_crop_dy=None, 
                   center_coord=None, REFINE_WCS=False) -> None:

    cx, cy = get_pixel_center_from_array(combined_image)
    
    with ImageModel(cutout_path_base) as model_base:
        model_dither = ImageModel() 
        model_dither.meta = model_base.meta
        if cutout_path_base is not None:
            # prepare GWCS change
            original_wcs = model_base.meta.wcs
            detector_frame = original_wcs.input_frame  
            sky_frame = original_wcs.output_frame  
            forward_transform = original_wcs.forward_transform  
            scale_transform = models.Scale(0.5) & models.Scale(0.5) 
            crop_transform = models.Shift(base_crop_dx) & models.Shift(base_crop_dy)
            new_forward_transform = scale_transform | crop_transform | forward_transform
            dither_wcs = gwcs.WCS(forward_transform=new_forward_transform,
                            input_frame=detector_frame,
                            output_frame=sky_frame)
            if REFINE_WCS: 
                agn_x, agn_y = dither_wcs.world_to_pixel(center_coord)
                refine_x = agn_x - cx
                refine_y = agn_y - cy
                refine_transform = models.Shift(refine_x, name='refine x') & \
                                models.Shift(refine_y, name='refine y') 
                updated_transform = refine_transform | new_forward_transform
                dither_wcs = gwcs.WCS(forward_transform=updated_transform,
                                    input_frame=detector_frame,
                                    output_frame=sky_frame)
            model_dither.meta.wcs = dither_wcs  # Assign updated GWCS
            # prepare WCS change
            model_dither.meta.wcsinfo.crpix1 -= base_crop_dx
            model_dither.meta.wcsinfo.crpix2 -= base_crop_dy
            # print(base_crop_dx, base_crop_dy)
            model_dither.meta.wcsinfo.crpix1 = (model_dither.meta.wcsinfo.crpix1 - 0.5) * 2 + 0.5
            model_dither.meta.wcsinfo.crpix2 = (model_dither.meta.wcsinfo.crpix2 - 0.5) * 2 + 0.5
            model_dither.meta.wcsinfo.cd1_1 /= 2  # Scale X transformation
            model_dither.meta.wcsinfo.cd1_2 /= 2  # Scale Y transformation
            model_dither.meta.wcsinfo.cd2_1 /= 2
            model_dither.meta.wcsinfo.cd2_2 /= 2
            if REFINE_WCS: 
                model_dither.meta.wcsinfo.crpix1 = cx + 0.5
                model_dither.meta.wcsinfo.crpix2 = cy + 0.5
                model_dither.meta.wcsinfo.crval1 = center_coord.ra.deg
                model_dither.meta.wcsinfo.crval2 = center_coord.dec.deg
            # transfer exposure time
        else: 
            pass # TODO: add wcs construction from 0
        # model_dither.meta.exposure.exposure_time*=9 # assuming a same exposure time
        # transfer data and save
        model_dither.data = combined_image
        model_dither.err = combined_err
        model_dither.save(dither_path, overwrite=True)
    print(f"Coadded image saved to {dither_path}")



# shift optimization

def compute_power_spectrum(image):
    f_image = np.fft.fft2(image)
    power_spectrum = np.abs(np.fft.fftshift(f_image))**2
    return power_spectrum

def sum_outer_power_spectrum(image, mask=None, threshold=90):
    if mask is None:
        h, w = image.shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = r > threshold
        return np.sum(image[mask])
    else: 
        return np.sum(image[mask])

def loss_function(params, images, wt, mask, oversample_factor):
    n = len(images)
    shifts = params.reshape((n, 2))
    combined_image = combine_image(images, shifts, wt, oversample=oversample_factor)
    power_spectrum = compute_power_spectrum(combined_image)
    return sum_outer_power_spectrum(power_spectrum, mask=mask)

def optimize_shifts(images, initial_shifts, wt, mask, oversample_factor=2):
    fixed_shift = initial_shifts[0]  # Keep this fixed
    initial_params = np.array(initial_shifts[1:]).flatten()  # Optimize only the remaining shifts

    def loss_function_wrapper(params, *args):
        # Reconstruct full shifts with the fixed first one
        full_params = np.insert(params.reshape(-1, 2), 0, fixed_shift, axis=0)
        return loss_function(full_params.flatten(), *args)  # Ensure proper input format
    
    # def callback_function(params):
    #     print(f"Current loss: {loss_function_wrapper(params, images, wt, mask, oversample_factor)}")

    result = minimize(
        loss_function_wrapper,
        initial_params,
        args=(images, wt, mask, oversample_factor),
        method='L-BFGS-B',
        # method='Powell',
        # callback=callback_function,  # Add callback for monitoring progress
        options={
            'maxiter': 500,
            'disp': True,
            # 'gtol': 1e-8,
            'ftol': 1e-12,
            # 'eps': 1e-8,
        }
    )
    # Reconstruct the full optimized shifts
    optimized_shifts = np.insert(result.x.reshape(-1, 2), 0, fixed_shift, axis=0)
    return optimized_shifts

# end shift optimziation

def create_jwst_band_limit_mask(filter, size, oversample=2) -> np.ndarray[Any, np.dtype[bool]]:
    nircam = NIRCam()
    nircam.filter = filter.upper()
    app = 0.063 if nircam.channel=='long' else 0.031
    if size%2==0: 
        nircam.options['source_offset_x'] = -app/2/oversample
        nircam.options['source_offset_y'] = -app/2/oversample
    oversampled_psf = nircam.calc_psf(oversample=oversample, fov_pixels=size)[0].data
    mask = compute_power_spectrum(oversampled_psf)<1e-7
    return mask


def iterative_nan_fill(array, kernel=None):
    if kernel is None:
        kernel = np.array([[0, 0.25, 0], 
                           [0.25, 0, 0.25], 
                           [0, 0.25, 0]])
    filled_array = array.copy()
    while np.isnan(filled_array).any():
        num_neighbors = convolve2d(~np.isnan(filled_array), kernel, mode='same', boundary='symm')
        neighbor_sum = convolve2d(np.nan_to_num(filled_array, nan=0), kernel, mode='same', boundary='symm')
        mask = np.isnan(filled_array) & (num_neighbors > 0)
        avg_neighbors = np.zeros_like(filled_array)
        valid = num_neighbors > 0
        avg_neighbors[valid] = neighbor_sum[valid] / num_neighbors[valid]
        filled_array[mask] = avg_neighbors[mask]
    return filled_array