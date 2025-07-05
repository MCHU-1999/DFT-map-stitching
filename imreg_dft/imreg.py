# -*- coding: utf-8 -*-
# imreg.py

# Copyright (c) 2014-?, Matěj Týč
# Copyright (c) 2011-2014, Christoph Gohlke
# Copyright (c) 2011-2014, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
FFT based image registration. --- main functions
"""

from __future__ import division, print_function

import math

import numpy as np
try:
    import pyfftw.interfaces.numpy_fft as fft
except ImportError:
    import numpy.fft as fft
import scipy.ndimage.interpolation as ndii
import imreg_dft.utils as utils

def _logpolar_filter(shape):
    """
    Make a radial cosine filter for the logpolar transform.
    This filter suppresses low frequencies and completely removes
    the zero freq.
    """
    yy = np.linspace(- np.pi / 2., np.pi / 2., shape[0])[:, np.newaxis]
    xx = np.linspace(- np.pi / 2., np.pi / 2., shape[1])[np.newaxis, :]
    # Supressing low spatial frequencies is a must when using log-polar
    # transform. The scale stuff is poorly reflected with low freqs.
    rads = np.sqrt(yy ** 2 + xx ** 2)
    filt = 1.0 - np.cos(rads) ** 2
    # vvv This doesn't really matter, very high freqs are not too usable anyway
    filt[np.abs(rads) > np.pi / 2] = 1
    return filt


def _get_pcorr_shape(shape):
    ret = (int(max(shape) * 1.0),) * 2
    return ret


def _get_ang_scale(ims, bgval, exponent='inf', constraints=None, reports=None):
    """
    Given two images, return their scale and angle difference.

    Args:
        ims (2-tuple-like of 2D ndarrays): The images
        bgval: We also pad here in the :func:`map_coordinates`
        exponent (float or 'inf'): The exponent stuff, see :func:`similarity`
        constraints (dict, optional)
        reports (optional)

    Returns:
        tuple: Scale, angle. Describes the relationship of
        the subject image to the first one.
    """
    assert len(ims) == 2, \
        "Only two images are supported as input"
    shape = ims[0].shape

    ims_apod = [utils._apodize(im) for im in ims]
    dfts = [fft.fftshift(fft.fft2(im)) for im in ims_apod]
    filt = _logpolar_filter(shape)
    dfts = [dft * filt for dft in dfts]

    # High-pass filtering used to be here, but we have moved it to a higher level interface

    pcorr_shape = _get_pcorr_shape(shape)
    log_base = _get_log_base(shape, pcorr_shape[1])
    stuffs = [_logpolar(np.abs(dft), pcorr_shape, log_base) for dft in dfts]

    (arg_ang, arg_rad), success = _phase_correlation(
        stuffs[0], stuffs[1],
        utils.argmax_angscale, log_base, exponent, constraints, reports)

    angle = -np.pi * arg_ang / float(pcorr_shape[0])
    angle = np.rad2deg(angle)
    angle = utils.wrap_angle(angle, 360)
    scale = log_base ** arg_rad

    angle = - angle
    scale = 1.0 / scale

    if reports is not None:
        reports["shape"] = filt.shape
        reports["base"] = log_base

        if reports.show("spectra"):
            reports["dfts_filt"] = dfts
        if reports.show("inputs"):
            reports["ims_filt"] = [fft.ifft2(np.fft.ifftshift(dft))
                                   for dft in dfts]
        if reports.show("logpolar"):
            reports["logpolars"] = stuffs

        if reports.show("scale_angle"):
            reports["amas-result-raw"] = (arg_ang, arg_rad)
            reports["amas-result"] = (scale, angle)
            reports["amas-success"] = success
            extent_el = pcorr_shape[1] / 2.0
            reports["amas-extent"] = (
                log_base ** (-extent_el), log_base ** extent_el,
                -90, 90
            )

    if not 0.5 < scale < 2:
        raise ValueError(
            "Images are not compatible. Scale change %g too big to be true."
            % scale)

    return scale, angle


def translation(im0, im1, filter_pcorr=0, odds=1, constraints=None,
                reports=None):
    """
    Return translation vector to register images.
    It tells how to translate the im1 to get im0.

    Args:
        im0 (2D numpy array): The first (template) image
        im1 (2D numpy array): The second (subject) image
        filter_pcorr (int): Radius of the minimum spectrum filter
            for translation detection, use the filter when detection fails.
            Values > 3 are likely not useful.
        constraints (dict or None): Specify preference of seeked values.
            For more detailed documentation, refer to :func:`similarity`.
            The only difference is that here, only keys ``tx`` and/or ``ty``
            (i.e. both or any of them or none of them) are used.
        odds (float): The greater the odds are, the higher is the preferrence
            of the angle + 180 over the original angle. Odds of -1 are the same
            as inifinity.
            The value 1 is neutral, the converse of 2 is 1 / 2 etc.

    Returns:
        dict: Contains following keys: ``angle``, ``tvec`` (Y, X),
            and ``success``.
    """
    angle = 0
    report_one = report_two = None
    if reports is not None and reports.show("translation"):
        report_one = reports.copy_empty()
        report_two = reports.copy_empty()

    # We estimate translation for the original image...
    tvec, succ = _translation(im0, im1, filter_pcorr, constraints, report_one)
    # ... and for the 180-degrees rotated image (the rotation estimation
    # doesn't distinguish rotation of x vs x + 180deg).
    tvec2, succ2 = _translation(im0, utils.rot180(im1), filter_pcorr,
                                constraints, report_two)

    pick_rotated = False
    if succ2 * odds > succ or odds == -1:
        pick_rotated = True

    if reports is not None and reports.show("translation"):
        reports["t0-orig"] = report_one["amt-orig"]
        reports["t0-postproc"] = report_one["amt-postproc"]
        reports["t0-success"] = succ
        reports["t0-tvec"] = tuple(tvec)

        reports["t1-orig"] = report_two["amt-orig"]
        reports["t1-postproc"] = report_two["amt-postproc"]
        reports["t1-success"] = succ2
        reports["t1-tvec"] = tuple(tvec2)

    if reports is not None and reports.show("transformed"):
        toapp = [
            transform_img(utils.rot180(im1), tvec=tvec2, mode="wrap", order=3),
            transform_img(im1, tvec=tvec, mode="wrap", order=3),
        ]
        if pick_rotated:
            toapp = toapp[::-1]
        reports["after_tform"].extend(toapp)

    if pick_rotated:
        tvec = tvec2
        succ = succ2
        angle += 180

    ret = dict(tvec=tvec, success=succ, angle=angle)
    return ret


def _get_precision(shape, scale=1):
    """
    Given the parameters of the log-polar transform, get width of the interval
    where the correct values are.

    Args:
        shape (tuple): Shape of images
        scale (float): The scale difference (precision varies)
    """
    pcorr_shape = _get_pcorr_shape(shape)
    log_base = _get_log_base(shape, pcorr_shape[1])
    # * 0.5 <= max deviation is half of the step
    # * 0.25 <= we got subpixel precision now and 0.5 / 2 == 0.25
    # sccale: Scale deviation depends on the scale value
    Dscale = scale * (log_base - 1) * 0.25
    # angle: Angle deviation is constant
    Dangle = 180.0 / pcorr_shape[0] * 0.25
    return Dangle, Dscale


def _similarity(im0, im1, numiter=1, order=3, constraints=None,
                filter_pcorr=0, exponent='inf', bgval=None, reports=None, use_scale=False):
    """
    This function takes some input and returns mutual rotation, scale
    and translation.
    It does these things during the process:

    * Handles correct constraints handling (defaults etc.).
    * Performs angle-scale determination iteratively.
      This involves keeping constraints in sync.
    * Performs translation determination.
    * Calculates precision.

    Returns:
        Dictionary with results.
    """
    if bgval is None:
        bgval = utils.get_borderval(im1, 5)

    shape = im0.shape
    if shape != im1.shape:
        raise ValueError("Images must have same shapes.")
    elif im0.ndim != 2:
        raise ValueError("Images must be 2-dimensional.")

    # We are going to iterate and precise scale and angle estimates
    scale = 1.0
    angle = 0.0
    im2 = im1

    constraints_default = dict(angle=[0, None], scale=[1, None])
    if constraints is None:
        constraints = constraints_default

    # We guard against case when caller passes only one constraint key.
    # Now, the provided ones just replace defaults.
    constraints_default.update(constraints)
    constraints = constraints_default

    # During iterations, we have to work with constraints too.
    # So we make the copy in order to leave the original intact
    constraints_dynamic = constraints.copy()
    constraints_dynamic["scale"] = list(constraints["scale"])
    constraints_dynamic["angle"] = list(constraints["angle"])

    if reports is not None and reports.show("transformed"):
        reports["after_tform"] = [im2.copy()]

    for ii in range(numiter):
        if use_scale:
            newscale, newangle = _get_ang_scale([im0, im2], bgval, exponent,
                                                constraints_dynamic, reports)
            scale *= newscale
            angle += newangle

            constraints_dynamic["scale"][0] /= newscale
            constraints_dynamic["angle"][0] -= newangle
        else:
            _, newangle = _get_ang_scale([im0, im2], bgval, exponent,
                                                constraints_dynamic, reports)
            angle += newangle

            constraints_dynamic["angle"][0] -= newangle

        im2 = transform_img(im1, scale, angle, bgval=bgval, order=order)

        if reports is not None and reports.show("transformed"):
            reports["after_tform"].append(im2.copy())

    # Here we look how is the turn-180
    # target, stdev = constraints.get("angle", (0, None))
    # odds = _get_odds(angle, target, stdev)
    odds = _get_odds_compare_simple_diff(im0, im2, bgval, order)

    # now we can use pcorr to guess the translation
    res = translation(im0, im2, filter_pcorr, odds,
                      constraints, reports)

    # The log-polar transform may have got the angle wrong by 180 degrees.
    # The phase correlation can help us to correct that
    angle += res["angle"]
    res["angle"] = utils.wrap_angle(angle, 360)

    # don't know what it does, but it alters the scale a little bit
    # scale = (im1.shape[1] - 1) / (int(im1.shape[1] / scale) - 1)

    Dangle, Dscale = _get_precision(shape, scale)

    res["scale"] = scale
    res["Dscale"] = Dscale
    res["Dangle"] = Dangle
    # 0.25 because we go subpixel now
    res["Dt"] = 0.25

    return res


def similarity(im0, im1, numiter=1, order=3, constraints=None,
               filter_pcorr=0, exponent='inf', reports=None):
    """
    Return similarity transformed image im1 and transformation parameters.
    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector.

    A similarity transformation is an affine transformation with isotropic
    scale and without shear.

    Args:
        im0 (2D numpy array): The first (template) image
        im1 (2D numpy array): The second (subject) image
        numiter (int): How many times to iterate when determining scale and
            rotation
        order (int): Order of approximation (when doing transformations). 1 =
            linear, 3 = cubic etc.
        filter_pcorr (int): Radius of a spectrum filter for translation
            detection
        exponent (float or 'inf'): The exponent value used during processing.
            Refer to the docs for a thorough explanation. Generally, pass "inf"
            when feeling conservative. Otherwise, experiment, values below 5
            are not even supposed to work.
        constraints (dict or None): Specify preference of seeked values.
            Pass None (default) for no constraints, otherwise pass a dict with
            keys ``angle``, ``scale``, ``tx`` and/or ``ty`` (i.e. you can pass
            all, some of them or none of them, all is fine). The value of a key
            is supposed to be a mutable 2-tuple (e.g. a list), where the first
            value is related to the constraint center and the second one to
            softness of the constraint (the higher is the number,
            the more soft a constraint is).

            More specifically, constraints may be regarded as weights
            in form of a shifted Gaussian curve.
            However, for precise meaning of keys and values,
            see the documentation section :ref:`constraints`.
            Names of dictionary keys map to names of command-line arguments.

    Returns:
        dict: Contains following keys: ``scale``, ``angle``, ``tvec`` (Y, X),
        ``success`` and ``timg`` (the transformed subject image)

    .. note:: There are limitations

        * Scale change must be less than 2.
        * No subpixel precision (but you can use *resampling* to get
          around this).
    """
    # bgval = utils.get_borderval(im1, 5)
    bgval = 0

    res = _similarity(im0, im1, numiter, order, constraints,
                      filter_pcorr, exponent, bgval, reports, False)

    im2 = transform_img_dict(im1, res, bgval, order)
    # Order of mask should be always 1 - higher values produce strange results.
    imask = transform_img_dict(np.ones_like(im1), res, 0, 1)
    # This removes some weird artifacts
    imask[imask > 0.8] = 1.0

    # Framing here = just blending the im2 with its BG according to the mask
    im3 = utils.frame_img(im2, imask, 10)

    res["timg"] = im3
    return res


def _get_odds(angle, target, stdev):
    """
    Determine whether we are more likely to choose the angle, or angle + 180°

    Args:
        angle (float, degrees): The base angle.
        target (float, degrees): The angle we think is the right one.
            Typically, we take this from constraints.
        stdev (float, degrees): The relevance of the target value.
            Also typically taken from constraints.

    Return:
        float: The greater the odds are, the higher is the preferrence
            of the angle + 180 over the original angle. Odds of -1 are the same
            as inifinity.
    """
    ret = 1
    if stdev is not None:
        diffs = [abs(utils.wrap_angle(ang, 360))
                 for ang in (target - angle, target - angle + 180)]
        odds0, odds1 = 0, 0
        if stdev > 0:
            odds0, odds1 = [np.exp(- diff ** 2 / stdev ** 2) for diff in diffs]
        if odds0 == 0 and odds1 > 0:
            # -1 is treated as infinity in _translation
            ret = -1
        elif stdev == 0 or (odds0 == 0 and odds1 == 0):
            ret = -1
            if diffs[0] < diffs[1]:
                ret = 0
        else:
            ret = odds1 / odds0
    return ret

def _get_odds_compare_simple_diff(im0, im1, bgval=None, order=3):
    """
    Simple difference-based comparison: choose the transformation with smaller difference.
    
    Args:
        im0 (np.array): Reference image
        im1 (np.array): Image to transform
        bgval: Background value
        order (int): Interpolation order
        
    Returns:
        float: Odds ratio. >1 means choose angle+180°, <1 means keep original
    """
    if bgval is None:
        bgval = utils.get_borderval(im1, 5)

    im1_orig = im1
    im1_180 = transform_img(im1, 1, 180, bgval=bgval, order=order)
    
    # Calculate differences
    diff_orig = np.mean(np.abs(im0 - im1_orig))  # Mean Absolute Error
    diff_180 = np.mean(np.abs(im0 - im1_180))
    
    print(f"Difference comparison: orig={diff_orig:.4f}, 180°={diff_180:.4f}")
    
    # Return odds: higher means prefer 180° version
    # If diff_180 < diff_orig, we want odds > 1
    if diff_180 > 0:
        return diff_orig / diff_180  # Higher when diff_orig > diff_180
    else:
        return float('inf')  # Perfect match with 180° version


def _translation(im0, im1, filter_pcorr=0, constraints=None, reports=None):
    """
    The plain wrapper for translation phase correlation, no big deal.
    """
    # Apodization and pcorr don't play along
    # im0, im1 = [utils._apodize(im, ratio=1) for im in (im0, im1)]
    ret, succ = _phase_correlation(
        im0, im1,
        utils.argmax_translation, filter_pcorr, constraints, reports)
    return ret, succ


def _phase_correlation(im0, im1, callback=None, *args):
    """
    Computes phase correlation between im0 and im1

    Args:
        im0
        im1
        callback (function): Process the cross-power spectrum (i.e. choose
            coordinates of the best element, usually of the highest one).
            Defaults to :func:`imreg_dft.utils.argmax2D`

    Returns:
        tuple: The translation vector (Y, X). Translation vector of (0, 0)
            means that the two images match.
    """
    if callback is None:
        callback = utils._argmax2D

    # TODO: Implement some form of high-pass filtering of PHASE correlation
    f0, f1 = [fft.fft2(arr) for arr in (im0, im1)]
    # spectrum can be filtered (already),
    # so we have to take precaution against dividing by 0
    eps = abs(f1).max() * 1e-15
    # cps == cross-power spectrum of im0 and im1
    cps = abs(fft.ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1) + eps)))
    # scps = shifted cps
    scps = fft.fftshift(cps)

    (t0, t1), success = callback(scps, *args)
    ret = np.array((t0, t1))

    # _compensate_fftshift is not appropriate here, this is OK.
    t0 -= f0.shape[0] // 2
    t1 -= f0.shape[1] // 2

    ret -= np.array(f0.shape, int) // 2
    return ret, success


def transform_img_dict(img, tdict, bgval=None, order=1, invert=False):
    """
    Wrapper of :func:`transform_img`, works well with the :func:`similarity`
    output.

    Args:
        img
        tdict (dictionary): Transformation dictionary --- supposed to contain
            keys "scale", "angle" and "tvec"
        bgval
        order
        invert (bool): Whether to perform inverse transformation --- doesn't
            work very well with the translation.

    Returns:
        np.ndarray: .. seealso:: :func:`transform_img`
    """
    scale = tdict["scale"]
    angle = tdict["angle"]
    tvec = np.array(tdict["tvec"])
    if invert:
        scale = 1.0 / scale
        angle *= -1
        tvec *= -1
    res = transform_img(img, scale, angle, tvec, bgval=bgval, order=order)
    return res


def transform_img(img, scale=1.0, angle=0.0, tvec=(0, 0),
                  mode="constant", bgval=None, order=1):
    """
    Return translation vector to register images.

    Args:
        img (2D or 3D numpy array): What will be transformed.
            If a 3D array is passed, it is treated in a manner in which RGB
            images are supposed to be handled - i.e. assume that coordinates
            are (Y, X, channels).
            Complex images are handled in a way that treats separately
            the real and imaginary parts.
        scale (float): The scale factor (scale > 1.0 means zooming in)
        angle (float): Degrees of rotation (clock-wise)
        tvec (2-tuple): Pixel translation vector, Y and X component.
        mode (string): The transformation mode (refer to e.g.
            :func:`scipy.ndimage.shift` and its kwarg ``mode``).
        bgval (float): Shade of the background (filling during transformations)
            If None is passed, :func:`imreg_dft.utils.get_borderval` with
            radius of 5 is used to get it.
        order (int): Order of approximation (when doing transformations). 1 =
            linear, 3 = cubic etc. Linear works surprisingly well.

    Returns:
        np.ndarray: The transformed img, may have another
        i.e. (bigger) shape than the source.
    """
    if img.ndim == 3:
        # A bloody painful special case of RGB images
        ret = np.empty_like(img)
        for idx in range(img.shape[2]):
            sli = (slice(None), slice(None), idx)
            ret[sli] = transform_img(img[sli], scale, angle, tvec,
                                     mode, bgval, order)
        return ret
    elif np.iscomplexobj(img):
        decomposed = np.empty(img.shape + (2,), float)
        decomposed[:, :, 0] = img.real
        decomposed[:, :, 1] = img.imag
        # The bgval makes little sense now, as we decompose the image
        res = transform_img(decomposed, scale, angle, tvec, mode, None, order)
        ret = res[:, :, 0] + 1j * res[:, :, 1]
        return ret

    if bgval is None:
        bgval = utils.get_borderval(img)

    bigshape = np.round(np.array(img.shape) * 1.2).astype(int)
    bg = np.zeros(bigshape, img.dtype) + bgval

    dest0 = utils.embed_to(bg, img.copy())
    # TODO: We have problems with complex numbers
    # that are not supported by zoom(), rotate() or shift()
    if scale != 1.0:
        dest0 = ndii.zoom(dest0, scale, order=order, mode=mode, cval=bgval)
    if angle != 0.0:
        dest0 = ndii.rotate(dest0, angle, order=order, mode=mode, cval=bgval)

    if tvec[0] != 0 or tvec[1] != 0:
        dest0 = ndii.shift(dest0, tvec, order=order, mode=mode, cval=bgval)

    bg = np.zeros_like(img) + bgval
    dest = utils.embed_to(bg, dest0)
    return dest


def similarity_matrix(scale, angle, vector):
    """
    Return homogeneous transformation matrix from similarity parameters.

    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector (of size 2).

    The order of transformations is: scale, rotate, translate.

    """
    raise NotImplementedError("We have no idea what this is supposed to do")
    m_scale = np.diag([scale, scale, 1.0])
    m_rot = np.identity(3)
    angle = math.radians(angle)
    m_rot[0, 0] = math.cos(angle)
    m_rot[1, 1] = math.cos(angle)
    m_rot[0, 1] = -math.sin(angle)
    m_rot[1, 0] = math.sin(angle)
    m_transl = np.identity(3)
    m_transl[:2, 2] = vector
    return np.dot(m_transl, np.dot(m_rot, m_scale))


EXCESS_CONST = 1.1


def _get_log_base(shape, new_r):
    """
    Basically common functionality of :func:`_logpolar`
    and :func:`_get_ang_scale`

    This value can be considered fixed, if you want to mess with the logpolar
    transform, mess with the shape.

    Args:
        shape: Shape of the original image.
        new_r (float): The r-size of the log-polar transform array dimension.

    Returns:
        float: Base of the log-polar transform.
        The following holds:
        :math:`log\_base = \exp( \ln [ \mathit{spectrum\_dim} ] / \mathit{loglpolar\_scale\_dim} )`,
        or the equivalent :math:`log\_base^{\mathit{loglpolar\_scale\_dim}} = \mathit{spectrum\_dim}`.
    """
    # The highest radius we have to accomodate is 'old_r',
    # However, we cut some parts out as only a thin part of the spectra has
    # these high frequencies
    old_r = shape[0] * EXCESS_CONST
    # We are radius, so we divide the diameter by two.
    old_r /= 2.0
    # we have at most 'new_r' of space.
    log_base = np.exp(np.log(old_r) / new_r)
    return log_base


def _logpolar(image, shape, log_base, bgval=None):
    """
    Return log-polar transformed image
    Takes into account anisotropicity of the freq spectrum
    of rectangular images

    Args:
        image: The image to be transformed
        shape: Shape of the transformed image
        log_base: Parameter of the transformation, get it via
            :func:`_get_log_base`
        bgval: The backround value. If None, use minimum of the image.

    Returns:
        The transformed image
    """
    if bgval is None:
        bgval = np.percentile(image, 1)
    imshape = np.array(image.shape)
    center = imshape[0] / 2.0, imshape[1] / 2.0
    # 0 .. pi = only half of the spectrum is used
    theta = utils._get_angles(shape)
    radius_x = utils._get_lograd(shape, log_base)
    radius_y = radius_x.copy()
    ellipse_coef = imshape[0] / float(imshape[1])
    # We have to acknowledge that the frequency spectrum can be deformed
    # if the image aspect ratio is not 1.0
    # The image is x-thin, so we acknowledge that the frequency spectra
    # scale in x is shrunk.
    radius_x /= ellipse_coef

    y = radius_y * np.sin(theta) + center[0]
    x = radius_x * np.cos(theta) + center[1]
    output = np.empty_like(y)
    ndii.map_coordinates(image, [y, x], output=output, order=3,
                         mode="constant", cval=bgval)
    return output


# def imshow(im0, im1, im2, cmap=None, fig=None, title="Pair 1", subtitle=True, **kwargs):
#     """
#     Plot images using matplotlib.
#     Opens a new figure with four subplots:

#     ::

#       +----------------------+---------------------+----------------------+----------------------+
#       |                      |                     |                      | <difference between  |
#       |   <template image>   |   <subject image>   | <transformed subject>|  template and the    |
#       |                      |                     |                      | transformed subject> |
#       +----------------------+---------------------+----------------------+----------------------+

#     Args:
#         im0 (np.ndarray): The template image
#         im1 (np.ndarray): The subject image
#         im2: The transformed subject --- it is supposed to match the template
#         cmap (optional): colormap
#         fig (optional): The figure you would like to have this plotted on

#     Returns:
#         matplotlib figure: The figure with subplots
#     """
#     import matplotlib.pyplot as plt

#     if fig is None:
#         fig = plt.figure(figsize=(15, 3))
#     if cmap is None:
#         cmap = 'coolwarm'

#     # We do the difference between the template and the result now
#     # To increase the contrast of the difference, we norm images according
#     # to their near-maximums
#     norm = np.percentile(np.abs(im2), 99.5) / np.percentile(np.abs(im0), 99.5)
#     # Divide by zero is OK here
#     phase_norm = np.median(np.angle(im2 / im0) % (2 * np.pi))
#     if phase_norm != 0:
#         norm *= np.exp(1j * phase_norm)
#     im3 = abs(im2 - im0 * norm)
    
#     # tick setting
#     N_TICKS = (im0.shape[0] // 5)

#     # First subplot: Title only
#     pl_title = fig.add_subplot(151)
#     pl_title.text(0.5, 0.5, title, fontsize=20,
#                   horizontalalignment='center', verticalalignment='center',
#                   transform=pl_title.transAxes, wrap=True)
#     pl_title.axis('off')
#     # if subtitle : pl_title.set_title("Pair no.")

#     # 2nd
#     pl0 = fig.add_subplot(152)
#     pl0.imshow(im0.real, cmap, **kwargs)

#     xlim = pl0.get_xlim()
#     ylim = pl0.get_ylim()
#     xticks = np.linspace(xlim[0], xlim[1], N_TICKS + 1)
#     yticks = np.linspace(ylim[0], ylim[1], N_TICKS + 1)

#     pl0.set_xticks(xticks)
#     pl0.set_yticks(yticks)
#     pl0.grid()
#     # if subtitle : pl0.set_title("image 1")
#     pl0.set_xticklabels([])
#     pl0.set_yticklabels([])

#     share = dict(sharex=pl0, sharey=pl0)

#     # 3rd
#     pl = fig.add_subplot(153, **share)
#     pl.imshow(im1.real, cmap, **kwargs)
#     pl.set_xticks(xticks)
#     pl.set_yticks(yticks)
#     pl.grid()
#     # if subtitle : pl.set_title("image 2")
#     pl.set_xticklabels([])
#     pl.set_yticklabels([])

#     # 4th
#     pl = fig.add_subplot(154, **share)
#     pl.imshow(im2.real, cmap, **kwargs)
#     pl.set_xticks(xticks)
#     pl.set_yticks(yticks)
#     pl.grid()
#     # if subtitle : pl.set_title("transformed")
#     pl.set_xticklabels([])
#     pl.set_yticklabels([])

#     # 5th
#     pl = fig.add_subplot(155, **share)
#     pl.imshow(im3, cmap='coolwarm', **kwargs)
#     pl.set_xticks(xticks)
#     pl.set_yticks(yticks)
#     pl.grid()
#     # if subtitle : pl.set_title("difference")
#     pl.set_xticklabels([])
#     pl.set_yticklabels([])

#     plt.tight_layout()

#     return fig


def imshow(im0, im1, im2, cmap=None, fig=None, title="Pair 1", subtitle=True, show_spectrum=True, **kwargs):
    """
    Plot images and their frequency spectra using matplotlib.
    Opens a new figure with subplots showing spatial and frequency domain representations:

    If show_spectrum=False (spatial domain only):
    +---------------------+----------------------+----------------------+----------------------+
    |   <template image>  |   <subject image>    | <transformed subject>| <difference between  |
    |                     |                      |                      |  template and the    |
    |                     |                      |                      | transformed subject> |
    +---------------------+----------------------+----------------------+----------------------+

    If show_spectrum=True (spatial + frequency domain):
    +---------------------+----------------------+----------------------+----------------------+---------------------+----------------------+----------------------+
    |   <template image>  |   <subject image>    | <transformed subject>| <difference between  | <template spectrum> | <subject spectrum>   |<transformed spectrum>|
    |                     |                      |                      |  template and the    |                     |                      |                      |
    |                     |                      |                      | transformed subject> |                     |                      |                      |
    +---------------------+----------------------+----------------------+----------------------+---------------------+----------------------+----------------------+

    Args:
        im0 (np.ndarray): The template image
        im1 (np.ndarray): The subject image  
        im2 (np.ndarray): The transformed subject --- it is supposed to match the template
        cmap (optional): colormap for spatial domain images
        fig (optional): The figure you would like to have this plotted on
        title (str): Title for the entire figure
        subtitle (bool): Whether to show subplot titles
        show_spectrum (bool): Whether to show frequency domain representations
        **kwargs: Additional arguments passed to imshow

    Returns:
        matplotlib figure: The figure with subplots
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Determine figure size and subplot configuration
    if show_spectrum:
        figsize = (21, 3.5)  # Wider for 7 columns
        n_rows = 1
        subplot_base = 170  # 1 row, 7 columns
    else:
        figsize = (15, 3.5)
        n_rows = 1
        subplot_base = 140  # 1 row, 4 columns

    if fig is None:
        fig = plt.figure(figsize=figsize)
    
    # Set the main title using suptitle instead of a dedicated subplot
    fig.text(0.01, 0.5, title, rotation=90, verticalalignment='center', 
                fontsize=20, fontweight='bold')

    if cmap is None:
        cmap = 'coolwarm'

    # Compute difference image with normalization
    norm = np.percentile(np.abs(im2), 99.5) / np.percentile(np.abs(im0), 99.5)
    phase_norm = np.median(np.angle(im2 / im0) % (2 * np.pi))
    if phase_norm != 0:
        norm *= np.exp(1j * phase_norm)
    im3 = abs(im2 - im0 * norm)
    
    # Compute frequency domain representations (FFT)
    def compute_spectrum(image):
        """Compute log magnitude spectrum of image"""
        fft = np.fft.fftshift(np.fft.fft2(image))
        spectrum = np.log(np.abs(fft) + 1e-10)  # Add small value to avoid log(0)
        return spectrum
    
    spectrum0 = compute_spectrum(im0)
    spectrum1 = compute_spectrum(im1) 
    spectrum2 = compute_spectrum(im2)
    
    # Tick settings
    N_TICKS = (im0.shape[0] // 5)

    # === SPATIAL DOMAIN PLOTS (top row) ===
    
    # Template image
    ax1 = fig.add_subplot(subplot_base + 1)
    im_plot1 = ax1.imshow(im0.real, cmap, **kwargs)
    
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    xticks = np.linspace(xlim[0], xlim[1], N_TICKS + 1)
    yticks = np.linspace(ylim[0], ylim[1], N_TICKS + 1)
    
    ax1.set_xticks(xticks)
    ax1.set_yticks(yticks)
    ax1.grid()
    if subtitle:
        ax1.set_title("im0", fontsize=16)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    # ax1.set_ylabel("Images", rotation=90, labelpad=10)

    share = dict(sharex=ax1, sharey=ax1)

    # Subject image
    ax2 = fig.add_subplot(subplot_base + 2, **share)
    ax2.imshow(im1.real, cmap, **kwargs)
    ax2.set_xticks(xticks)
    ax2.set_yticks(yticks)
    ax2.grid()
    if subtitle:
        ax2.set_title("im1", fontsize=16)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    # Transformed subject
    ax3 = fig.add_subplot(subplot_base + 3, **share)
    ax3.imshow(im2.real, cmap, **kwargs)
    ax3.set_xticks(xticks)
    ax3.set_yticks(yticks)
    ax3.grid()
    if subtitle:
        ax3.set_title("im1 transformed", fontsize=16)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])

    # Difference
    ax4 = fig.add_subplot(subplot_base + 4, **share)
    ax4.imshow(im3, cmap='coolwarm', **kwargs)
    ax4.set_xticks(xticks)
    ax4.set_yticks(yticks)
    ax4.grid()
    if subtitle:
        ax4.set_title("difference", fontsize=16)
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])

    # === FREQUENCY DOMAIN PLOTS (bottom row) ===
    if show_spectrum:
        # Template spectrum
        ax5 = fig.add_subplot(subplot_base + 5)
        ax5.imshow(spectrum0, cmap='viridis', **kwargs)
        ax5.set_xticks(xticks)
        ax5.set_yticks(yticks)
        ax5.grid()
        if subtitle:
            ax5.set_title("im0 spectrum", fontsize=16)
        ax5.set_xticklabels([])
        ax5.set_yticklabels([])
        # ax5.set_ylabel("Frequency Domain", rotation=90, labelpad=10)

        share_freq = dict(sharex=ax5, sharey=ax5)

        # Subject spectrum
        ax6 = fig.add_subplot(subplot_base + 6, **share_freq)
        ax6.imshow(spectrum1, cmap='viridis', **kwargs)
        ax6.set_xticks(xticks)
        ax6.set_yticks(yticks)
        ax6.grid()
        if subtitle:
            ax6.set_title("im1 spectrum", fontsize=16)
        ax6.set_xticklabels([])
        ax6.set_yticklabels([])

        # Transformed spectrum
        ax7 = fig.add_subplot(subplot_base + 7, **share_freq)
        ax7.imshow(spectrum2, cmap='viridis', **kwargs)
        ax7.set_xticks(xticks)
        ax7.set_yticks(yticks)
        ax7.grid()
        if subtitle:
            ax7.set_title("im1 transformed spectrum", fontsize=16)
        ax7.set_xticklabels([])
        ax7.set_yticklabels([])

    plt.tight_layout()
    
    # Adjust layout to accommodate suptitle
    plt.subplots_adjust(left=0.03, wspace=0.03)

    return fig

