# -*- coding: utf-8 -*-
"""
Construction of the measurement model describing the formation
of the acquired data.
"""

from typing import Optional, Tuple
import numpy as np


# TODO Add an attribute to store noise variance information?
class GIMeasurementModel:
    """
    The mathematical model of how the measurement data from the bucket detector
    relate to the studied object.

    Parameters
    ----------
    n_patterns : int
        The number of illumination patterns, corresponding to measurement size.
    img_shape : 2-tuple of ints or None, optional
        The shape of the ghost image. Can be omitted if the illumination patterns
        are loaded from files instead of calculations. In that case if it is not None,
        the loaded patterns are resized to the given size.
    pattern_type : {"pseudorandom", "quasirandom", "speckle"}, optional
        What illumination patterns to use. Valid values are "pseudorandom",
        "quasirandom" (corresponding to binary pseudo- or quasirandom patterns
        which then pass through the optical fiber) and "speckle" (corresponding
        to acquired photos of illumination patterns). Photos are loaded from
        the files 'speckle_patterns/slmX', where X are integers starting from 0
        and the format is any supported by `imageio.imread`.
        The default is "pseudorandom".
    pixel_size : float or None, optional
        Image pixel size in e.g. metric units.
    unit : str or None, optional
        The unit of pixel_size.

    Attributes
    ----------
    mt_op : numpy.ndarray
        The matrix of the linear operator whose rows are raveled illumination
        patterns. Has shape (n_patterns, img_shape[0]*img_shape[1]).
    img_shape : 2-tuple of ints
        The shape of the ghost image.
    pixel_size : float, optional
        Image pixel size in e.g. metric units. In the future, it will be
        a mandatory argument if illumination patterns are calculated,
        in which case it has to be measured in μm.
    unit : str, optional
        The unit of pixel_size.
    """

    def __init__(self, ref_data,  # pylint: disable=R0913
                 pattern_type: str = "pseudorandom", pixel_size: float = 1.,
                 unit: str = "px"):
        self.n_patterns = ref_data.shape[0]
        self.pixel_size = pixel_size
        self.unit = unit
        self.img_shape = ref_data.shape[1:]
        self.mt_op = ref_data.reshape((self.n_patterns, -1)).astype(float)

    def mt_op_part(self, n_patterns: Optional[int] = None) -> np.ndarray:
        """
        Extract a part of `mt_op` corresponding to first `n_patterns` measurements.

        Parameters
        ----------
        n_patterns : int or None, optional
            The number of patterns. The default is None.

        Returns
        -------
        numpy.ndarray of shape (n_patterns, img_shape[0]*img_shape[1])
            The extracted part of mt_op.

        """
        if n_patterns is None:
            return self.mt_op
        return self.mt_op[: n_patterns, ...]

    def illumination_patterns(self, n_patterns: int = None) -> np.ndarray:
        """
        Reshapes the mt_op so that mt_op[i, ...] are the illumination patterns.

        Parameters
        ----------
        n_patterns : int or None, optional
            The number of illumination patterns to return. If None, all patterns
            are returned.

        Returns
        -------
        numpy.ndarray of shape (n_patterns, img_shape[0], img_shape[1])
            The illumination patterns.

        """
        return self.mt_op[: n_patterns, ...].reshape((-1,) + self.img_shape)


def pad_or_trim_to_shape(img, shape: Tuple[int, int]) -> np.ndarray:
    """
    Pad an image with zeros or trim it, in both cases keeping the image
    as centered as possible, to make it the specified shape.

    Parameters
    ----------
    img : array_like
        The image.
    shape : 2-tuple of int
        The shape.

    Returns
    -------
    img : ndarray
        The adjusted image.

    """
    if shape[0] == img.shape[0] and shape[1] == img.shape[1]:
        return img
    if (shape[0] != img.shape[0]
            or shape[1] != img.shape[1]):
        to_add_0 = shape[0] - img.shape[0]
        to_add_1 = shape[1] - img.shape[1]
        if to_add_0 < 0:
            diff = -to_add_0//2
            img = img[diff: diff + shape[0], :]
            to_add_0 = 0
        if to_add_1 < 0:
            diff = -to_add_1//2
            img = img[:, diff: diff + shape[1]]
            to_add_1 = 0
        img = np.pad(img,
                     ((to_add_0//2, to_add_0 - to_add_0//2),
                      (to_add_1//2, to_add_1 - to_add_1//2)),
                     mode="constant", constant_values=0.)
    return img


class GIProcessingMethod:
    """
    An abstract class for various methods used to process the measurement data
    into the ghost image based on them.

    Parameters
    ----------
    model : GIMeasurementModel
        The ghost image measurement model on which to base the processing.

    Class attributes
    ----------------
    name : str
        Short name, typically used to refer to method's results when saving
        it to a file.
    desc : str
        Description of a method to use for plotting.
    """
    name = ""
    desc = ""

    def __init__(self, model: GIMeasurementModel):
        self._measurement_model = model

    def _mt_op(self, n_patterns: Optional[int] = None) -> np.ndarray:
        if n_patterns is None:
            return self._measurement_model.mt_op
        return self._measurement_model.mt_op_part(n_patterns)

    def __call__(self, measurement, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError("Not implemented in the general case!")


class TraditionalGI(GIProcessingMethod):
    """
    Ghost image formation using the traditional approach, that is,
    summation of the illumination patterns weighted by the corresponding
    measurement components after subtracting the mean.

    Parameters
    ----------
    model : GIMeasurementModel
        The ghost image measurement model on which to base the processing.
    """
    name = "gi"
    desc = "Обычное ФИ"

    def __call__(self, measurement) -> np.ndarray:  # pylint: disable=W0221
        """
        Process the measurement using the traditional approach, that is,
        summation of the illumination patterns weighted by the corresponding
        measurement components after subtracting the mean.
        If the measurement is shorter than the available number of patterns,
        only the first `measurement.size` ones are used.

        Parameters
        ----------
        measurement : array_like
            The measurement.

        Returns
        -------
        result : numpy.ndarray
            The processing result.
        """
        illum_patterns = self._measurement_model.illumination_patterns(
            measurement.size)
        result = np.tensordot(measurement - measurement.mean(),
                              illum_patterns - illum_patterns.mean(axis=0),
                              axes=1)/measurement.size
        return result
