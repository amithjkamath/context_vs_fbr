# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import numpy as np

from monai.transforms.utils import rescale_array

__all__ = ["create_circle_image_2d", "create_circle_image_3d", "create_flat_circle_image_2d",
           "create_rect_image_2d", "generate_random_mask_at", "generate_flat_mask_at"]


# Copied over from monai/data/synthetic.py
def create_circle_image_2d(
    width: int,
    height: int,
    center_x: int,
    center_y: int,
    rad: int = 30,
    num_objs: int = 1,
    noise_max: float = 0.0,
    num_seg_classes: int = 1,
    channel_dim: Optional[int] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a noisy 2D image with `num_objs` circles and a 2D mask image. The maximum and minimum radii of the circles
    are given as `rad_max` and `rad_min`. The mask will have `num_seg_classes` number of classes for segmentations labeled
    sequentially from 1, plus a background class represented as 0. If `noise_max` is greater than 0 then noise will be
    added to the image taken from the uniform distribution on range `[0,noise_max)`. If `channel_dim` is None, will create
    an image without channel dimension, otherwise create an image with channel dimension as first dim or last dim.

    Args:
        width: width of the image. The value should be larger than `2 * rad_max`.
        height: height of the image. The value should be larger than `2 * rad_max`.
        num_objs: number of circles to generate. Defaults to `12`.
        rad_max: maximum circle radius. Defaults to `30`.
        rad_min: minimum circle radius. Defaults to `5`.
        noise_max: if greater than 0 then noise will be added to the image taken from
            the uniform distribution on range `[0,noise_max)`. Defaults to `0`.
        num_seg_classes: number of classes for segmentations. Defaults to `5`.
        channel_dim: if None, create an image without channel dimension, otherwise create
            an image with channel dimension as first dim or last dim. Defaults to `None`.
        random_state: the random generator to use. Defaults to `np.random`.
    """

    min_size = min(width, height)
    if min_size <= 2 * rad:
        raise ValueError("the minimal size of the image should be larger than `2 * rad_max`.")
    if center_x >= width:
        raise ValueError("the center_x should be smaller than the image width.")
    if center_y >= height:
        raise ValueError("the center_y should be smaller than the image height.")

    image = np.zeros((width, height))
    rs: np.random.RandomState = np.random.random.__self__ if random_state is None else random_state  # type: ignore

    for _ in range(num_objs):
        x = center_x
        y = center_y
        spy, spx = np.ogrid[-x : width - x, -y : height - y]
        circle = (spx * spx + spy * spy) <= rad * rad

        if num_seg_classes > 1:
            image[circle] = np.ceil(rs.random() * num_seg_classes)
        else:
            image[circle] = rs.random() * 0.5 + 0.5

    labels = np.ceil(image).astype(np.int32, copy=False)

    norm = rs.uniform(0, num_seg_classes * noise_max, size=image.shape)
    noisyimage: np.ndarray = rescale_array(np.maximum(image, norm))  # type: ignore

    # Force the inside of the circle to be all flat white.
    noisyimage[circle] = 1.0

    if channel_dim is not None:
        if not (isinstance(channel_dim, int) and channel_dim in (-1, 0, 2)):
            raise AssertionError("invalid channel dim.")
        if channel_dim == 0:
            noisyimage = noisyimage[None]
            labels = labels[None]
        else:
            noisyimage = noisyimage[..., None]
            labels = labels[..., None]

    return noisyimage, labels


def create_circle_image_3d(
    height: int,
    width: int,
    depth: int,
    center_x: int,
    center_y: int,
    center_z: int,
    num_objs: int = 12,
    rad: int = 30,
    noise_max: float = 0.0,
    num_seg_classes: int = 5,
    channel_dim: Optional[int] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a noisy 3D image and segmentation.

    Args:
        height: height of the image. The value should be larger than `2 * rad_max`.
        width: width of the image. The value should be larger than `2 * rad_max`.
        depth: depth of the image. The value should be larger than `2 * rad_max`.
        num_objs: number of circles to generate. Defaults to `12`.
        rad_max: maximum circle radius. Defaults to `30`.
        rad_min: minimum circle radius. Defaults to `5`.
        noise_max: if greater than 0 then noise will be added to the image taken from
            the uniform distribution on range `[0,noise_max)`. Defaults to `0`.
        num_seg_classes: number of classes for segmentations. Defaults to `5`.
        channel_dim: if None, create an image without channel dimension, otherwise create
            an image with channel dimension as first dim or last dim. Defaults to `None`.
        random_state: the random generator to use. Defaults to `np.random`.

    See also:
        :py:meth:`~create_test_image_2d`
    """
    min_size = min(width, height, depth)
    if min_size <= 2 * rad:
        raise ValueError("the minimal size of the image should be larger than `2 * rad_max`.")

    image = np.zeros((width, height, depth))
    rs: np.random.RandomState = np.random.random.__self__ if random_state is None else random_state  # type: ignore

    for _ in range(num_objs):
        x = center_x
        y = center_y
        z = center_z
        spy, spx, spz = np.ogrid[-x : width - x, -y : height - y, -z : depth - z]
        circle = (spx * spx + spy * spy + spz * spz) <= rad * rad

        if num_seg_classes > 1:
            image[circle] = np.ceil(rs.random() * num_seg_classes)
        else:
            image[circle] = rs.random() * 0.5 + 0.5

    labels = np.ceil(image).astype(np.int32, copy=False)

    norm = rs.uniform(0, num_seg_classes * noise_max, size=image.shape)
    noisyimage: np.ndarray = rescale_array(np.maximum(image, norm))  # type: ignore

    # Force the inside of the circle to be all flat white.
    noisyimage[circle] = 1.0

    if channel_dim is not None:
        if not (isinstance(channel_dim, int) and channel_dim in (-1, 0, 3)):
            raise AssertionError("invalid channel dim.")
        noisyimage, labels = (
            (noisyimage[None], labels[None]) if channel_dim == 0 else (noisyimage[..., None], labels[..., None])
        )

    return noisyimage, labels


def create_flat_circle_image_2d(
    width: int,
    height: int,
    center_x: int,
    center_y: int,
    gray_in: int,
    gray_out: int,
    rad: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a noisy 2D image with `num_objs` circles and a 2D mask image. The maximum and minimum radii of the circles
    are given as `rad_max` and `rad_min`. The mask will have `num_seg_classes` number of classes for segmentations labeled
    sequentially from 1, plus a background class represented as 0. If `noise_max` is greater than 0 then noise will be
    added to the image taken from the uniform distribution on range `[0,noise_max)`. If `channel_dim` is None, will create
    an image without channel dimension, otherwise create an image with channel dimension as first dim or last dim.

    Args:
        width: width of the image. The value should be larger than `2 * rad_max`.
        height: height of the image. The value should be larger than `2 * rad_max`.
        center_x: location of center in the horizontal dimension, in pixels.
        center_y: location of center in the vertical dimension, in pixels.
        rad: circle radius. Defaults to `30`.
        gray_in: gray value of pixels within circle, in range 0 to 255.
        gray_out: gray value of pixels outside circle, in range 0 to 255.
    """

    min_size = min(width, height)
    if min_size <= 2 * rad:
        raise ValueError("the minimal size of the image should be larger than `2 * rad_max`.")
    if center_x >= width:
        raise ValueError("the center_x should be smaller than the image width.")
    if center_y >= height:
        raise ValueError("the center_y should be smaller than the image height.")

    image = np.ones((width, height)) * gray_out
    x = center_x
    y = center_y
    spy, spx = np.ogrid[-x : width - x, -y : height - y]
    circle = (spx * spx + spy * spy) <= rad * rad

    image[circle] = gray_in
    labels = image == gray_in
    
    # Normalize image to be in range 0 to 1, independent of max pixel.
    image = image / 255.0

    return image, labels


# Inspired by monai/data/synthetic.py
def create_rect_image_2d(
    width: int,
    height: int,
    num_objs: int = 12,
    dim_max: int = 100,
    dim_min: int = 20,
    noise_max: float = 0.0,
    num_seg_classes: int = 5,
    channel_dim: Optional[int] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a noisy 2D image with `num_objs` rectangles and a 2D mask image. The maximum and minimum lengths of the rectangles
    are given as `dim_max` and `dim_min`. The mask will have `num_seg_classes` number of classes for segmentations labeled
    sequentially from 1, plus a background class represented as 0. If `noise_max` is greater than 0 then noise will be
    added to the image taken from the uniform distribution on range `[0,noise_max)`. If `channel_dim` is None, will create
    an image without channel dimension, otherwise create an image with channel dimension as first dim or last dim.

    Args:
        width: width of the image. The value should be larger than `2 * dim_max`.
        height: height of the image. The value should be larger than `2 * dim_max`.
        num_objs: number of circles to generate. Defaults to `12`.
        dim_max: maximum rectangle length. Defaults to `30`.
        dim_min: minimum rectangle length. Defaults to `5`.
        noise_max: if greater than 0 then noise will be added to the image taken from
            the uniform distribution on range `[0,noise_max)`. Defaults to `0`.
        num_seg_classes: number of classes for segmentations. Defaults to `5`.
        channel_dim: if None, create an image without channel dimension, otherwise create
            an image with channel dimension as first dim or last dim. Defaults to `None`.
        random_state: the random generator to use. Defaults to `np.random`.
    """

    if dim_max <= dim_min:
        raise ValueError("`dim_min` should be less than `dim_max`.")
    if dim_min < 1:
        raise ValueError("`dim_min` should be no less than 1.")
    min_size = min(width, height)
    if min_size <= dim_max:
        raise ValueError("the minimal size of the image should be larger than `dim_max`.")

    image = np.zeros((width, height))
    rs: np.random.RandomState = np.random.random.__self__ if random_state is None else random_state  # type: ignore

    for _ in range(num_objs):
        rect = np.zeros((width, height), dtype=bool)
        while not np.any(rect):
            x = rs.randint(dim_min, dim_max)
            y = rs.randint(dim_min, dim_max)
            w = rs.randint(dim_min, dim_max) // 2
            h = rs.randint(dim_min, dim_max) // 2
            rect[x - w: x + w, y - h: y + h] = True

        if num_seg_classes > 1:
            image[rect] = np.ceil(rs.random() * num_seg_classes)
        else:
            image[rect] = rs.random() * 0.5 + 0.5

    labels = np.ceil(image).astype(np.int32, copy=False)

    norm = rs.uniform(0, num_seg_classes * noise_max, size=image.shape)
    noisyimage: np.ndarray = rescale_array(np.maximum(image, norm))  # type: ignore

    noisyimage[rect] = 1.0

    if channel_dim is not None:
        if not (isinstance(channel_dim, int) and channel_dim in (-1, 0, 2)):
            raise AssertionError("invalid channel dim.")
        if channel_dim == 0:
            noisyimage = noisyimage[None]
            labels = labels[None]
        else:
            noisyimage = noisyimage[..., None]
            labels = labels[..., None]

    return noisyimage, labels


def generate_random_mask_at(
    width: int,
    height: int,
    x_center: int,
    y_center: int,
    rad: int,
    noise_max: float = 0.0,
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function generates a random perturbation.
    """
    min_size = min(width, height)
    if min_size <= 2 * rad:
        raise ValueError("the minimal size of the image should be larger than `2 * rad_max`.")

    rs: np.random.RandomState = np.random.random.__self__ \
        if random_state is None else random_state  # type: ignore

    norm = rs.uniform(0, noise_max, size=(width, height))    
    spy, spx = np.ogrid[-x_center : width - x_center, -y_center : height - y_center]
    circle = (spx * spx + spy * spy) >= rad * rad
    norm[circle] = -1

    labels = np.zeros((width, height))
    labels[norm >= 0] = 1

    norm[circle] = 0

    return norm, labels


def generate_flat_mask_at(
    width: int,
    height: int,
    x_center: int,
    y_center: int,
    rad: int,
    val: np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function generates a flat perturbation.
    """
    min_size = min(width, height)
    if min_size <= 2 * rad:
        raise ValueError("the minimal size of the image should be larger than `2 * rad_max`.")

    image = np.zeros((width, height))
    spy, spx = np.ogrid[-x_center : width - x_center, -y_center : height - y_center]
    circle = (spx * spx + spy * spy) <= rad * rad
    image[circle] = val

    labels = np.zeros((width, height))
    labels[image == val] = 1

    return image, labels