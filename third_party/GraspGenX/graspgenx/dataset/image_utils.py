# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualization utils."""

# Standard Library
import io
import math
import os
import sys
import scipy
import imageio
from PIL import Image
from torchvision import transforms
from functools import lru_cache
from typing import Dict, List, Tuple, Union

# Third Party
try:
    import cv2
except ImportError:
    logger.warning("Unable to import cv2")
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

from graspgenx.utils.logging_config import get_logger

logger = get_logger(__name__)


def gen_lut() -> np.ndarray:
    """Generate a label colormap compatible with opencv lookup table.

    Based on Rick Szelski algorithm in Computer Vision: Algorithms and Applications.
    Creates a color lookup table for converting label images to RGB.

    Returns:
        np.ndarray: OpenCV compatible color lookup table of shape (256, 3)
    """
    tobits = lambda x, o: np.array(list(np.binary_repr(x, 24)[o::-3]), np.uint8)
    arr = np.arange(256)
    r = np.concatenate([np.packbits(tobits(x, -3)) for x in arr])
    g = np.concatenate([np.packbits(tobits(x, -2)) for x in arr])
    b = np.concatenate([np.packbits(tobits(x, -1)) for x in arr])
    return np.concatenate([[[b]], [[g]], [[r]]]).T


def labels2rgb(labels: np.ndarray, lut: np.ndarray = None) -> np.ndarray:
    """Convert a label image to an rgb image using a lookup table.

    Applies color mapping to label images for visualization purposes.
    Uses the provided lookup table or generates a default one.

    Args:
        labels: Label image of type np.uint8 2D array
        lut: Lookup table of shape (256, 3) and type np.uint8

    Returns:
        np.ndarray: Colorized label image
    """
    if lut is None:
        lut = gen_lut()
    return cv2.LUT(cv2.merge((labels, labels, labels)), lut)


def depth2rgb(depth: np.ndarray, colormap: int = None) -> np.ndarray:
    """Colorize a depth image using a colormap.

    Converts depth values to RGB colors for better visualization.
    Normalizes depth values to [0, 255] range before applying colormap.

    Args:
        depth: Depth image array
        colormap: Colormap to use, defaults to cv2.COLORMAP_JET

    Returns:
        np.ndarray: Colorized depth image
    """
    if colormap is None:
        colormap = cv2.COLORMAP_JET
    min_depth, max_depth = depth.min(), depth.max()
    if max_depth == min_depth:
        rgb = np.zeros_like(depth)
    else:
        rgb = (depth - min_depth) / (max_depth - min_depth) * 255.0
    return cv2.applyColorMap(rgb.astype(np.uint8), colormap)


def draw_circles(
    im: np.ndarray, circles: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255)
) -> np.ndarray:
    """Draw circles on an image at specified locations.

    Draws filled circles with given radius and color at each point.
    Modifies the input image in-place and returns the result.

    Args:
        im: Input image to draw on
        circles: Nx2 numpy array of circle centers
        color: RGB color tuple for circles

    Returns:
        np.ndarray: Image with circles drawn
    """
    circles = circles.astype(np.int64)
    for n in range(circles.shape[0]):
        circle = tuple(circles[n, :2])
        im = cv2.circle(im, circle, 2, color, 2)
    return im


def draw_lines(
    im: np.ndarray,
    points: np.ndarray,
    lines: np.ndarray,
    colors: List[List[int]] = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw lines on an image connecting specified points.

    Draws lines between pairs of points with specified colors and thickness.
    Each line connects two points from the points array.

    Args:
        im: Input image to draw on
        points: Array of point coordinates
        lines: Nx2 array of point indices to connect
        colors: List of RGB colors for each line
        thickness: Line thickness in pixels

    Returns:
        np.ndarray: Image with lines drawn
    """
    if colors is None:
        colors = [[0, 1, 0] for i in range(len(lines))]

    points = np.array(points, dtype=np.int64)
    for line, color in zip(lines, colors):
        im = cv2.line(
            im, tuple(points[line[0]][:2]), tuple(points[line[1]][:2]), color, thickness
        )
    return im


def draw_rectangles(
    im: np.ndarray,
    boxes: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw rectangles on an image using bounding box coordinates.

    Draws rectangles for each bounding box in the format [xmin, ymin, xmax, ymax].
    Uses OpenCV rectangle drawing function.

    Args:
        im: Input image to draw on
        boxes: Nx4 array of bounding boxes [xmin, ymin, xmax, ymax]
        color: RGB color tuple for rectangles
        thickness: Rectangle border thickness

    Returns:
        np.ndarray: Image with rectangles drawn
    """
    for box in boxes:
        im = cv2.rectangle(im, tuple(box[:2]), tuple(box[2:]), color, thickness)
    return im


def blend_images(images: List[np.ndarray], ratios: List[float] = None) -> np.ndarray:
    """Blend a list of images with specified ratios.

    Combines multiple images using weighted averaging. All images must have
    the same dimensions and be in the range [0, 255].

    Args:
        images: List of images to blend
        ratios: List of blending weights, defaults to equal weights

    Returns:
        np.ndarray: Blended image
    """
    out = None
    if ratios is None:
        ratios = [1.0 / len(images)] * len(images)

    for im, ratio in zip(images, ratios):
        if out is None:
            out = ratio * im.astype(np.float32)
        else:
            out += ratio * im.astype(np.float32)
    return out.astype(np.uint8)


def image_grid(
    image_list: List[np.ndarray], rows: int = -1, margin: int = 10
) -> np.ndarray:
    """Generate a grid layout of images.

    Arranges multiple images in a grid format with specified margins.
    All images must have the same shape and number of channels.

    Args:
        image_list: List of images to arrange in grid
        rows: Number of rows, -1 for single row
        margin: Pixel margin between images

    Returns:
        np.ndarray: Combined grid image

    Raises:
        ValueError: If images have different shapes
    """
    n = len(image_list)
    if rows == -1:
        cols = n
        rows = 1
    else:
        cols = math.ceil(n * 1.0 / rows)

    if any(i.shape != image_list[0].shape for i in image_list[1:]):
        raise ValueError("Not all images have the same shape.")

    img_h, img_w, img_c = image_list[0].shape
    imgmatrix = np.zeros(
        (img_h * cols + margin * (cols - 1), img_w * rows + margin * (rows - 1), img_c),
        np.uint8,
    )
    imgmatrix.fill(255)
    for idx, im in enumerate(image_list):
        x_i = idx % rows
        y_i = idx // rows
        x = x_i * (img_w + margin)
        y = y_i * (img_h + margin)

        imgmatrix[y : y + img_h, x : x + img_w, :] = im

    return imgmatrix


def imshow(im: np.ndarray, name: str = "", is_blocking: bool = True) -> None:
    """Display an image using OpenCV window.

    Shows image in a named window. Can be blocking or non-blocking.
    Blocking mode waits for key press to close window.

    Args:
        im: Image to display
        name: Window name
        is_blocking: Whether to wait for key press
    """
    cv2.imshow(name, im)
    if is_blocking:
        logger.info("press any key to exit...")
        cv2.waitKey(0)
    else:
        cv2.waitKey(1)


def save_image(img: Union[np.ndarray, Image.Image], image_fname: str) -> None:
    """Save an image to file.

    Converts numpy array to PIL Image if needed and saves to specified path.
    Supports common image formats based on file extension.

    Args:
        img: Image as numpy array or PIL Image
        image_fname: Output file path
    """
    if type(img) == np.ndarray:
        img = Image.fromarray(img)
    img.save(image_fname)


def draw_bbox(
    im: np.ndarray,
    boxes: np.ndarray = None,
    centers: np.ndarray = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> Image.Image:
    """Draw bounding boxes and center points on an image.

    Uses PIL drawing functions to draw rectangles and circles.
    Can draw both bounding boxes and center point markers.

    Args:
        im: Input image as numpy array
        boxes: Nx4 array of bounding boxes [xmin, ymin, xmax, ymax]
        centers: Nx2 array of center points
        color: RGB color for drawing
        thickness: Line thickness

    Returns:
        Image.Image: PIL Image with drawings
    """
    img = Image.fromarray(im)
    draw = ImageDraw.Draw(img)

    for box in boxes:
        draw.rectangle(box, outline=color)

    if centers is not None:
        for center in centers:
            xmid = center[0]
            ymid = center[1]
            draw.ellipse(
                (xmid - 10, ymid - 10, xmid + 10, ymid + 10), fill="red", outline="red"
            )
    return img


def convert_label_img_to_seg(seg: np.ndarray) -> np.ndarray:
    """Convert label image to segmentation visualization.

    Assigns random colors to each unique label value for visualization.
    Creates a color-coded segmentation map.

    Args:
        seg: Label image with integer labels

    Returns:
        np.ndarray: RGB segmentation visualization
    """
    map_label_to_color = np.random.uniform(size=[200, 3]) * 255.0
    h, w = seg.shape
    img_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for seg_id in list(set(seg.flatten().tolist())):
        img_rgb[seg == seg_id, :] = map_label_to_color[seg_id]

    return img_rgb


def decompress_png(image_comp: np.ndarray, return_unit8: bool) -> np.ndarray:
    """Decompress PNG image from compressed byte array.

    Reads PNG data from numpy byte array and optionally normalizes to [0,1].
    Uses imageio for PNG decoding.

    Args:
        image_comp: Compressed PNG data as uint8 array
        return_unit8: Whether to return uint8 or float32 [0,1]

    Returns:
        np.ndarray: Decompressed image
    """
    assert image_comp.dtype == "uint8"

    with io.BytesIO(image_comp) as f:
        image = imageio.imread(f, format="png")

    if not return_unit8:
        image = image.astype(np.float32)
        image = image / 255.0

    return image


def convert_img_float_uint8(image: np.ndarray) -> Image.Image:
    """Convert float32 image in range [0,1] to uint8 PIL Image.

    Scales float values from [0,1] to [0,255] and converts to PIL Image.
    Assumes input is float32 with values in valid range.

    Args:
        image: Float32 image with values in [0,1]

    Returns:
        Image.Image: Converted PIL Image

    Raises:
        AssertionError: If input is not float32 or values out of range
    """
    assert isinstance(image, np.ndarray)
    assert not image.dtype == "uint8"
    assert image.dtype == "float32"
    assert np.all(np.logical_and(image >= 0.0, image <= 1.0))
    return Image.fromarray((image * 255).round().astype(np.uint8))


def compress_img(image: Union[np.ndarray, Image.Image], is_unit8: bool) -> np.ndarray:
    """Compress image to PNG format as byte array.

    Converts image to PNG format and returns as compressed byte array.
    Handles both uint8 and float32 input formats.

    Args:
        image: Input image as numpy array or PIL Image
        is_unit8: Whether input is uint8 or float32 [0,1]

    Returns:
        np.ndarray: Compressed PNG data as uint8 array
    """
    _image = np.array(image)

    if is_unit8:
        assert _image.dtype == "uint8"
        assert np.all(np.logical_and(_image >= 0, _image <= 255))
    else:
        assert np.all(np.logical_and(_image >= 0.0, _image <= 1.0)), _image
        assert _image.dtype == "float32"
        image = convert_img_float_uint8(_image)

    with io.BytesIO() as b:
        imageio.imwrite(b, image, format="png")
        b.seek(0)
        image_comp = np.frombuffer(b.read(), dtype="uint8")

    return image_comp


def jitter_gaussian(xyz: torch.Tensor, std: float, clip: float) -> torch.Tensor:
    """Add Gaussian jitter to xyz coordinates.

    Adds random Gaussian noise to point cloud coordinates.
    Clips the noise to specified range to prevent excessive distortion.

    Args:
        xyz: Point cloud coordinates
        std: Standard deviation of Gaussian noise
        clip: Maximum absolute value for noise clipping

    Returns:
        torch.Tensor: Jittered coordinates
    """
    return xyz + torch.clip(torch.randn_like(xyz) * std, -clip, clip)


def add_noise_to_xyz(
    xyz_img: np.ndarray, depth_img: np.ndarray, noise_params: Dict
) -> np.ndarray:
    """Add Gaussian Process noise to ordered point cloud.

    Applies approximate Gaussian Process noise using bicubic interpolation.
    Adapted from DexNet 2.0 codebase for depth sensor simulation.

    Args:
        xyz_img: HxWx3 ordered point cloud
        depth_img: HxW depth image for masking
        noise_params: Dictionary with 'gp_rescale_factor_range' and 'gaussian_scale_range'

    Returns:
        np.ndarray: Noisy point cloud
    """
    assert isinstance(xyz_img, np.ndarray)
    assert isinstance(depth_img, np.ndarray)

    H, W, C = xyz_img.shape
    gp_rescale_factor = np.random.uniform(
        noise_params["gp_rescale_factor_range"][0],
        noise_params["gp_rescale_factor_range"][1],
    )
    gaussian_scale = np.random.uniform(
        noise_params["gaussian_scale_range"][0], noise_params["gaussian_scale_range"][1]
    )
    small_H, small_W = (np.array([H, W]) / gp_rescale_factor).astype(int)
    additive_noise = np.random.normal(
        loc=0.0, scale=gaussian_scale, size=(small_H, small_W, C)
    )
    additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)
    xyz_img[depth_img != 0, :] += additive_noise[depth_img != 0, :]

    return xyz_img


def add_gaussian_noise_to_depth(
    depth_img: np.ndarray, noise_params: Dict
) -> np.ndarray:
    """Add Gaussian noise to depth image.

    Applies zero-mean Gaussian noise with random standard deviation.
    Used for simulating depth sensor noise.

    Args:
        depth_img: HxW depth image
        noise_params: Dictionary with 'gaussian_std_range' parameter

    Returns:
        np.ndarray: Noisy depth image
    """
    assert isinstance(depth_img, np.ndarray)
    std = np.random.uniform(
        noise_params["gaussian_std_range"][0], noise_params["gaussian_std_range"][1]
    )
    rng = np.random.default_rng()
    noise = rng.standard_normal(size=depth_img.shape, dtype=np.float32) * std
    depth_img += noise
    return depth_img


def dropout_random_ellipses(
    depth_img: Union[np.ndarray, torch.Tensor], noise_params: Dict
) -> np.ndarray:
    """Randomly drop ellipses in depth image for robustness.

    Creates random elliptical regions of zero depth to simulate occlusions.
    Uses PyTorch for efficient computation and gamma distribution for ellipse sizes.

    Args:
        depth_img: HxW depth image
        noise_params: Dictionary with ellipse parameters

    Returns:
        np.ndarray: Depth image with elliptical dropouts
    """
    depth_img = (
        torch.tensor(depth_img, dtype=torch.float32)
        if not isinstance(depth_img, torch.Tensor)
        else depth_img.clone()
    )

    num_ellipses_to_dropout = torch.poisson(
        torch.tensor(noise_params["ellipse_dropout_mean"])
    )
    num_ellipses_to_dropout = int(num_ellipses_to_dropout.item())

    valid_pixel_indices = torch.nonzero(
        (depth_img != 0) & (depth_img != float("inf")) & (depth_img != float("-inf"))
    )

    if valid_pixel_indices.shape[0] == 0:
        return depth_img.numpy()

    dropout_centers_indices = torch.randint(
        0, valid_pixel_indices.shape[0], (num_ellipses_to_dropout,)
    )
    dropout_centers = valid_pixel_indices[dropout_centers_indices, :]

    x_radii = torch.round(
        torch.distributions.Gamma(
            noise_params["ellipse_gamma_shape"], noise_params["ellipse_gamma_scale"]
        ).sample((num_ellipses_to_dropout,))
    ).int()
    y_radii = torch.round(
        torch.distributions.Gamma(
            noise_params["ellipse_gamma_shape"], noise_params["ellipse_gamma_scale"]
        ).sample((num_ellipses_to_dropout,))
    ).int()
    angles = torch.randint(0, 360, (num_ellipses_to_dropout,))

    mask = torch.zeros_like(depth_img)

    for i in range(num_ellipses_to_dropout):
        center = dropout_centers[i].tolist()
        x_radius = x_radii[i].item()
        y_radius = y_radii[i].item()
        angle = angles[i].item()

        yy, xx = torch.meshgrid(
            torch.arange(depth_img.shape[0]),
            torch.arange(depth_img.shape[1]),
            indexing="ij",
        )

        cos_a = torch.cos(torch.deg2rad(torch.tensor(angle, dtype=torch.float32)))
        sin_a = torch.sin(torch.deg2rad(torch.tensor(angle, dtype=torch.float32)))

        x_shifted = xx - center[1]
        y_shifted = yy - center[0]

        x_rot = cos_a * x_shifted + sin_a * y_shifted
        y_rot = -sin_a * x_shifted + cos_a * y_shifted

        ellipse_mask = ((x_rot / x_radius) ** 2 + (y_rot / y_radius) ** 2) <= 1
        mask[ellipse_mask] = 1

    depth_img[mask == 1] = 0
    return depth_img.numpy()


@lru_cache(maxsize=2)
def get_xp_yp(rows: int, cols: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate meshgrid coordinates for image processing.

    Creates coordinate grids for bilinear interpolation and warping.
    Cached for efficiency when called repeatedly with same dimensions.

    Args:
        rows: Number of rows
        cols: Number of columns

    Returns:
        Tuple[np.ndarray, np.ndarray]: X and Y coordinate grids
    """
    xp, yp = np.meshgrid(
        np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32)
    )
    return xp, yp


def add_gaussian_shifts(
    depth: Union[np.ndarray, torch.Tensor], std: float = 1 / 2.0
) -> np.ndarray:
    """Add Gaussian shifts to depth image using bilinear interpolation.

    Applies random pixel shifts to simulate camera motion or calibration errors.
    Uses both bilinear and nearest-neighbor interpolation for robust handling.

    Args:
        depth: HxW depth image
        std: Standard deviation of Gaussian shifts

    Returns:
        np.ndarray: Shifted depth image
    """
    depth = (
        torch.tensor(depth, dtype=torch.float32)
        if not isinstance(depth, torch.Tensor)
        else depth.clone()
    )
    rows, cols = depth.shape

    rng = torch.randn((rows, cols, 2), dtype=torch.float32) * std

    xp, yp = get_xp_yp(rows, cols)
    xp = torch.tensor(xp, dtype=torch.float32)
    yp = torch.tensor(yp, dtype=torch.float32)

    xp_interp = torch.clamp(xp + rng[:, :, 0], 0.0, cols - 1)
    yp_interp = torch.clamp(yp + rng[:, :, 1], 0.0, rows - 1)

    depth_interp = F.grid_sample(
        depth.unsqueeze(0).unsqueeze(0),
        torch.stack((xp_interp, yp_interp), dim=-1).unsqueeze(0),
        mode="bilinear",
        align_corners=True,
    ).squeeze()
    robot_depth_interp = F.grid_sample(
        depth.unsqueeze(0).unsqueeze(0),
        torch.stack((xp_interp, yp_interp), dim=-1).unsqueeze(0),
        mode="nearest",
        align_corners=True,
    ).squeeze()

    depth[depth == float("inf")] = depth.min()
    combine_depth_interp = torch.where(
        (depth_interp > depth.max() - 0.6) | (depth_interp < depth.min()),
        robot_depth_interp,
        depth_interp,
    )
    combine_depth_interp[combine_depth_interp != combine_depth_interp] = float("inf")

    return combine_depth_interp.numpy()


def mask_object_edge(
    depth: Union[np.ndarray, torch.Tensor], thresh: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Mask object edges in depth image using Sobel edge detection.

    Detects edges using Sobel filters and masks them to remove boundary artifacts.
    Useful for cleaning depth images before processing.

    Args:
        depth: HxW depth image
        thresh: Threshold for edge detection

    Returns:
        Tuple[np.ndarray, np.ndarray]: Masked depth image and edge mask
    """
    depth = (
        torch.tensor(depth, dtype=torch.float32)
        if not isinstance(depth, torch.Tensor)
        else depth.clone()
    )
    img = depth.clone()
    img[img == float("inf")] = 0
    img = (img / img.max() * 256).to(torch.uint8)

    sobel_x = (
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    sobel_y = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    img = img.unsqueeze(0).unsqueeze(0).float()

    sobelx = F.conv2d(img, sobel_x, padding=1).squeeze()
    sobely = F.conv2d(img, sobel_y, padding=1).squeeze()

    index_x = (sobelx > -thresh) & (sobelx < thresh)
    index_y = (sobely > -thresh) & (sobely < thresh)
    mask_index = index_x & index_y

    depth[~mask_index] = float("inf")
    return depth.numpy(), mask_index.numpy()


def add_kinect_noise_to_depth(depth: np.ndarray, noise_params: Dict) -> np.ndarray:
    """Add Kinect-style noise to depth image.

    Combines edge masking and Gaussian shifts to simulate Kinect sensor noise.
    Typical noise pattern includes edge artifacts and depth-dependent noise.

    Args:
        depth: HxW depth image
        noise_params: Dictionary with 'std_range' and 'thresh_range' parameters

    Returns:
        np.ndarray: Depth image with Kinect-style noise
    """
    assert isinstance(depth, np.ndarray)

    std = np.random.uniform(noise_params["std_range"][0], noise_params["std_range"][1])
    thresh = np.random.uniform(
        noise_params["thresh_range"][0], noise_params["thresh_range"][1]
    )

    depth, _ = mask_object_edge(depth, thresh)
    noisy_depth = add_gaussian_shifts(depth, std)
    return noisy_depth


class NormalizeInverse(transforms.Normalize):
    """Inverse normalization transform for denormalizing images.

    Computes inverse of normalization parameters to reverse the normalization.
    Useful for converting normalized tensors back to original image range.
    """

    def __init__(
        self,
        mean: Union[List[float], torch.Tensor],
        std: Union[List[float], torch.Tensor],
    ):
        """Initialize inverse normalization transform.

        Args:
            mean: Mean values used in original normalization
            std: Standard deviation values used in original normalization
        """
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply inverse normalization to tensor.

        Args:
            tensor: Normalized tensor to denormalize

        Returns:
            torch.Tensor: Denormalized tensor
        """
        return super().__call__(tensor.clone())


normalize_rgb = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


denormalize_rgb = transforms.Compose(
    [NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)
