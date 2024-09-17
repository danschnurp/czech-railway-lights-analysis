import cv2
import numpy as np
import os


def rotate_and_crop(img, destination, angle, crop_type='fit'):
    """
    Rotates the image by a given angle and crops it.
    :param img: input image
    :param destination: output file path
    :param angle: rotation angle in degrees
    :param crop_type: 'fit' to maintain aspect ratio, 'left' or 'right' for additional cropping
    :return: new filename
    """
    suffix = f"_rot_{angle}"
    if crop_type in ['left', 'right']:
        suffix += f"_crop_{crop_type}"

    name = f"{destination}{suffix}.jpg"


    height, width = img.shape[:2]
    center = (width / 2, height / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_matrix[0, 2] += bound_w / 2 - center[0]
    rotation_matrix[1, 2] += bound_h / 2 - center[1]

    rotated = cv2.warpAffine(img, rotation_matrix, (bound_w, bound_h))

    if crop_type == 'fit':
        new_width, new_height = aspect(width, height, bound_w, bound_h)
        start_x = int((bound_w - new_width) / 2)
        start_y = int((bound_h - new_height) / 2)
        cropped = rotated[start_y:start_y + int(new_height), start_x:start_x + int(new_width)]
    elif crop_type == 'left':
        if bound_w > bound_h:
            cropped = rotated[:, int(bound_w * 0.1):]
        else:
            cropped = rotated[int(bound_h * 0.1):, :]
    elif crop_type == 'right':
        if bound_w > bound_h:
            cropped = rotated[:, :int(bound_w * 0.9)]
        else:
            cropped = rotated[:int(bound_h * 0.9), :]
    else:
        cropped = rotated  # No cropping if invalid crop_type


    return cropped


def aspect(width, height, aspect_x, aspect_y):
    """
    Calculates dimensions to crop the image to fit after rotation.
    :param width: original width
    :param height: original height
    :param aspect_x: rotated width
    :param aspect_y: rotated height
    :return: new width, new height
    """
    old_ratio = width / height
    new_ratio = aspect_x / aspect_y
    if old_ratio < 1:
        old_ratio = height / width
        new_ratio = aspect_y / aspect_x
    reduction_x = aspect_x - width
    reduction_y = aspect_y - height
    if abs(old_ratio - new_ratio) < 0.1:
        if new_ratio > old_ratio:
            return width / new_ratio - reduction_x, height / new_ratio - reduction_y
        else:
            return width / new_ratio - reduction_x, height / new_ratio - reduction_y
    elif new_ratio > old_ratio:
        return width / new_ratio, height / new_ratio
    else:
        return width / new_ratio, height / new_ratio


def add_noise(img, destination):
    """
    Adds noise to the image.
    :param img: input image
    :param destination: output file path
    :return: new filename
    """
    name = f"{destination}_noise.jpg"


    # Convert to float32
    img_float = img.astype(np.float32) / 255.0

    # Add Gaussian noise
    noise = np.random.normal(0, 0.1, img.shape)
    noisy = np.clip(img_float + noise, 0, 1)

    # Convert back to uint8
    noisy_img = (noisy * 255).astype(np.uint8)


    return noisy_img


def brightness_contrast(img, destination, alpha, beta):
    """
    Adjusts brightness and contrast of the image.
    :param img: input image
    :param destination: output file path
    :param alpha: contrast control (1.0-3.0)
    :param beta: brightness control (0-100)
    :return: new filename
    """
    name = f"{destination}_bright_contrast.jpg"


    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted

