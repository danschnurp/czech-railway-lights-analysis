import cv2
import numpy as np
from scipy.fftpack import dct


def calculate_phash(image, hash_size=32, highfreq_factor=4):
    """
    Calculate perceptual hash for an image.
    """
    img = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (hash_size, hash_size),
                     interpolation=cv2.INTER_LANCZOS4)
    pixels = np.array(img)

    # Calculate Discrete Cosine Transform
    dct_result = dct(dct(pixels.astype(float), axis=0), axis=1)

    # Extract low frequencies
    dct_low = dct_result[:hash_size // highfreq_factor,
              :hash_size // highfreq_factor]

    # Calculate median value
    med = np.median(dct_low)

    # Generate hash
    hash_bits = (dct_low > med).flatten()
    hash_value = 0

    for bit in hash_bits:
        hash_value = (hash_value << 1) | bit

    return format(hash_value, 'x').zfill(hash_size)


def hamming_distance(hash1, hash2):
    """
    Calculate Hamming distance between two hashes.
    """
    return bin(int(hash1, 16) ^ int(hash2, 16)).count('1')


def calculate_pictures_similarity(img1, img2):
    return hamming_distance(calculate_phash(img1), calculate_phash(img2))
