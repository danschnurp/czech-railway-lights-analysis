import os
import shutil

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from pathlib import Path

from general_utils import get_jpg_files
from image_utils import crop_sides_percentage, calculate_aspect_ratio, crop_top_bottom_percentage


def extract_rgb_histogram(image_path, bins):
    """
    Extract RGB histogram features from an image.
    Using fewer bins (32 instead of 256) to reduce feature dimensionality.
    """
    # Read and convert image to RGB
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # if calculate_aspect_ratio(img)[0] > 0.8:
    #     print(image_path, print(calculate_aspect_ratio(img)))
    #     os.remove(image_path)
    #     cv2.imshow(" ", img)
    #     cv2.waitKey(0)
    # cv2.imshow("", img)
    # cv2.waitKey(0)
    img = crop_sides_percentage(img, crop_percentage=25)
    # cv2.imshow("", img)
    # cv2.waitKey(0)
    img = crop_top_bottom_percentage(img, crop_percentage=5)
    # cv2.imshow("", img)
    # cv2.waitKey(0)
    # Calculate histogram for each channel
    hist_r = cv2.calcHist([img], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([img], [2], None, [bins], [0, 256])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # # Normalize histograms
    # hist_r = cv2.normalize(hist_r, hist_r).flatten()
    # hist_g = cv2.normalize(hist_g, hist_g).flatten()
    # hist_b = cv2.normalize(hist_b, hist_b).flatten()

    # Concatenate the histograms
    hist_features = np.array([hist_r.squeeze(), hist_g.squeeze(), hist_b.squeeze(), gray_hist.squeeze()])
    hist_features = np.mean(hist_features, axis=0)

    # return gray_hist.squeeze()
    return hist_features

def cluster_images(image_paths, n_clusters, bins):
    """
    Cluster images using RGB histograms and GMM.
    Returns the clustered images and their features.
    """

    # Extract features for all images
    features = []
    valid_paths = []

    for path in image_paths:
        try:
            hist_features = extract_rgb_histogram(path, bins)
            features.append(hist_features)
            valid_paths.append(path)
        except Exception as e:
            print(f"Error processing {path}: {e}")

    features = np.array(features)

    # Fit GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    clusters = gmm.fit_predict(features)

    return valid_paths, features, clusters, gmm


def visualize_clusters(image_paths, clusters, max_images_per_cluster=5):
    """
    Create a visualization of the clustered images.
    Shows representative images from each cluster.
    """
    n_clusters = len(np.unique(clusters))
    # todo upravit vizualizaci aby se tam vescko veslo
    # Create a figure with a row for each cluster
    plt.figure(figsize=(10,  n_clusters))

    for cluster_id in range(n_clusters):
        # Get images in this cluster
        cluster_images = [path for path, c in zip(image_paths, clusters) if c == cluster_id]
        n_images = min(len(cluster_images), max_images_per_cluster)

        for i in range(n_images):
            plt.subplot(n_clusters, max_images_per_cluster,
                        cluster_id * max_images_per_cluster + i + 1)

            img = cv2.imread(cluster_images[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.axis('off')

            if i == 0:
                plt.title(f'Cluster {cluster_id}')

    plt.tight_layout()
    plt.show()


def analyze_clusters(features, clusters, gmm):
    """
    Analyze the clustering results.
    Shows cluster sizes and feature importance.
    """
    n_clusters = len(np.unique(clusters))

    # Count images in each cluster
    cluster_sizes = np.bincount(clusters)

    # Calculate cluster centers
    cluster_centers = gmm.means_

    # Plot cluster sizes
    plt.figure(figsize=(10, 4))
    plt.bar(range(n_clusters), cluster_sizes)
    plt.title('Number of Images per Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Images')
    plt.show()

    # Plot feature importance (using variance of cluster centers)
    feature_importance = np.var(cluster_centers, axis=0)
    plt.figure(figsize=(10, 4))
    plt.plot(feature_importance)
    plt.title('Feature Importance (Variance of Cluster Centers)')
    plt.xlabel('Feature Index')
    plt.ylabel('Variance')
    plt.show()


def main(image_paths, n_clusters=20, bins=256):
    """
    Main function to run the complete clustering pipeline.
    """
    # Cluster the images
    image_paths, features, clusters, gmm = cluster_images(
        image_paths, n_clusters, bins
    )

    # Visualize the results
    visualize_clusters(image_paths, clusters)

    # Analyze the clusters
    analyze_clusters(features, clusters, gmm)

    os.mkdir("./data")
    for i in range(n_clusters):
        os.mkdir(f"./data/cluster{i}")

    for i, j in zip(image_paths, clusters):
            img = cv2.imread(i)
            pth = Path(i)
            cv2.imwrite(f"./data/cluster{j}/{pth.name}", img)


    return image_paths, features, clusters, gmm


if __name__ == '__main__':
    image_paths = get_jpg_files("../reconstructed")


    main(image_paths)
