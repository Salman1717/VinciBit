import cv2
import numpy as np 
from PIL import Image
import json
import os

# Color Quantization

INPUT_IMAGE = "outputs/enhanced.png"
OUTPUT_IMAGE = "outputs/quantized.png"
PALETTE_IMAGE = "outputs/palette.png"
PALETTE_JSON = "outputs/palette.json"

K_COLORS = 8
MAX_ITERS = 10

def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype = np.uint8)


def initialize_centroids(pixels, k):
    indices = np.random.choice(len(pixels), k, replace = False)
    return pixels[indices].astype(np.float32)

def assign_clusters(pixels, centroids):
    distances = np.linalg.norm(
        pixels[:, None] - centroids[None, :], axis = 2
    )
    return np.argmin(distances, axis = 1)

def update_centroids(pixels, labels, k):
    new_centroids = []
    for i in range(k):
        cluster_pixels = pixels[labels == i]
        if len(cluster_pixels) == 0:
            new_centroids.append(pixels[np.random.randint(0, len(pixels))])
        else:
            new_centroids.append(cluster_pixels.mean(axis = 0))
    return np.array(new_centroids, dtype = np.float32)

def k_means(pixels, k, max_iter = 10):
    centroids = initialize_centroids(pixels, k)

    for _ in range(max_iter):
        labels = assign_clusters(pixels, centroids)
        centroids = update_centroids(pixels, labels, k)

    return centroids.astype(np.uint8), labels


def save_quantized_image(pixels, labels, centroids, shape):
    quantized_pixels = centroids[labels]
    quantized_pixels = quantized_pixels.reshape(shape)

    img = Image.fromarray(quantized_pixels.astype(np.uint8))
    img.save(OUTPUT_IMAGE)


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0],rgb[1],rgb[2])

def save_palette(centroids):
    palette = []
    for idx, color in enumerate(centroids):
        palette.append({
            "id" : idx,
            "rgb" : color.tolist(),
            "hex" : rgb_to_hex(color)
        })
    
    with open(PALETTE_JSON, "w") as f:
        json.dump(palette, f, indent = 2)

    
    # palette image

    swatch_size = 64
    palette_img = Image.new(
        "RGB",
        (swatch_size * len(centroids), swatch_size)
    )

    for i, color in enumerate(centroids):
        swatch  = Image.new("RGB", (swatch_size, swatch_size), tuple(color))
        palette_img.paste(swatch, ( i * swatch_size, 0))
    
    palette_img.save(PALETTE_IMAGE)


def main():
    if not os.path.exists(INPUT_IMAGE):
        raise FileNotFoundError("Enhanced image not found")
    
    print("VinciBit ==> Loading enhanced image...")
    img = load_image(INPUT_IMAGE)

    h, w, c = img.shape
    pixels = img.reshape(-1,3)

    print("VinciBit ==> Runing K-Means Color Quantization...")
    centroids, labels = k_means(pixels, K_COLORS, MAX_ITERS)

    print("VinciBit ==> Saving Quantized Image...")
    save_quantized_image(pixels, labels, centroids, (h, w, c))

    print("VinciBit ==> Saving palette...")
    save_palette(centroids)

    print("VinciBit ==> COMPLETEE!")


if __name__ == "__main__":
    main()




