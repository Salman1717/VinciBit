import cv2
import numpy as np 
from PIL import Image
import sys 
import os

#Image Preprocessing

TARGET_SIZE = 256
OUTPUT_DIR = "outputs"

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def load_image(path):
    img = Image.open(path).convert("RGB")
    img.save(f"{OUTPUT_DIR}/original.png")
    return img

def resize_to_square(img, size = 256):
    w, h = img.size
    min_dim = min(w, h)

    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    img_cropped = img.crop((left, top, right, bottom))
    img_resized = img_cropped.resize((size,size), Image.LANCZOS)

    img_resized.save(f"{OUTPUT_DIR}/resized.png")

    return img_resized

def enhanced_image(img):
    img_np = np.array(img, dtype = np.uint8)
    img_np = np.ascontiguousarray(img_np)

    #OpenCV Conversion
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    #Gaussian Blur (noise reduction)
    blurred = cv2.GaussianBlur(img_cv, (5, 5), 0)

    #Contrast Enhancement
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit = 2.0 , tileGridSize = (8,8))
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    enhanced_img = Image.fromarray(enhanced_rgb)

    enhanced_img.save(f"{OUTPUT_DIR}/enhanced.png")

    return enhanced_img

def main():
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <image_path>")
        sys.exit(1)

    input_path = sys.argv[1]

    ensure_output_dir()

    print("[VinciBit] Loading image...")
    img = load_image(input_path)

    print("[VinciBit] Resizing image...")
    img_resized = resize_to_square(img, TARGET_SIZE)

    print("[VinciBit] Enhancing image...")
    enhanced_image(img_resized)

    print("[VinciBit] Task 1 complete. Outputs saved in /outputs")


if __name__ == "__main__":
    main()






