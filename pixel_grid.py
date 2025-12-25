import numpy as np 
from PIL import Image, ImageDraw
import json
import os

#Pixel Grid Generation

INPUT_IMAGE = "outputs/quantized.png"
PALETTE_JSON = "outputs/palette.json"

GRID_PREVIEW = "outputs/grid_preview.png"
PIXEL_GRID_JSON = "outputs/pixel_grid.json"
STEPS_JSON = "outputs/steps.json"

GRID_SIZE = 32

def load_palette():
    with open(PALETTE_JSON, "r") as f:
        palette = json.load(f)
    id_to_rgb = {
        item["id"]: tuple(item["rgb"]) for item in palette
    }
    return palette, id_to_rgb

def load_image():
    img = Image.open(INPUT_IMAGE).convert("RGB")
    return img

def closest_color_id(pixel, palette):
    distances = []
    for p in palette:
        prgb = np.array(p["rgb"])
        distances.append(np.linalg.norm(pixel - prgb))
    return palette[int(np.argmin(distances))]["id"]

def generate_pixel_grid(img, palette):
    w, h = img.size
    cell_w = w // GRID_SIZE
    cell_h = h // GRID_SIZE

    img_np = np.array(img, dtype = np.uint8)

    cells = []

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            x0 = x * cell_w
            y0 = y * cell_h
            block = img_np[y0 : y0 + cell_h, x0 : x0 + cell_w]

            avg_color = block.reshape(-1,3).mean(axis = 0)
            color_id = closest_color_id(avg_color, palette)

            cells.append({
                "x" : x,
                "y" : y,
                "colorId" : color_id
            })
        
    return cells, cell_w, cell_h


def save_grid_preview(cells, id_to_rgb, cell_w, cell_h):
    img = Image.new(
        "RGB",
        (GRID_SIZE * cell_w, GRID_SIZE * cell_h),
        (255, 255, 255)
    )

    draw = ImageDraw.Draw(img)

    for cell in cells:
        x = cell["x"]
        y = cell["y"]
        color = id_to_rgb[cell["colorId"]]

        x0 = x * cell_w
        y0 = y * cell_h
        x1 = x0 + cell_w
        y1 = y0 + cell_h

        draw.rectangle([x0, y0, x1, y1], fill = color)
    
    # Grid lines
    for i in range(GRID_SIZE + 1):
        draw.line(
            [(0, i * cell_h), (GRID_SIZE * cell_w, i * cell_h)],
            fill = (0, 0, 0),
            width = 1
        )
        draw.line(
            [(i * cell_w, 0), (i * cell_w, GRID_SIZE * cell_h)],
            fill = (0, 0, 0),
            width = 1
        )

    img.save(GRID_PREVIEW)

def save_pixel_grid_json(cells):
    data = {
        "gridSize" : GRID_SIZE,
        "cells" : cells
    }

    with open(PIXEL_GRID_JSON, "w") as f:
        json.dump(data, f, indent = 2)

def save_steps(cells):
    steps = []
    step_id = 0

    for cell in cells:
        steps.append({
            "step" : step_id,
            "x" : cell["x"],
            "y" : cell["y"],
            "colorId" : cell["colorId"]
        })
        step_id += 1
    
    with open(STEPS_JSON, "w") as f:
        json.dump(steps, f, indent = 2)

def main():
    if not os.path.exists(INPUT_IMAGE):
        raise FileNotFoundError("Run quantize.py first.")
    
    print("VinciBit ==> Loading palette & image...")
    palette , id_to_rgb = load_palette()
    img = load_image()

    print("VinciBit ==> Generating pixel grid...")
    cells, cell_w, cell_h = generate_pixel_grid(img, palette)

    print("VinciBit ==> Saving grid preview...")
    save_grid_preview(cells, id_to_rgb, cell_w, cell_h)

    print("VinciBit ==> Saving pixel grid JSON...")
    save_pixel_grid_json(cells)

    print("VinciBit ==> Saving drawing steps...")
    save_steps(cells)

    print("VinciBit ==> TASK 3 COMPLETE!")

if __name__ == "__main__":
    main()









    