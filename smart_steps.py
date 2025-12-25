# smart_steps.py
# Salman Mhaskar

import json
import numpy as np 
import os
from PIL import Image, ImageDraw
from collections import defaultdict, deque

PIXEL_GRID_JSON = "outputs/pixel_grid.json"
PALETTE_JSON = "outputs/palette.json"

STEPS_BY_COLOR = "outputs/steps_by_color.json"
STEPS_BY_REGION = "outputs/steps_by_region.json"
INSTRUCTIONS_PREVIEW = "outputs/instructions_preview.png"

GRID_PREVIEW =  "outputs/grid_preview.png"


def load_data():
    with open(PIXEL_GRID_JSON) as f:
        grid = json.load(f)\

    with open(PALETTE_JSON) as f:
        palette = json.load(f)
    
    id_to_rgb = {p["id"] : tuple(p["rgb"]) for p in palette}

    return grid, palette, id_to_rgb

#Group by COLOR

def group_by_color(cells):
    color_groups = defaultdict(list)
    for cell in cells:
        color_groups[cell["colorId"]].append(cell)
    return color_groups

def save_steps_by_color(color_groups):
    steps = []
    step_id = 0

    for color_id, cells in color_groups.items():
        steps.append({
            "step" : step_id,
            "type" : "color",
            "colorId" : color_id,
            "cells" : cells
        })
        step_id += 1
    
    with open(STEPS_BY_COLOR, "w") as f:
        json.dump(steps, f, indent = 2)

#Region Detection

def get_neighbors(x, y, grid_size):
    neighbors = []

    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid_size and 0 <= ny <grid_size:
            neighbors.append((nx, ny))
    
    return neighbors

def group_by_region(cells, grid_size):
    grid = {}
    
    for c in cells:
        grid[(c["x"], c["y"])] = c 
    
    visited = set()
    regions = []

    for cell in cells:
        pos = (cell["x"], cell["y"])
        if pos in visited:
            continue
        
        region = []
        queue = deque([pos])
        visited.add(pos)

        while queue:
            cx, cy = queue.popleft()
            current = grid[(cx, cy)]
            region.append(current)

            for nx, ny in get_neighbors(cx, cy, grid_size):
                if (nx, ny) in grid:
                    neighbor = grid[(nx, ny)]
                    if neighbor["colorId"] == current["colorId"] and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
            
        regions.append({
            "colorId" : cell["colorId"],
            "cells" : region
        })
    
    return regions

def save_steps_by_region(regions):
    steps = []

    for i, region in enumerate(regions):
        steps.append({
            "steps" : i,
            "type" : "region",
            "colorId" : region["colorId"],
            "cells" : region["cells"]
        })
    
    with open(STEPS_BY_REGION, "w") as f:
        json.dump(steps, f, indent = 2)

# ---
# Instructions Preview

def generate_instructions_preview(regions, id_to_rgb, grid_size):
    img = Image.open(GRID_PREVIEW).convert("RGB")

    draw = ImageDraw.Draw(img)

    cell_size = img.size[0]

    first_region = regions[0]
    highlight_color = (255, 255, 0)

    for cell in first_region["cells"]:
        x0 = cell["x"] * cell_size
        y0 = cell["y"] * cell_size
        x1 = x0 + cell_size
        y1 = y0 + cell_size
        draw.rectangle([x0, y0, x1, y1], outline = highlight_color, width = 3)
    
    img.save(INSTRUCTIONS_PREVIEW)

def main():
    if not os.path.exists(PIXEL_GRID_JSON):
        raise FileNotFoundError("Run previous files first.")
    
    print("VinciBit ==> Loading grid & palette...")
    grid, palette, id_to_rgb = load_data()
    cells = grid["cells"]
    grid_size = grid["gridSize"]

    print("VinciBit ==> Grouping steps by color...")
    color_groups = group_by_color(cells)
    save_steps_by_color(color_groups)

    print("VinciBit ==> Detecting connected regions...")
    regions = group_by_region(cells,grid_size)
    save_steps_by_region(regions)

    print("VinciBit ==> Generating instruction preview...")
    generate_instructions_preview(regions, id_to_rgb, grid_size)

    print("VinciBit ==> COMPLETE!")

if __name__ == "__main__":
    main()

