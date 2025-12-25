# smart_steps.py
# Salman Mhaskar

import json
import numpy as np 
import os
from PIL import Image, Image.draw
from colletions import defaultdict, deque

PIXEL_GRID_JSON = "outputs/pixel_grid.json"
PALETTE_JSON = "outputs/palette.json"

STEPS_BY_COLOR = "outputs/steps_by_color.json"
STEPS_BY_REGION = "outputs/steps_by_region.json"
INSTRUCTIONS_PREVIEW = "outputs/instructions_preview.png"

GRID_PREVIEWS =  "outputs/grid_preview.png"


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
            "step" : step_id
            "type" : "color",
            "colorId" : color_id
            "cells" : cells
        })
        step_id += 1
    
    with open(STEPS_BY_COLOR, "w") as f:
        json.dump(steps, f, indent = 2)

