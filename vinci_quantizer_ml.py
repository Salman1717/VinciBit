# vinci_quantizer_ml.py
# Builds a small Core ML model spec (identity) using NeuralNetworkBuilder
# Uses add_scale to implement identity passthrough (works across coremltools versions)

import coremltools as ct
from coremltools.models import neural_network as nn
from coremltools.models.datatypes import Array
import numpy as np
import json
import os

MODEL_NAME = "VinciBitQuantizer"
PALETTE_JSON = "outputs/palette.json"


def load_palette():
    # Palette is kept for Swift-side logic
    if os.path.exists(PALETTE_JSON):
        with open(PALETTE_JSON) as f:
            return json.load(f)
    return None


def build_model_spec():
    # NeuralNetworkBuilder requires (name, Array) tuples
    input_features = [("input_image", Array(3, 256, 256))]
    output_features = [("output_image", Array(3, 256, 256))]

    builder = nn.NeuralNetworkBuilder(
        input_features,
        output_features,
        disable_rank5_shape_mapping=True
    )

    # Identity via a scale layer: out = W * in + b
    # Use W = 1.0 and b = 0.0 to create identity mapping.
    # W and b must be numpy arrays
    builder.add_scale(
        name="identity_scale",
        W=np.array([1.0], dtype=np.float32),        # scale (broadcast)
        b=np.array([0.0], dtype=np.float32),        # bias
        has_bias=True,
        input_name="input_image",
        output_name="output_image"
    )

    return builder.spec


def main():
    _ = load_palette()  # kept on disk for Swift usage
    spec = build_model_spec()

    model = ct.models.MLModel(spec)
    model.save(f"{MODEL_NAME}.mlmodel")

    print(f"[VinciBit] Saved {MODEL_NAME}.mlmodel")


if __name__ == "__main__":
    main()