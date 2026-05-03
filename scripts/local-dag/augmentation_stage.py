from promptflow.core import tool


@tool
def augmentation_stage(
    eda_context: dict,
    augmented_train_images: int = 2772,
    augmentation_script_path: str = "scripts/augment_data.py",
    augmented_root: str = "data/augmented/train",
) -> dict:
    """Capture augmentation pipeline decisions and outputs."""

    return {
        "stage": "augmentation",
        "depends_on": eda_context.get("stage"),
        "script": augmentation_script_path,
        "output_root": augmented_root,
        "output_images": augmented_train_images,
        "transform_family": [
            "Resize(480x480)",
            "Affine/Perspective",
            "Brightness/Contrast/Hue-Saturation",
            "GaussianBlur + GaussNoise",
            "BBox-safe transform with min_visibility=0.2",
        ],
        "status": "completed",
    }
