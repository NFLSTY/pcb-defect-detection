from promptflow.core import tool


@tool
def eda_stage(
    source_context: dict,
    brightness_mean_raw: float = 60.8,
    sharpness_mean_raw: float = 1941.0,
    eda_notebook_path: str = "scripts/eda.ipynb",
) -> dict:
    """Summarize EDA outputs and quality signals."""

    return {
        "stage": "eda",
        "source_strategy": source_context.get("strategy"),
        "notebook": eda_notebook_path,
        "metrics": {
            "brightness_mean_raw": brightness_mean_raw,
            "sharpness_mean_raw": sharpness_mean_raw,
        },
        "visual_checks": [
            "Raw brightness/sharpness histograms",
            "Raw vs augmented KDE comparisons",
            "Class-wise 2D spatial histogram (defect location distribution)",
            "Single-image before/after augmentation with YOLO boxes",
        ],
        "status": "completed",
    }
