from promptflow.core import tool


@tool
def kaggle_data_source(
    kaggle_dataset_slug: str,
    use_kaggle_raw_directly: bool = True,
    kaggle_raw_root: str = "/kaggle/input/pcb-defect-dataset",
) -> dict:
    """Describe how raw data is sourced from Kaggle.

    This node does not download data; it documents and normalizes source metadata
    for downstream steps in the DAG.
    """

    strategy = "kaggle-native" if use_kaggle_raw_directly else "external-upload"

    return {
        "stage": "data_source",
        "strategy": strategy,
        "dataset_slug": kaggle_dataset_slug,
        "raw_root": kaggle_raw_root,
        "splits": {
            "train": f"{kaggle_raw_root}/train",
            "valid": f"{kaggle_raw_root}/valid",
            "test": f"{kaggle_raw_root}/test",
        },
        "notes": [
            "Use Kaggle raw dataset directly to avoid redundant uploads.",
            "Keep validation/test unaugmented for fair evaluation.",
        ],
    }
