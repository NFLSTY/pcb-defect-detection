from promptflow.core import tool


@tool
def pipeline_summary(
    source_context: dict,
    eda_context: dict,
    augmentation_context: dict,
    tracking_context: dict,
) -> str:
    """Generate a readable process summary from all DAG stages."""

    return "\n".join(
        [
            "PCB Defect Pipeline DAG Summary",
            "- 1) Data Source: Kaggle raw dataset is used directly for train/valid/test.",
            f"  • Dataset slug: {source_context.get('dataset_slug')}",
            "- 2) EDA: Completed notebook-driven checks on brightness, sharpness, and spatial distribution.",
            f"  • Raw brightness mean: {eda_context.get('metrics', {}).get('brightness_mean_raw')}",
            f"  • Raw sharpness mean: {eda_context.get('metrics', {}).get('sharpness_mean_raw')}",
            "- 3) Augmentation: Completed CPU-multiprocess augmentation with synchronized YOLO bboxes.",
            f"  • Augmented train images: {augmentation_context.get('output_images')}",
            f"  • Script: {augmentation_context.get('script')}",
            "- 4) Tracking: W&B artifact stores augmented dataset and enables lineage graph.",
            f"  • Project: {tracking_context.get('wandb_project')}",
            f"  • Artifact: {tracking_context.get('artifact')}",
            "- Next: Start Kaggle training run and log model artifact to complete end-to-end DAG provenance.",
        ]
    )
