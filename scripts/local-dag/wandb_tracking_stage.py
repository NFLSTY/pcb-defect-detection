from promptflow.core import tool


@tool
def wandb_tracking_stage(
    augmentation_context: dict,
    wandb_project: str = "pcb-defect-detection",
    dataset_artifact_name: str = "pcb-augmented-dataset",
    upload_script_path: str = "scripts/upload_dataset_wandb.py",
) -> dict:
    """Describe dataset tracking via W&B artifact lineage."""

    return {
        "stage": "tracking",
        "depends_on": augmentation_context.get("stage"),
        "wandb_project": wandb_project,
        "artifact": f"{dataset_artifact_name}:latest",
        "uploader": upload_script_path,
        "lineage_next": [
            "Kaggle training run consumes dataset artifact",
            "Training run logs model artifact + metrics",
            "W&B renders provenance DAG automatically",
        ],
        "status": "ready",
    }
