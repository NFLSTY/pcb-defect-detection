import os
from pathlib import Path
import wandb

def upload_code(project: str = "pcb-defect-detection"):
    print("Authenticating with W&B...")
    wandb.login()

    # Initialize a W&B run for uploading the model core
    print("Initializing W&B run...")
    run = wandb.init(project=project, job_type="code-versioning")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create Artifacts
    print("Creating core model artifact...")
    core_artifact = wandb.Artifact(
        name="pcb-core-models",
        type="model",
        description="Core MobileViT-XXS + LeYOLO definitions",
        metadata={"architecture": "MobileViT-XXS-Core + LeYOLO-Core"}
    )

    core_dir = Path(script_dir).parent / "src" / "models"
    if core_dir.exists():
        for file_path in core_dir.glob("*.py"):
            # Add the contents directly to the artifact
            core_artifact.add_file(str(file_path), name=f"models/{file_path.name}")
    else:
        print(f"Warning: core directory not found at {core_dir}")

    # Upload it to the cloud W&B servers
    print("Uploading artifacts to W&B...")
    run.log_artifact(core_artifact)
    run.finish()
    print("Upload complete! Kaggle can now download these artifacts.")


if __name__ == "__main__":
    upload_code()