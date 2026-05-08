import os
import wandb

def upload_dataset():
    print("Authenticating with W&B...")
    wandb.login()

    # Initialize a W&B run for uploading the dataset
    print("Initializing W&B run...")
    run = wandb.init(project="pcb-defect-detection", job_type="dataset-creation")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create Artifacts
    print("Creating dataset artifact...")
    dataset_artifact = wandb.Artifact(
        name="pcb-augmented-dataset", 
        type="dataset", 
        description="Dataset of augmented training data",
        metadata={"augmentation": "albumentations"}
    )
    
    # Safely construct the paths to the data folders relative to this script
    augmented_path = os.path.normpath(os.path.join(script_dir, "..", "data", "augmented"))
    
    if os.path.exists(augmented_path):
        print(f"Adding augmented data from {augmented_path}...")
        # Add the contents directly to the artifact
        dataset_artifact.add_dir(augmented_path)
    else:
        print("Warning: Augmented data folder not found!")

    # Upload it to the cloud W&B servers
    print("Uploading to W&B...")
    run.log_artifact(dataset_artifact)
    run.finish()
    print("Upload complete! Kaggle can now download this artifact.")

if __name__ == "__main__":
    upload_dataset()