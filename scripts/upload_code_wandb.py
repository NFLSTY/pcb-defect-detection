import wandb
import os

def upload_code():
    print("Authenticating with W&B...")
    # This will prompt you to paste your W&B API key if you haven't logged in locally yet
    wandb.login()

    # Initialize a W&B run solely for uploading the dataset
    print("Initializing W&B run...")
    run = wandb.init(project="pcb-defect-detection", job_type="code-versioning")

    # Create an "Artifact" (a versioned bundle of files)
    print("Creating code artifact...")
    code_artifact = wandb.Artifact(
        name="pcb-model-code", 
        type="code", 
        description="V0 code containing the combination of MobileViT and LeYOLO architectures",
        metadata={"architecture": "MobileViT-LeYOLO-Hybrid"}
    )
    
    # Get the absolute path of the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Safely construct the paths to the data folders relative to this script
    code_path = os.path.normpath(os.path.join(script_dir, "..", "src", "model", "pcb_model.py"))
    
    if os.path.exists(code_path):
        print(f"Adding pcb_model.py from {code_path}...")
        # Add the contents directly to the artifact
        code_artifact.add_file(code_path)
    else:
        print("Warning: pcb_model.py not found!")

    # Upload it to the cloud W&B servers
    print("Uploading to W&B...")
    run.log_artifact(code_artifact)
    run.finish()
    print("Upload complete! Kaggle can now download this artifact.")

if __name__ == "__main__":
    upload_code()