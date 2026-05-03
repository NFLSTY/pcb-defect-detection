# Local Prompt Flow DAG for PCB Pipeline

This folder contains a Prompt Flow DAG that documents the current ML pipeline progress:

1. Kaggle raw data sourcing
2. EDA completion
3. Augmentation completion
4. W&B dataset tracking readiness

## Files

- `flow.dag.yaml`: DAG definition used by Prompt Flow visual editor.
- `kaggle_data_source.py`: Data source node.
- `eda_stage.py`: EDA node.
- `augmentation_stage.py`: Augmentation node.
- `wandb_tracking_stage.py`: W&B tracking node.
- `pipeline_summary.py`: Final summary node.

## Notes

- This DAG is a process-documentation flow (metadata), not a training flow.
- You can update node defaults as your pipeline evolves (e.g., artifact version, metric values).
