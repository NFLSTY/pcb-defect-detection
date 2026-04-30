# PCB Defect Detection

This project focuses on developing a machine learning model to detect defects in printed circuit boards (PCBs). The dataset is sourced from Kaggle and includes images of PCBs along with corresponding labels indicating the presence of defects. The project involves data preprocessing, exploratory data analysis (EDA), data augmentation, model training, and evaluation. The model architecture is based on MobileViT-XXS, and the training process is tracked using Weights & Biases (W&B) for experiment management and visualization.

## Project Structure

```bash
pcb-defect-detection/
├── .dvc/                      
├── .git/		               
├── data/
│   ├── raw/ 		          
│   │	├── train/ 
│   │   │	├── images/
│   │   │	└── labels/	     
│   │	├── test/
│   │   │	├── images/
│   │   │	└── labels/	     
│   │	└── valid/          
│   │   │	├── images/
│   │   │	└── labels/	     
│   ├── augmented/ 	            
│   │	├── train/	     
│   │	├── test/
│   │	└── valid/        
├── notebooks/
│   └── kaggle_training.ipynb    
├── src/
│   ├── model/
│   │   ├── mobilevit_xxs.py    
│   │   └── leyolo_head.py      
│   └── utils/
│	    └── wandb_logger.py    
└── scripts/
    ├── augment_data.py
    ├── eda.py       
    └── export_tflite.py     
```