import albumentations as A
import cv2
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

# Define directories based on standard YOLO structure
RAW_IMG_DIR = "../data/raw/train/images/"
RAW_LBL_DIR = "../data/raw/train/labels/"
AUG_IMG_DIR = "../data/augmented/train/images/"
AUG_LBL_DIR = "../data/augmented/train/labels/"

# Ensure output directories exist
os.makedirs(AUG_IMG_DIR, exist_ok=True)
os.makedirs(AUG_LBL_DIR, exist_ok=True)

# The Fine-Tuned Pipeline with CLAHE
transform = A.Compose([
    A.RandomResizedCrop(size=(640, 640), scale=(0.5, 1.0), p=0.5),
    A.Resize(height=640, width=640, p=0.5),
    
    # Geometric Transformations
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=[-15, 15], p=0.4, border_mode=cv2.BORDER_REFLECT_101),
    A.Perspective(scale=(0.02, 0.05), p=0.2), 
    
    # --- THE LOW-LIGHT FIX ---
    # CLAHE dynamically equalizes the histogram in local patches, rescuing shadow details
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    
    # Pixel/Color Transformations
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.OneOf([
        A.MotionBlur(p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
    ], p=0.2),
    
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

def process_single_image(img_path):
    filename = os.path.basename(img_path)
    name_only = os.path.splitext(filename)[0]
    lbl_path = os.path.join(RAW_LBL_DIR, f"{name_only}.txt")
    
    # Skip if image doesn't have a matching label file
    if not os.path.exists(lbl_path):
        return False, filename, "Missing label file"
        
    try:
        # 1. Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Load YOLO Bounding Boxes
        bboxes = []
        class_labels = []
        with open(lbl_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    # YOLO format: [x_center, y_center, width, height]
                    bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                    bboxes.append(bbox)
                    class_labels.append(class_id)
        
        # 3. Apply Mathematical Transformations to Image AND Boxes
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_labels = transformed['class_labels']
        
        # 4. Save Augmented Image
        out_img_path = os.path.join(AUG_IMG_DIR, f"aug_{filename}")
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_img_path, transformed_image)
        
        # 5. Save Augmented YOLO Labels
        out_lbl_path = os.path.join(AUG_LBL_DIR, f"aug_{name_only}.txt")
        with open(out_lbl_path, 'w') as f:
            for bbox, cls_lbl in zip(transformed_bboxes, transformed_labels):
                # Write back in YOLO format
                f.write(f"{cls_lbl} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        
        return True, filename, "Success"
        
    except Exception as e:
        return False, filename, str(e)

def process_images():
    image_paths = glob.glob(os.path.join(RAW_IMG_DIR, "*.jpg"))
    success_count = 0
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_image, path): path for path in image_paths}
        
        for future in as_completed(futures):
            success, filename, msg = future.result()
            if success:
                success_count += 1
            else:
                if msg != "Missing label file":
                    print(f"Error processing {filename}: {msg}")
            
    print(f"Augmentation complete! Successfully processed {success_count} image-label pairs.")

if __name__ == "__main__":
    process_images()