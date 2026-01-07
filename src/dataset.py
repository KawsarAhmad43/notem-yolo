import os
import shutil
from pathlib import Path
import random
import cv2  # For image dimensions

DATA_ROOT = Path("data")
YOLO_ROOT = DATA_ROOT / "yolo_dataset"
CLASSES = ["1", "2", "5", "10", "20", "50", "100", "500", "1000"]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASSES)}

def create_yolo_structure():
    for split in ["train", "val", "test"]:
        (YOLO_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

def get_image_dimensions(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    height, width = img.shape[:2]
    return width, height

def convert_to_yolo():
    create_yolo_structure()
    
    all_images = []
    for cls_name in CLASSES:
        class_dir = DATA_ROOT / "training" / cls_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist")
            continue
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
        all_images.extend([(img, cls_name) for img in images])

    # Shuffle and split: 80% train, 20% val (modify as needed)
    random.shuffle(all_images)
    train_size = int(0.8 * len(all_images))
    train_images = all_images[:train_size]
    val_images = all_images[train_size:]

    def process_split(image_list, split_name):
        for img_path, cls_name in image_list:
            class_id = CLASS_MAP[cls_name]
            
            # Copy image
            new_img_name = f"{cls_name}_{img_path.stem}{img_path.suffix}"
            dest_img = YOLO_ROOT / "images" / split_name / new_img_name
            shutil.copy(img_path, dest_img)

            # Create label (dummy: full image bbox, center 0.5,0.5 width/height ~0.9 to avoid edges)
            w, h = get_image_dimensions(img_path)
            label_path = YOLO_ROOT / "labels" / split_name / (dest_img.stem + ".txt")
            with open(label_path, "w") as f:
                x_center = 0.5
                y_center = 0.5
                width = 0.9  # Normalized
                height = 0.9
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    process_split(train_images, "train")
    process_split(val_images, "val")

    # Copy testing images to yolo test (no labels needed for inference)
    test_dir = DATA_ROOT / "testing"
    for img_path in test_dir.glob("*"):
        shutil.copy(img_path, YOLO_ROOT / "images" / "test" / img_path.name)

    # Handle takas.png as test
    takas_path = DATA_ROOT / "training" / "takas.png"
    if takas_path.exists():
        shutil.copy(takas_path, YOLO_ROOT / "images" / "test" / "takas.png")

    print(f"Dataset ready! Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(list((YOLO_ROOT/'images/test').glob('*')))}")

if __name__ == "__main__":
    convert_to_yolo()