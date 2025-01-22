import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import os
from tqdm import tqdm
import shutil

from common import get_image_paths, split_data, data_preview

MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
IMAGES_DIR = "datasets/stanford-cars-dataset"
OUTPUT_DIR = "output/stanford-cars-dataset-image-classification"
SPLIT_SIZE = (0.8, 0.1, 0.1)
SEED = 42


def classification_forward(image_path, processor, model, label_names, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=label_names, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_image[0]
    probs = logits.softmax(dim=-1).cpu().numpy()
    scores = probs.tolist()

    result = [
        {"score": score, "label": candidate_label}
        for score, candidate_label in sorted(
            zip(scores, label_names), key=lambda x: -x[0]
        )
    ]
    top_result = result[0]
    return image, top_result["label"]


def auto_labeler_classification(
    image_paths, output_path, model, processor, label_names, device, split_size, seed
):
    # Init output directory
    if os.path.exists(output_path):
        shutil.rmtree(output_path)  # Remove existing output directory

    # Init output directory
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "test"), exist_ok=True)
    for names in label_names:
        os.makedirs(os.path.join(output_path, "train", names), exist_ok=True)
        os.makedirs(os.path.join(output_path, "val", names), exist_ok=True)
        os.makedirs(os.path.join(output_path, "test", names), exist_ok=True)

    # Create a temp dir
    temp_dir = os.path.join(output_path, "data.temp")
    os.makedirs(temp_dir, exist_ok=True)

    image_label_pairs = []
    for image_path in tqdm(
        image_paths, desc="Processing images", total=len(image_paths)
    ):
        image, label = classification_forward(
            image_path, processor, model, label_names, device
        )
        filename = os.path.basename(image_path)
        image_save_path = os.path.join(temp_dir, filename)
        image_label_pairs.append((image_save_path, label))
        image.save(image_save_path)

    # Split data
    train_data, val_data, test_data = split_data(image_label_pairs, split_size, seed)
    data_preview(train_data, None, output_path, "image_classification")
    for image, label in tqdm(
        train_data, desc="Moving train data", total=len(train_data)
    ):
        os.rename(
            image,
            os.path.join(output_path, "train", label, os.path.basename(image)),
        )

    for image, label in tqdm(val_data, desc="Moving val data", total=len(val_data)):
        os.rename(
            image,
            os.path.join(output_path, "val", label, os.path.basename(image)),
        )

    for image, label in tqdm(test_data, desc="Moving test data", total=len(test_data)):
        os.rename(
            image,
            os.path.join(output_path, "test", label, os.path.basename(image)),
        )

    # Delete temp dir
    os.rmdir(temp_dir)


def main():
    print("Hello from auto-label!")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor and model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotImageClassification.from_pretrained(MODEL_ID).to(device)

    # Load labels
    label_names = [
        "black sedan",
        "white sedan",
        "red sedan",
        "blue sedan",
        "orange sedan",
    ]

    # Get image paths
    image_paths = get_image_paths(IMAGES_DIR)

    # Run auto-labeler
    auto_labeler_classification(
        image_paths, OUTPUT_DIR, model, processor, label_names, device, SPLIT_SIZE, SEED
    )


if __name__ == "__main__":
    main()
