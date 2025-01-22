import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import os
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
IMAGES_DIR = "datasets/stanford-cars-dataset"
OUTPUT_DIR = "output/stanford-cars-dataset"
SPLIT_SIZE = (0.8, 0.1, 0.1)
SEED = 42


def split_data(data, split_size, seed=42):
    random.seed(seed)
    random.shuffle(data)
    train_data = data[: int(len(data) * split_size[0])]
    val_data = data[
        int(len(data) * split_size[0]) : int(len(data) * (split_size[0] + split_size[1]))
    ]
    test_data = data[int(len(data) * (split_size[0] + split_size[1])) :]
    return train_data, val_data, test_data


def data_preview(data, label_names):
    # Get only the first 9 images for a 3x3 grid
    if len(data) < 9:
        data = data[:len(data)]
    else:
        data = data[:9]

    # Create a 3x3 grid
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i, (image, label) in enumerate(data):
        ax = axs[i // 3, i % 3]
        ax.imshow(Image.open(image))
        ax.axis("off")
        ax.set_title(label, fontsize=12, color="r")
    
    plt.savefig(os.path.join(OUTPUT_DIR, "data_preview.jpg"))


def main():
    print("Hello from auto-label!")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor and model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotImageClassification.from_pretrained(MODEL_ID).to(device)

    # Load labels
    label_names = ["black sedan", "white sedan", "red sedan", "blue sedan", "orange sedan"]

    # Init output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "val"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)
    for names in label_names:
        os.makedirs(os.path.join(OUTPUT_DIR, "train", names), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "val", names), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "test", names), exist_ok=True)


    # Get all image paths first in a single list
    image_paths = []
    for root, _, files in os.walk(IMAGES_DIR):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                if len(image_paths) > 100:
                    break
                image_paths.append(os.path.join(root, file))

    # Create a temp dir
    temp_dir = "data.temp"
    os.makedirs(temp_dir, exist_ok=True)

    for image_path in tqdm(
        image_paths, desc="Processing images", total=len(image_paths)
    ):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=label_names, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=-1).cpu().numpy()
        scores = probs.tolist()

        result = [
            {"score": score, "label": candidate_label}
            for score, candidate_label in sorted(zip(scores, label_names), key=lambda x: -x[0])
        ]
        top_result = result[0]
        label = top_result["label"]
        filename = f"{label}.{os.path.basename(image_path)}"
        image.save(os.path.join(temp_dir, filename))

    # Split data
    print("Splitting data")
    images = [os.path.join(temp_dir, file) for file in os.listdir(temp_dir)]
    labels = [file.split(".")[0] for file in os.listdir(temp_dir)]
    data = list(zip(images, labels))

    data_preview(data, label_names)
    train_data, val_data, test_data = split_data(data, SPLIT_SIZE, SEED)

    for image, label in tqdm(
        train_data, desc="Moving train data", total=len(train_data)
    ):  
        new_name = ".".join(os.path.basename(image).split(".")[1:])
        os.rename(
            image,
            os.path.join(OUTPUT_DIR, "train", label, new_name),
        )

    for image, label in tqdm(val_data, desc="Moving val data", total=len(val_data)):
        new_name = ".".join(os.path.basename(image).split(".")[1:])
        os.rename(
            image,
            os.path.join(OUTPUT_DIR, "val", label, new_name),
        )

    for image, label in tqdm(
        test_data, desc="Moving test data", total=len(test_data)
    ):
        new_name = ".".join(os.path.basename(image).split(".")[1:])
        os.rename(
            image,
            os.path.join(OUTPUT_DIR, "test", label, new_name),
        )

    # Delete temp dir
    os.rmdir(temp_dir)


if __name__ == "__main__":
    main()
