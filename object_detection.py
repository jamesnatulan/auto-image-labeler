import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os
import yaml
from tqdm import tqdm
import random

MODEL_ID = "IDEA-Research/grounding-dino-base"
IMAGES_DIR = "datasets/stanford-cars-dataset"
CONF_THRESHOLD = 0.25
OUTPUT_DIR = "output/stanford-cars-dataset"
SPLIT_SIZE = (0.8, 0.1, 0.1)
SEED = 42


def to_normalized_coordinates(box, image_size):
    x, y, w, h = box
    return x / image_size[0], y / image_size[1], w / image_size[0], h / image_size[1]


def to_pixel_coordinates(box, image_size):
    x, y, w, h = box
    return x * image_size[0], y * image_size[1], w * image_size[0], h * image_size[1]


def split_data(data, split_size, seed=42):
    random.seed(seed)
    random.shuffle(data)
    train_data = data[: int(len(data) * split_size[0])]
    val_data = data[
        int(len(data) * split_size[0]) : int(len(data) * (split_size[0] + split_size[1]))
    ]
    test_data = data[int(len(data) * (split_size[0] + split_size[1])) :]
    return train_data, val_data, test_data


def main():
    print("Hello from auto-label!")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor and model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

    # Load labels
    labels = ["sedan", "van", "motorcycle"]
    text = ". ".join(labels) + "."

    # Init output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "train/images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "val/images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "test/images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "train/labels"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "val/labels"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "test/labels"), exist_ok=True)

    # Generate yaml file
    data_yaml_file = os.path.join(OUTPUT_DIR, "classes.yaml")
    data_yaml = {
        "train": "../train/images",
        "val": "../val/images",
        "test": "../test/images",
        "nc": len(labels),
        "names": labels,
    }
    with open(data_yaml_file, "w") as f:
        yaml.dump(data_yaml, f)

    # Get all image paths first in a single list
    image_paths = []
    for root, _, files in os.walk(IMAGES_DIR):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                if len(image_paths) > 100:
                    break
                image_paths.append(os.path.join(root, file))

    # Temporary dir
    temp_dir = os.path.join(OUTPUT_DIR, "data.temp")
    os.makedirs(temp_dir, exist_ok=True)
    for image_path in tqdm(
        image_paths, desc="Processing images", total=len(image_paths)
    ):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]],
        )

        # Generate labels
        scores_labels_boxes = zip(
            results[0]["scores"], results[0]["labels"], results[0]["boxes"]
        )
        output_labels = []
        for score, label, box in scores_labels_boxes:
            if score > CONF_THRESHOLD:
                w = box[2] - box[0]
                h = box[3] - box[1]
                x = box[0] + w / 2
                y = box[1] + h / 2
                x, y, w, h = to_normalized_coordinates((x, y, w, h), image.size)

                # Cleanup label
                if len(label.split(" ")) > 1:
                    label = label.split(" ")[0]

                obj_class = labels.index(label)

                output_labels.append(f"{obj_class} {x} {y} {w} {h}")

        # Save image and label
        file_name = os.path.basename(image_path)
        image.save(os.path.join(temp_dir, file_name))
        with open(os.path.join(temp_dir, file_name.replace(".jpg", ".txt")), "w") as f:
            f.write("\n".join(output_labels))

    # Split data
    print("Splitting data")
    images = [file for file in os.listdir(temp_dir) if file.endswith(".jpg")]
    labels = [file for file in os.listdir(temp_dir) if file.endswith(".txt")]
    data = list(zip(images, labels))
    train_data, val_data, test_data = split_data(data, SPLIT_SIZE, SEED)

    for image, label in tqdm(
        train_data, desc="Moving train data", total=len(train_data)
    ):
        os.rename(
            os.path.join(temp_dir, image),
            os.path.join(OUTPUT_DIR, "train/images", image),
        )
        os.rename(
            os.path.join(temp_dir, label),
            os.path.join(OUTPUT_DIR, "train/labels", label),
        )

    for image, label in tqdm(val_data, desc="Moving val data", total=len(val_data)):
        os.rename(
            os.path.join(temp_dir, image),
            os.path.join(OUTPUT_DIR, "val/images", image),
        )
        os.rename(
            os.path.join(temp_dir, label),
            os.path.join(OUTPUT_DIR, "val/labels", label),
        )

    for image, label in tqdm(
        test_data, desc="Moving test data", total=len(test_data)
    ):
        os.rename(
            os.path.join(temp_dir, image),
            os.path.join(OUTPUT_DIR, "test/images", image),
        )
        os.rename(
            os.path.join(temp_dir, label),
            os.path.join(OUTPUT_DIR, "test/labels", label),
        )

    # Delete temp dir
    os.rmdir(temp_dir)


if __name__ == "__main__":
    main()
