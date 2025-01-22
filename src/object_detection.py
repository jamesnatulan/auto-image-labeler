import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os
import shutil
import yaml
from tqdm import tqdm

from common import (
    split_data,
    get_image_paths,
    data_preview,
    to_normalized_coordinates,
)

MODEL_ID = "IDEA-Research/grounding-dino-base"
IMAGES_DIR = "datasets/stanford-cars-dataset"
CONF_THRESHOLD = 0.6
OUTPUT_DIR = "output/stanford-cars-dataset-object-detection"
SPLIT_SIZE = (0.8, 0.1, 0.1)
SEED = 42


def object_detection_forward(image_path, processor, model, label_names, device):
    text = ". ".join(label_names) + "."
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

            obj_class = label_names.index(label)

            output_labels.append(f"{obj_class} {x} {y} {w} {h}")

    return image, output_labels


def auto_labeler_detection(
    image_paths, output_path, model, processor, label_names, device, split_size, seed
):
    # Init output directory
    if os.path.exists(output_path):
        shutil.rmtree(output_path)  # Remove existing output directory

    os.makedirs(output_path)
    os.makedirs(os.path.join(output_path, "train/images"))
    os.makedirs(os.path.join(output_path, "val/images"))
    os.makedirs(os.path.join(output_path, "test/images"))
    os.makedirs(os.path.join(output_path, "train/labels"))
    os.makedirs(os.path.join(output_path, "val/labels"))
    os.makedirs(os.path.join(output_path, "test/labels"))

    # Generate yaml file
    data_yaml_file = os.path.join(output_path, "classes.yaml")
    data_yaml = {
        "train": "../train/images",
        "val": "../val/images",
        "test": "../test/images",
        "nc": len(label_names),
        "names": label_names,
    }
    with open(data_yaml_file, "w") as f:
        yaml.dump(data_yaml, f)

    # Temporary dir
    temp_dir = os.path.join(output_path, "data.temp")
    os.makedirs(temp_dir, exist_ok=True)

    image_label_pairs = []
    for image_path in tqdm(
        image_paths, desc="Processing images", total=len(image_paths)
    ):
        image, output_labels = object_detection_forward(
            image_path, processor, model, label_names, device
        )

        # Save image and label
        file_name = os.path.basename(image_path)
        save_image_path = os.path.join(temp_dir, file_name)
        image.save(save_image_path)
        save_label_path = os.path.join(temp_dir, file_name.replace(".jpg", ".txt"))
        with open(save_label_path, "w") as f:
            f.write("\n".join(output_labels))
        image_label_pairs.append((save_image_path, save_label_path))

    # Split data
    print("Splitting data")
    data_preview(image_label_pairs, label_names, output_path, "object_detection")
    train_data, val_data, test_data = split_data(image_label_pairs, split_size, seed)

    for image, label in tqdm(
        train_data, desc="Moving train data", total=len(train_data)
    ):
        os.rename(
            image,
            os.path.join(output_path, "train/images", os.path.basename(image)),
        )
        os.rename(
            label,
            os.path.join(output_path, "train/labels", os.path.basename(label)),
        )

    for image, label in tqdm(val_data, desc="Moving val data", total=len(val_data)):
        os.rename(
            image,
            os.path.join(output_path, "val/images", os.path.basename(image)),
        )
        os.rename(
            label,
            os.path.join(output_path, "val/labels", os.path.basename(label)),
        )

    for image, label in tqdm(test_data, desc="Moving test data", total=len(test_data)):
        os.rename(
            image,
            os.path.join(output_path, "test/images", os.path.basename(image)),
        )
        os.rename(
            label,
            os.path.join(output_path, "test/labels", os.path.basename(label)),
        )

    # Delete temp dir
    os.rmdir(temp_dir)


def main():
    print("Hello from auto-label!")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor and model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

    # Load labels
    label_names = ["sedan", "van", "motorcycle"]

    # Get image paths
    image_paths = get_image_paths(IMAGES_DIR)

    # Run auto-labeler
    auto_labeler_detection(
        image_paths, OUTPUT_DIR, model, processor, label_names, device, SPLIT_SIZE, SEED
    )


if __name__ == "__main__":
    main()
