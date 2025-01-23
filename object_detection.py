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
N_PREVIEWS = 5


class AutoLabelerObjectDetection:
    def __init__(
        self,
        input_dir,
        output_path,
        model,
        processor,
        label_names,
        device,
        seed,
        split_size=None,
    ):
        self.image_paths = get_image_paths(input_dir)
        self.output_path = output_path
        self.model = model
        self.processor = processor
        self.label_names = label_names
        self.device = device
        self.seed = seed
        self.split_size = split_size
        self.image_label_pairs = []
        self.idx = 0

        # Init output directory
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)  # Remove existing output directory

        # Monodirectory too hold all images with their label
        self.data_dir = os.path.join(self.output_path, "data")
        os.makedirs(self.data_dir, exist_ok=True)

    def forward(self):
        """
        Single pass through the images for progress bar access
        """
        text = ". ".join(self.label_names) + "."
        image_path = self.image_paths[self.idx]
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
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

                obj_class = self.label_names.index(label)

                output_labels.append(f"{obj_class} {x} {y} {w} {h}")
        
        # Save image and label
        file_name = os.path.basename(image_path)
        save_image_path = os.path.join(self.data_dir, file_name)
        image.save(save_image_path)
        save_label_path = os.path.join(self.data_dir, file_name.replace(".jpg", ".txt"))
        with open(save_label_path, "w") as f:
            f.write("\n".join(output_labels))
        self.image_label_pairs.append((save_image_path, save_label_path))
        self.idx += 1


    def autorun(self):
        """
        Run the auto-labeler
        """
        for i in tqdm(range(len(self.image_paths)), desc="Processing images"):
            self.forward()

            if i % 100 == 0 and i > 0:
                data_preview(
                    self.image_label_pairs[i - 10 : i],
                    self.label_names,
                    f"{self.output_path}/preview_{i}.png",
                    "object_detection",
                )

        # Save the dataset
        if self.split_size:
            self.split()
            # Generate yaml file
            data_yaml_file = os.path.join(self.output_path, "classes.yaml")
            data_yaml = {
                "train": "../train/images",
                "val": "../val/images",
                "test": "../test/images",
                "nc": len(self.label_names),
                "names": self.label_names,
            }
            with open(data_yaml_file, "w") as f:
                yaml.dump(data_yaml, f)

        else:
            # Generate yaml file
            data_yaml_file = os.path.join(self.output_path, "classes.yaml")
            data_yaml = {
                "images": "../data/images",
                "nc": len(self.label_names),
                "names": self.label_names,
            }
            with open(data_yaml_file, "w") as f:
                yaml.dump(data_yaml, f)
            

    def split(self):
        # Split data
        train_data, val_data, test_data = split_data(
            self.image_label_pairs, self.split_size, self.seed
        )

        # Init splits directory
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "train/images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "val/images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "test/images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "train/labels"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "val/labels"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "test/labels"), exist_ok=True)

        for image, label in tqdm(
            train_data, desc="Moving train data", total=len(train_data)
        ):
            os.rename(
                image,
                os.path.join(self.output_path, "train/images", os.path.basename(image)),
            )
            os.rename(
                label,
                os.path.join(self.output_path, "train/labels", os.path.basename(label)),
            )

        for image, label in tqdm(val_data, desc="Moving val data", total=len(val_data)):
            os.rename(
                image,
                os.path.join(self.output_path, "val/images", os.path.basename(image)),
            )
            os.rename(
                label,
                os.path.join(self.output_path, "val/labels", os.path.basename(label)),
            )

        for image, label in tqdm(test_data, desc="Moving test data", total=len(test_data)):
            os.rename(
                image,
                os.path.join(self.output_path, "test/images", os.path.basename(image)),
            )
            os.rename(
                label,
                os.path.join(self.output_path, "test/labels", os.path.basename(label)),
            )

        # Delete mono dir
        shutil.rmtree(self.data_dir)


def main():
    print("Hello from auto-label!")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor and model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

    # Load labels
    label_names = ["sedan", "van", "motorcycle"]

    # Init auto-labeler
    auto_labeler = AutoLabelerObjectDetection(
        IMAGES_DIR, OUTPUT_DIR, model, processor, label_names, device, SEED, SPLIT_SIZE
    )

    # Run auto-labeler
    auto_labeler.autorun()


if __name__ == "__main__":
    main()
