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
N_PREVIEWS = 5


class AutoLabelerClassification:
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
        image_path = self.image_paths[self.idx]
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            images=image, text=self.label_names, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=-1).cpu().numpy()
        scores = probs.tolist()

        result = [
            {"score": score, "label": candidate_label}
            for score, candidate_label in sorted(
                zip(scores, self.label_names), key=lambda x: -x[0]
            )
        ]
        top_result = result[0]
        label = top_result["label"]
        os.makedirs(os.path.join(self.data_dir, label), exist_ok=True)
        filename = os.path.basename(image_path)
        image_save_path = os.path.join(self.data_dir, label, filename)
        image.save(image_save_path)
        self.image_label_pairs.append((image_save_path, label))
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
                    "image_classification",
                )

        # Save the dataset
        if self.split_size:
            self.split()

    def split(self):
        # Split data
        train_data, val_data, test_data = split_data(
            self.image_label_pairs, self.split_size, self.seed
        )

        # Init splits directory
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "val"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "test"), exist_ok=True)
        for names in self.label_names:
            os.makedirs(os.path.join(self.output_path, "train", names), exist_ok=True)
            os.makedirs(os.path.join(self.output_path, "val", names), exist_ok=True)
            os.makedirs(os.path.join(self.output_path, "test", names), exist_ok=True)

        for image, label in tqdm(
            train_data, desc="Moving train data", total=len(train_data)
        ):
            os.rename(
                image,
                os.path.join(self.output_path, "train", label, os.path.basename(image)),
            )

        for image, label in tqdm(val_data, desc="Moving val data", total=len(val_data)):
            os.rename(
                image,
                os.path.join(self.output_path, "val", label, os.path.basename(image)),
            )

        for image, label in tqdm(
            test_data, desc="Moving test data", total=len(test_data)
        ):
            os.rename(
                image,
                os.path.join(self.output_path, "test", label, os.path.basename(image)),
            )

        # Delete mono dir
        shutil.rmtree(self.data_dir)



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

    # Init auto-labeler
    auto_labeler = AutoLabelerClassification(
        IMAGES_DIR, OUTPUT_DIR, model, processor, label_names, device, SEED, split_size=SPLIT_SIZE
    )

    # Run auto-labeler
    auto_labeler.autorun()


if __name__ == "__main__":
    main()
