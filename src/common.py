import os
import random
from PIL import Image
from matplotlib import pyplot as plt

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
        int(len(data) * split_size[0]) : int(
            len(data) * (split_size[0] + split_size[1])
        )
    ]
    test_data = data[int(len(data) * (split_size[0] + split_size[1])) :]
    return train_data, val_data, test_data


def data_preview(data, label_names, out, task):
    # Get only the first 9 images for a 3x3 grid
    if len(data) < 9:
        data = data[: len(data)]
    else:
        data = data[:9]

    # Create a 3x3 grid
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i, (image, label) in enumerate(data):
        ax = axs[i // 3, i % 3]
        ax.imshow(Image.open(image))
        ax.axis("off")

        if task == "object_detection":
            with open(label, "r") as f:
                labels = f.readlines()

            for label in labels:
                obj_class, x, y, w, h = label.split()
                x, y, w, h = to_pixel_coordinates(
                    (float(x), float(y), float(w), float(h)), Image.open(image).size
                )
                ax.add_patch(
                    plt.Rectangle(
                        (x - w / 2, y - h / 2),
                        w,
                        h,
                        linewidth=2,
                        edgecolor="r",
                        facecolor="none",
                    )
                )
                ax.text(
                    x - w / 2,
                    y - h / 2,
                    label_names[int(obj_class)],
                    fontsize=12,
                    color="r",
                )
        
        elif task == "image_classification":
            ax.set_title(label, fontsize=12, color="r")

    plt.savefig(out)


def get_image_paths(input_dir):
    # Get all image paths first in a single list
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                if len(image_paths) > 100:
                    break
                image_paths.append(os.path.join(root, file))

    return image_paths
