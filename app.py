import streamlit as st
import os

import torch
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    AutoModelForZeroShotImageClassification,
)

from image_classification import AutoLabelerClassification
from object_detection import AutoLabelerObjectDetection
from common import data_preview


def start_labeling():
    st.session_state.start_labeling = True
    st.session_state.cancel_labeling = False


def cancel_labeling():
    st.session_state.start_labeling = False
    st.session_state.cancel_labeling = True


def empty_label_input():
    st.session_state.label_input = ""
    st.session_state.labels = []


def data_selector(folder_path="datasets"):
    folders = os.listdir(folder_path)
    selected_filename = st.sidebar.selectbox(
        "Select a folder",
        folders,
        help="Looks into your 'datasets' directory for image folders to use for labeling",
    )
    return os.path.join(folder_path, selected_filename)


@st.cache_resource
def load_model(task, model_id, device):
    if task == "object_detection":
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
    elif task == "image_classification":
        model = AutoModelForZeroShotImageClassification.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
    else:
        raise ValueError(f"Task {task} not supported")

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model.to(device)
    return processor, model, device


def main():
    # Start streamlit app
    st.title("Image Auto-Labeling")
    st.markdown(
        """Automatically label images for Image Classification or Object Detection tasks using zero-shot models.
        The app will automatically split the data into **train**, **validation**, and **test** sets based on the provided splits.
        Object Detection outputs the data in **YOLOv8 format**, while Image Classification follows the format that 
        **CIFAR-10** dataset uses(a directory for each label).
        """
    )

    st.divider()

    # Initialize session state vars
    if "start_labeling" not in st.session_state:
        st.session_state.start_labeling = False
    if "cancel_labeling" not in st.session_state:
        st.session_state.cancel_labeling = True

    # Sidebar
    # Initialize models
    st.sidebar.header("Model Configuration")
    task = st.sidebar.selectbox(
        "Task",
        ["image_classification", "object_detection"],
        help="Select the task to perform",
    )
    if task == "object_detection":
        model_id = st.sidebar.text_input(
            "Model ID",
            value="IDEA-Research/grounding-dino-base",
            help="The model to use for the task",
        )
    elif task == "image_classification":
        model_id = st.sidebar.text_input(
            "Model ID",
            value="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            help="The model to use for the task",
        )

    device_choice = st.sidebar.radio(
        "Device",
        ["auto", "cpu", "cuda", "mps"],
        help="The device to run the model on. Auto will select the best device available",
    )
    processor, model, device = load_model(task, model_id, device_choice)

    # Select dataset
    st.sidebar.header("Data Settings")
    dataset_dir = data_selector()

    # Output path
    output_dir = st.sidebar.text_input(
        "Output directory",
        value=f"{os.path.basename(dataset_dir)}-{task}",
        help="The directory where the labeled data will be saved",
    )

    # Splits
    seed = st.sidebar.number_input(
        "Seed", value=42, help="Random seed for reproducibility"
    )
    splits = st.sidebar.slider(
        "Splits",
        min_value=0.0,
        max_value=1.0,
        value=(0.8, 0.9),
        help="Number of splits to divide the dataset into",
    )
    train_split = round(splits[0], 2)
    val_split = round(splits[1] - splits[0], 2)
    test_split = round(1.0 - splits[1], 2)
    st.sidebar.text(f"Train: {train_split} | Val: {val_split} | Test: {test_split}")

    # Initialize labels
    st.sidebar.header("Labels")
    if "labels" not in st.session_state:
        st.session_state.labels = []

    # Add labels
    new_label = st.sidebar.text_input(
        "Add label: ",
        key="label_input",
        help="Add a new label. Separate multiple labels with a comma",
    )
    if len(new_label.split(",")) > 0:
        new_labels = new_label.split(",")
        new_labels = [label.strip() for label in new_labels]

    else:
        new_labels = [new_label.strip()]

    for label in new_labels:
        if label not in st.session_state.labels and label != "":
            st.session_state.labels.append(label)

    # Display labels
    st.sidebar.markdown(" | ".join(st.session_state.labels))

    # Clear labels
    st.sidebar.button("Clear labels", on_click=empty_label_input)

    # Start labeling
    st.sidebar.divider()
    left, right = st.sidebar.columns(2)
    left.button(
        "Start labeling!",
        disabled=not st.session_state.cancel_labeling,
        on_click=start_labeling,
        use_container_width=True,
    )
    right.button(
        "Cancel",
        disabled=not st.session_state.start_labeling,
        on_click=cancel_labeling,
        use_container_width=True,
    )

    if st.session_state.start_labeling:
        # Prepare previews
        st.header("Data Previews")
        st.text("The tabs will fill up with preview images as the labeling progresses")
        tabs = st.tabs([str(i) for i in range(1, 11, 1)])
        tab_idx = 0

        output_path = os.path.join("output", output_dir)
        if task == "object_detection":
            # Init auto-labeler
            auto_labeler_detect = AutoLabelerObjectDetection(
                dataset_dir,
                output_path,
                model,
                processor,
                st.session_state.labels,
                device,
                (train_split, val_split, test_split),
                seed,
            )

            # Run auto-labeler
            progress_bar = st.progress(0.0, text="Processing images")
            for i in range(len(auto_labeler_detect.image_paths)):
                auto_labeler_detect.forward()

                if i % 100 == 0 and i > 0:
                    data_preview(
                        auto_labeler_detect.image_label_pairs[i - 10 : i],
                        auto_labeler_detect.label_names,
                        f"{auto_labeler_detect.output_path}/preview_{i}.png",
                        "object_detection",
                    )
                    if tab_idx < 10:
                        tabs[tab_idx].image(
                            f"{auto_labeler_detect.output_path}/preview_{i}.png"
                        )
                        tab_idx += 1

                if st.session_state.cancel_labeling:
                    break

                progress_bar.progress(
                    i / len(auto_labeler_detect.image_paths), text="Processing images"
                )
            st.success("Labeling complete")

        elif task == "image_classification":
            # Init auto-labeler
            auto_labeler_classify = AutoLabelerClassification(
                dataset_dir,
                output_path,
                model,
                processor,
                st.session_state.labels,
                device,
                (train_split, val_split, test_split),
                seed,
            )

            # Run auto-labeler
            progress_bar = st.progress(0.0, text="Processing images")
            for i in range(len(auto_labeler_classify.image_paths)):
                auto_labeler_classify.forward()

                if i % 100 == 0 and i > 0:
                    data_preview(
                        auto_labeler_classify.image_label_pairs[i - 10 : i],
                        auto_labeler_classify.label_names,
                        f"{auto_labeler_classify.output_path}/preview_{i}.png",
                        "image_classification",
                    )
                    if tab_idx < 10:
                        tabs[tab_idx].image(
                            f"{auto_labeler_classify.output_path}/preview_{i}.png"
                        )
                        tab_idx += 1

                if st.session_state.cancel_labeling:
                    break

                progress_bar.progress(
                    i / len(auto_labeler_classify.image_paths), text="Processing images"
                )
            st.success("Labeling complete")


if __name__ == "__main__":
    main()
