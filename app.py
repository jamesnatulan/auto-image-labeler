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

def empty_label_input():
    st.session_state.label_input = ""
    st.session_state.labels = []

def data_selector(folder_path="datasets"):
    folders = os.listdir(folder_path)
    selected_filename = st.selectbox(
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
    
    # Sidebar
    st.sidebar.title("Settings")
    seed = st.sidebar.number_input("Seed", value=42, help="Random seed for reproducibility")

    # Initialize models
    task = st.sidebar.selectbox(
        "Task",
        ["object_detection", "image_classification"],
        index=1,
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
        "Device", ["auto", "cpu", "cuda", "mps"], help="The device to run the model on"
    )
    processor, model, device = load_model(task, model_id, device_choice)

    # Select dataset
    st.header("Images")
    dataset_dir = data_selector()

    # Output path
    output_dir = st.text_input(
        "Output directory",
        value=f"{os.path.basename(dataset_dir)}-{task}",
        help="The directory where the labeled data will be saved",
    )

    # Splits
    splits = st.slider(
        "Splits",
        min_value=0.0,
        max_value=1.0,
        value=(0.8, 0.9),
        help="Number of splits to divide the dataset into",
    )
    train_split = round(splits[0], 2)
    val_split = round(splits[1] - splits[0], 2)
    test_split = round(1.0 - splits[1], 2)

    st.text(
        f"Train: {train_split} | Val: {val_split} | Test: {test_split}"
    )

    
    # Initialize labels
    st.header("Labels")
    if "labels" not in st.session_state:
        st.session_state.labels = []
    
    # Add labels
    new_label = st.text_input("Add label: ", key="label_input")
    if len(new_label.split(",")) > 0:
        new_labels = new_label.split(",")
        new_labels = [label.strip() for label in new_labels]
    
    else:
        new_labels = [new_label.strip()]
    
    for label in new_labels:
        if label not in st.session_state.labels and label != "":
            st.session_state.labels.append(label)

    # Display labels
    with st.container():
        st.write(" | ".join(st.session_state.labels))

    # Clear labels
    st.button("Clear labels", on_click=empty_label_input)

    # Start labeling
    if st.button("Start labeling"):
        output_path = os.path.join("output", output_dir)
        if task=="object_detection":
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

            # Prepare previews
            tabs = st.tabs([str(i) for i in range(1, 11, 1)])
            tab_idx = 0

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
                        tabs[tab_idx].image(f"{auto_labeler_detect.output_path}/preview_{i}.png")
                        tab_idx += 1

            st.success("Labeling complete")

        elif task=="image_classification":
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

            # Prepare previews
            tabs = st.tabs([str(i) for i in range(1, 11, 1)])
            tab_idx = 0

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
                        tabs[tab_idx].image(f"{auto_labeler_classify.output_path}/preview_{i}.png")
                        tab_idx += 1
                
                progress_bar.progress(i / len(auto_labeler_classify.image_paths), text="Processing images")
            st.success("Labeling complete")



if __name__ == "__main__":
    main()
