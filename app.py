import streamlit as st
import os

import torch
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    AutoModelForImageClassification,
)

def data_selector(folder_path='datasets'):
    folders = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a folder', folders)
    return os.path.join(folder_path, selected_filename)

@st.cache_resource
def load_model(task, model_id, device):
    processor = AutoProcessor.from_pretrained(model_id)
    if task == "object_detection":
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    elif task == "image_classification":
        model = AutoModelForImageClassification.from_pretrained(model_id)
    else:
        raise ValueError(f"Task {task} not supported")

    if device=="auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model.to(device)
    return processor, model


def main():
    # Start streamlit app
    st.title("Image Auto-Labeling")

    # Initialize models
    task = st.sidebar.selectbox(
        "Task",
        ["object_detection", "image_classification"],
        help="Select the task to perform",
    )
    model_id = st.sidebar.text_input(
        "Model ID",
        value="IDEA-Research/grounding-dino-base",
        help="The model to use for the task",
    )
    device = st.sidebar.radio(
        "Device", ["auto", "cpu", "cuda", "mps"], help="The device to run the model on"
    )
    processor, model = load_model(task, model_id, device)

    # Select dataset
    dataset_dir = data_selector()

if __name__ == "__main__":
    main()
