import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import requests

def main():
    print("Hello from auto-label!")

    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # Load image
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Check for cats and remote controls
    text = "a cat. a remote control." # Labels

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    # Draw boxes
    draw = ImageDraw.Draw(image)
    scores_labels_boxes = zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"])
    for score, label, box in scores_labels_boxes:
        draw.rectangle(box.cpu().tolist(), outline="red", width=3)
        draw.text((box[0], box[1]), f"{label}: {score:4f}", fill="red")

    image.show()

if __name__ == "__main__":
    main()
