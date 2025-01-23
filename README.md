# Auto Labeler

Auto Labeler is a tool for labeling images for Image Classification or Object Detection tasks. It utilizes Zero Shot models to achieve performant labeling on the given images from a user-defined set of labels. The labels could be anything, but performance will vary based on the pretrained model used. By default, [GroundingDINO](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino) is used for Object Detection while [CLIP](https://huggingface.co/docs/transformers/en/model_doc/clip) is used for Image Classification. 

This app uses [Streamlit](https://docs.streamlit.io/) for a basic UI. You can run this app by following the setup below:

## Setup

This repository uses [UV](https://astral.sh/blog/uv) as its python dependency management tool. Install UV by:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Initialize virtual env and activate
```bash
uv venv
source .venv/bin/activate
```

Install dependencies with:
```bash
uv sync
```

To try out the app, you can use the [Stanford Cars Dataset]() as your image source (its the one used in the Demo section), but feel free to use any other set of images.

Download the stanford cars dataset, and unzip:
```bash
curl -L -o ~/Downloads/stanford-cars-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/jessicali9530/stanford-cars-dataset
mkdir -p datasets/stanford-cars-dataset
unzip -d datasets/stanford-cars-dataset ~/Downloads/stanford-cars-dataset.zip 
```

Note: Make sure your images are placed under `datasets` directory at this project's root, as the app will look at that directory for the images.

## Demo

Run the app with the command below and it will launch it in your browser:
```bash
streamlit run app.py
```

![demo](assets/docs/sample.png)


## Citations

GroundingDINO
```BibTeX
@misc{liu2024groundingdinomarryingdino,
      title={Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection}, 
      author={Shilong Liu and Zhaoyang Zeng and Tianhe Ren and Feng Li and Hao Zhang and Jie Yang and Qing Jiang and Chunyuan Li and Jianwei Yang and Hang Su and Jun Zhu and Lei Zhang},
      year={2024},
      eprint={2303.05499},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2303.05499}, 
}
```

CLIP
```BibTeX
@misc{radford2021learningtransferablevisualmodels,
      title={Learning Transferable Visual Models From Natural Language Supervision}, 
      author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
      year={2021},
      eprint={2103.00020},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2103.00020}, 
}
```