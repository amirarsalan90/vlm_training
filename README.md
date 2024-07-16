# vlm_training
This repository includes training code for the Vision Language Model (VLM) using the Transformers library. The LLM used is [phi-1_5](https://huggingface.co/microsoft/phi-1_5) and the vision model is [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14). The LLM is relatively lightweight, with only 1.5 billion parameters. This makes it feasible to train on consumer-grade GPUs, such as the NVIDIA RTX 4090.

The training process involves two steps:

1. Training the projector (while freezing the LLM and Vision tower)
2. LORA fine-tuning for instruction tuning

# Training the Projector
Download the 558K subset of the LAION-CC-SBU dataset with BLIP captions from [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) . Organize the images according to the folder structure described in the README file.

# LORA Instruction Fine-tuning
Download the annotation of the final mixture llava instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from following datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading all of them, organize the data as follows in `data` based on the following structure:


Data Folder Structure
```
data
├── instruction_finetuning
│   ├── coco
│   │   └── train2017
│   ├── gqa
│   │   └── images
│   ├── llava_v1_5_mix665k.json
│   ├── ocr_vqa
│   │   └── images
│   ├── text_vqa
│   │   └── train_images
│   └── vg
│       ├── VG_100K
│       └── VG_100K_2
└── projector_training
    ├── blip_laion_cc_sbu_558k.json
    └── images
```


