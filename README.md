# vlm_training

# Training the projector
Download the 558K subset of the LAION-CC-SBU dataset with BLIP captions from [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) . Images should go based on the folder structure determined in the readme file.




Download the annotation of the final mixture llava instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

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


