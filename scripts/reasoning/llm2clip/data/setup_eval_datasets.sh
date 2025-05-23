#!/bin/bash

echo "flickr30k"
mkdir -p eval_data/flickr30k/
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json -O eval_data/flickr30k/test.json
mkdir -P eval_data/flickr30k/flickr30k-images

echo "coco"
mkdir -p eval_data/coco/
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json -P eval_data/coco/
mkdir eval_data/coco/val2014

echo "sharegpt4v"
mkdir -p eval_data/sharegpt4v/
wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/share-captioner_coco_lcs_sam_1246k_1107.json -P eval_data/sharegpt4v/

echo "Urban1k"
mkdir -p eval_data/
wget https://huggingface.co/datasets/BeichenZhang/Urban1k/resolve/main/Urban1k.zip -P eval_data/
unzip eval_data/Urban1k.zip -d eval_data/

echo "docci"
mkdir -p eval_data/docci/
wget  https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines -P eval_data/docci/

echo "Please download the images of flickr30k, coco2014, sharegpt4v and docci manually, and then change the paths in the eval_datasets.yaml accordingly"

