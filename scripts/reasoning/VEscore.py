import torch
import sys
import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
# from safetensors import safe_open
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Func
import numpy as np
from torchvision import transforms
from PIL import Image
import pandas as pd
from transformers import CLIPImageProcessor
# from accelerate import Accelerator
# from accelerate.logging import get_logger
# from accelerate.utils import ProjectConfiguration
# from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from glob import glob
import megfile 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Styleencoder_supp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        init_tau, init_b = np.log(10), -10
        self.t_prime = torch.nn.Parameter(torch.ones([]) * init_tau)
        self.b = torch.nn.Parameter(torch.ones([]) * init_b)
        
def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)

image_encoder_path = ""
pretrained_path =''

weight_dtype=torch.float16
image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_path).to(device,dtype=weight_dtype)


image_encoder_supp =Styleencoder_supp().to(device,dtype=weight_dtype)

state_dict = torch.load(image_encoder_path, map_location="cpu")


if 'image_encoder' in state_dict.keys():
    image_encoder.load_state_dict(state_dict["image_encoder"], strict=False)
    
clip_image_processor = CLIPImageProcessor()


def extract_style_embed(image_path):
    image_ref=Image.open(image_path).convert('RGB')
    with torch.no_grad():
        image_ref_1_ids= clip_image_processor(images=image_ref, return_tensors="pt").pixel_values
        image_ref_1_output = image_encoder(
                    image_ref_1_ids.to(device, dtype=weight_dtype))
        image_ref_1_embeds = image_ref_1_output.image_embeds
        z1 = l2_normalize(image_ref_1_embeds)
    
    return z1




def split_2x2_grid(image_path, grid_size, cache_dir):
    # Load the 2x2 grid image
    with megfile.smart_open(image_path, 'rb') as f:
        grid_image = Image.open(f)

        # Get the dimensions of the grid image
        width, height = grid_image.size

        # Calculate the dimensions of each individual image
        individual_width = width // grid_size[0]
        individual_height = height // grid_size[1]

        # Create a list to store the individual images
        image_list = []

        # Split the grid into individual images
        for i in range(grid_size[1]):
            for j in range(grid_size[0]):
                # Define the box (left, upper, right, lower)
                box = (
                    j * individual_width,      # left
                    i * individual_height,     # upper
                    (j + 1) * individual_width,  # right
                    (i + 1) * individual_height   # lower
                )
                
                # Crop the image
                individual_image = grid_image.crop(box)
                
                # Append the cropped image to the list
                image_list.append(individual_image)
    image_path_list = []
    for i, image in enumerate(image_list):
        image_path = os.path.join(cache_dir, f"{i}.jpg")
        image.save(image_path)
        image_path_list.append(image_path)
    return image_path_list


    
def get_name_en_from_csv(csv_dir):
    # 创建一个空列表来存储所有的 "name_en"
    name_en_list = []

    # 遍历目录中的所有文件
    for filename in os.listdir(csv_dir):
        # 只处理 CSV 文件
        if filename.endswith(".csv"):
            # 构建文件的完整路径
            file_path = os.path.join(csv_dir, filename)
            
            # 读取 CSV 文件
            try:
                df = pd.read_csv(file_path)
                # 如果 "name_en" 列存在，提取该列
                if "name_en" in df.columns:
                    name_en_list.extend(df["name_en"].dropna().tolist())  # dropna() 确保不包含 NaN 值
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    return name_en_list

dataset_path = ''
model_names = []
class_items = ["anime"]
csv_dir = ""
cache_dir = "../tmp"
os.makedirs(cache_dir, exist_ok=True)
image_base_dir = "../reference_image"
# style_score_csv = f"../style_score_max.csv"
name_en_list = get_name_en_from_csv(csv_dir)
style_score_csv = "VE_score.csv"
style_point_csv = "VE_style_point_score.csv"
# csv_class_items = ["sentence_bert"]
score_csv = pd.DataFrame(index=model_names, columns=class_items)
score_point_csv = pd.DataFrame(index=name_en_list, columns=model_names)
# Use tqdm to add progress bar for iterating through csv files
for class_item in class_items:
    print(f"We process {class_item} now.")
    csv_file = os.path.join(csv_dir, f"{class_item}.csv")
    df = pd.read_csv(csv_file)
    for model_name in model_names:
        cnt = 0
        print(f"It is {model_name} time.")
        img_grid = (2, 2)
        image_dir = dataset_path + class_item + '/' + model_name
        img_list = megfile.smart_glob(image_dir + '/*')
        print(f"We fetch {len(img_list)} images.")
        img_list = sorted(img_list)
        style_score = []
        # Use tqdm to add progress bar for iterating through rows in the DataFrame
        for idx, img_path in tqdm(enumerate(img_list), total=len(img_list), desc="Processing images"):
            img_name = img_path.split('/')[-1].split('.')[0]
            style =  str(df[df.iloc[:, 2] == img_name].iloc[0, 6])
            if (style == "") or ("nan" in style) or ("None" in style) or ('monochrome' in style):
                continue
            else:
                style = str(style).lower().replace(' ', '_')
            if ('illustration_design' in style):
                continue
            split_img_list = split_2x2_grid(img_path, img_grid, cache_dir)
            style_img_list = [os.path.join(image_base_dir, style, img) for img in os.listdir(os.path.join(image_base_dir, style)) if img.endswith(('.jpg', '.png', '.webp', '.jpeg'))]
            if len(style_img_list) == 0:
                print(style)
                break
            style_embeds = []         
            for style_img_path in style_img_list:
                style_embeds.append(extract_style_embed(style_img_path))
            for num, split_img_path in enumerate(split_img_list):
                embed = extract_style_embed(split_img_path)
                logits=-2.0
                for style_img_idx in range(len(style_embeds)):
                    # logits = torch.matmul(embed, style_embeds[style_img_idx].T)
                    logits=max(logits,torch.matmul(embed, style_embeds[style_img_idx].T))
                    # style_score.append(embed @ style_embeds[style_img_idx].T)
                style_score.append(logits)
            score_point_csv.loc[img_name, model_name] = logits.item()
            cnt += 1
            # if cnt == 10:
            #     break
        if len(style_score) != 0:
            score_csv.loc[model_name, class_item] = sum(style_score)/len(style_score)
        else:
            score_csv.loc[model_name, class_item] = 0


score_csv.to_csv(style_score_csv)

print(f"Results saved to {style_score_csv}") 

score_point_csv.to_csv(style_point_csv)
print(f"Results saved to {style_point_csv}")