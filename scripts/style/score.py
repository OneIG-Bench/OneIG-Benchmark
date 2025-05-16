
import os
import json
import megfile
import pandas as pd
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision import transforms
from tqdm import tqdm
from model import CSD_CLIP, convert_state_dict

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# init model
model = CSD_CLIP("vit_large", "default")

# load model
model_path = "models/checkpoint.pth"
checkpoint = torch.load(model_path, map_location="cpu")
state_dict = convert_state_dict(checkpoint['model_state_dict'])
model.load_state_dict(state_dict, strict=False)
model = model.cuda()

# normalization
normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
preprocess = transforms.Compose([
                transforms.Resize(size=224, interpolation=F.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

def is_black_image(image):
    # 检查图像是否为完全黑色图像
    pixels = image.load()  # 获取图像像素
    for i in range(image.width):
        for j in range(image.height):
            # 如果某个像素的颜色不是黑色，则返回 False
            if pixels[i, j] != (0, 0, 0):
                return False
    return True

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

                # Check if the cropped image is black
                if is_black_image(individual_image):
                    print(f"Detected a black image at position ({i},{j}) in {image_path}")
                else:
                    # Append the cropped image to the list if it's not black
                    image_list.append(individual_image)

    # Save the individual images
    image_path_list = []
    for i, image in enumerate(image_list):
        image_path = os.path.join(cache_dir, f"{i}.jpg")
        image.save(image_path)
        image_path_list.append(image_path)

    return image_path_list

def style_embed(image_path):
    with torch.no_grad():
        image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to("cuda")
        _, _, style_output = model(image)
    return style_output
def get_name_en_from_csv(csv_dir):
    # 创建一个空列表来存储所有的 "name_en"
    name_en_list = []

    # 遍历目录中的所有文件
    for filename in os.listdir(csv_dir):
        # 只处理 CSV 文件
        if filename == "anime.csv":
            # 构建文件的完整路径
            file_path = os.path.join(csv_dir, filename)
            
            # 读取 CSV 文件
            try:
                df = pd.read_csv(file_path)
                # 如果 "name_en" 列存在，提取该列
                if "name_en" in df.columns and "style" in df.columns:
                    # 过滤 style 列，排除为空和不等于 "monochrome" 和 "illustration design" 的项
                    filtered_df = df[df["style"].notna() & ~df["style"].isin(["monochrome", "illustration design"])]
                    
                    # 将符合条件的 name_en 添加到 name_en_list 中
                    name_en_list.extend(filtered_df["name_en"].dropna().tolist())  # dropna() 确保不包含 NaN 值
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    return name_en_list

dataset_path = ''
model_names = []
class_items = []
csv_dir = ""
name_en_list = get_name_en_from_csv(csv_dir)
cache_dir = ""
os.makedirs(cache_dir, exist_ok=True)
image_base_dir = ""
style_list = [d for d in os.listdir(image_base_dir) if os.path.isdir(os.path.join(image_base_dir, d))]
style_score_csv = "style_score.csv"
style_point_score_csv = "style_point_score.csv"
# csv_class_items = ["sentence_bert"]
score_csv = pd.DataFrame(index=model_names, columns=style_list)
score_point_csv = pd.DataFrame(index=name_en_list, columns=model_names)
# Use tqdm to add progress bar for iterating through csv files
for class_item in class_items:
    print(f"We process {class_item} now.")
    csv_file = os.path.join(csv_dir, f"{class_item}.csv")
    df = pd.read_csv(csv_file)
    
    for model_name in model_names:
        # 使用字典来存储每个风格对应的空列表
        style_dict = {style: [] for style in style_list}
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
            if (style == "") or ("nan" in style) or ("None" in style):
                continue
            else:
                style = str(style).lower().replace(' ', '_')
            if (style == "monochrome") or (style == "illustration_design"):
                continue
            split_img_list = split_2x2_grid(img_path, img_grid, cache_dir)
            style_img_list = [os.path.join(image_base_dir, style, img) for img in os.listdir(os.path.join(image_base_dir, style)) if img.endswith(('.jpg', '.png', '.webp', '.jpeg'))]
            if len(style_img_list) == 0:
                print(style)
                break
            style_embeds = []         
            for style_img_path in style_img_list:
                style_embeds.append(style_embed(style_img_path))
                
            style_score_per_list = []
            for num, split_img_path in enumerate(split_img_list):
                embed = style_embed(split_img_path)
                max_style_score = 0
                for style_img_idx in range(len(style_embeds)):
                    style_score_per = (embed @ style_embeds[style_img_idx].T).item()
                    if style_score_per > max_style_score:
                        max_style_score = style_score_per
                style_score.append(max_style_score)
                style_score_per_list.append(max_style_score)
            cnt += 1
            score_point_csv.loc[img_name, model_name] = sum(style_score_per_list) / len(style_score_per_list)
            style_dict[style].append(sum(style_score_per_list) / len(style_score_per_list))
    
        for style in style_list:
            if len(style_dict[style]) != 0:
                score_csv.loc[model_name, style] = sum(style_dict[style])/len(style_dict[style])
            else:
                score_csv.loc[model_name, style] = 0


score_csv.to_csv(style_score_csv)

print(f"Results saved to {style_score_csv}")

score_point_csv.to_csv(style_point_score_csv)

print(f"Results saved to {style_point_score_csv}")