from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from dreamsim import dreamsim
import torch
import os
import pandas as pd
from tqdm import tqdm
import megfile
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


device = "cuda"
model, preprocess = dreamsim(pretrained=True, device=device)


def img_similar_score(img_path_1, img_path_2):
    img1 = preprocess(Image.open(img_path_1)).to(device)
    img2 = preprocess(Image.open(img_path_2)).to(device)
    distance = model(img1, img2) # The model takes an RGB image from [0, 1], size batch_sizex3x224x224
    
    return distance.item()


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

def get_name_en_from_csv(csv_dir):
    # 创建一个空列表来存储所有的 "name_en"
    name_en_list = []

    # 遍历目录中的所有文件
    for filename in os.listdir(csv_dir):
        # 只处理 CSV 文件
        if filename == "reasoning.csv":
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

dataset_path = ""
model_names = []
class_items = ["anime", "human", "reasoning", "object", "text"]
cache_dir = ""
csv_dir = ""
name_en_list = get_name_en_from_csv(csv_dir)
img_similar_csv = "diversity_score.csv"
img_similar_point_csv = "diversity_point_score.csv"
score_csv = pd.DataFrame(index=model_names, columns=class_items)
score_point_csv = pd.DataFrame(index=name_en_list, columns=model_names)
os.makedirs(cache_dir, exist_ok=True)
# Use tqdm to add progress bar for iterating through csv files
for class_item in class_items:
    print(f"We process {class_item} now.")
    csv_file = os.path.join(csv_dir, f"{class_item}.csv")
    df = pd.read_csv(csv_file)
    for model_name in model_names:
        print(f"It is {model_name} time.")
        img_grid = (2, 2)
        image_dir = dataset_path + class_item + '/' + model_name
        img_list = megfile.smart_glob(image_dir + '/*.{webp,png,jpg,jpeg}')
        print(f"We fetch {len(img_list)} images.")
        img_list = sorted(img_list)
        diversity_score = []
        cnt = 0
        # Use tqdm to add progress bar for iterating through rows in the DataFrame
        for idx, img_path in tqdm(enumerate(img_list), total=len(img_list), desc="Processing images"):
            split_img_list = split_2x2_grid(img_path, img_grid, cache_dir)
            img_name = img_path.split('/')[-1].split('.')[0]
            img_score = []
            for i in range(len(split_img_list)):
                for j in range(i+1, len(split_img_list)):
                    prob = img_similar_score(split_img_list[i], split_img_list[j])
                    diversity_score.append(prob)
                    img_score.append(prob)
            if len(img_score)!=0:
                score_point_csv.loc[img_name, model_name] = sum(img_score)/len(img_score)
            else:
                score_point_csv.loc[img_name, model_name] = None
                
            cnt = cnt + 1
            
        if len(diversity_score) != 0:
            score_csv.loc[model_name, class_item] = sum(diversity_score)/len(diversity_score)
        else:
            score_csv.loc[model_name, class_item] = 0


score_csv.to_csv(img_similar_csv)

print(f"Results saved to {img_similar_csv}")

score_point_csv.to_csv(img_similar_point_csv)

print(f"Results saved to {img_similar_point_csv}")
            