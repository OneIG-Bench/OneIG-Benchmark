from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import json
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import CLIPImageProcessor
import torch
from llm2clip.llm2vec import LLM2Vec
import os
import pandas as pd
from tqdm import tqdm
import megfile
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
model_name_or_path = "microsoft/LLM2CLIP-Openai-L-14-336"  # or /path/to/local/LLM2CLIP-Openai-L-14-336
model = AutoModel.from_pretrained(
    model_name_or_path, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True).to('cuda').eval()

llm_model_name = 'microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned'
config = AutoConfig.from_pretrained(
    llm_model_name, trust_remote_code=True
)
llm_model = AutoModel.from_pretrained(llm_model_name, torch_dtype=torch.bfloat16, config=config, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model.config._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'  # Workaround for LLM2VEC
l2v = LLM2Vec(llm_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)

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

def text_img_similarity_score(image_path, text_prompt):
    try:
        captions = []
        captions.append(text_prompt)
        image = Image.open(image_path)
        input_pixels = processor(images=image, return_tensors="pt").pixel_values.to('cuda')
        text_features = l2v.encode(captions, convert_to_tensor=True).to('cuda')

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.get_image_features(input_pixels)
            text_features = model.get_text_features(text_features)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.T

        return text_probs.item()
    except:
        return None

dataset_path = ''
model_names = ''
class_items = ''
cache_dir = ''
csv_dir = ''
name_en_list = get_name_en_from_csv(csv_dir)
answer_dir = ""
text_img_similar_csv = "similar_score.csv"
text_img_similar_point_csv = "similar_point_score.csv"
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
        similar_score = []
        cnt = 0
        # Use tqdm to add progress bar for iterating through rows in the DataFrame
        for idx, img_path in tqdm(enumerate(img_list), total=len(img_list), desc="Processing images"):
            split_img_list = split_2x2_grid(img_path, img_grid, cache_dir)
            img_name = img_path.split('/')[-1].split('.')[0]
            text_prompt = df[df.iloc[:, 2] == img_name].iloc[0, 4]
            answer_base_json_dir = answer_dir + '/' + img_name + ".json"
            with open(answer_base_json_dir, 'r', encoding='utf-8') as infile:
                data = json.load(infile)
                answer_base = data["prompts"]
            img_score = []
            for num, split_img_path in enumerate(split_img_list):
                prob = text_img_similarity_score(split_img_path, answer_base)
                similar_score.append(prob)
                img_score.append(prob)
            
            score_point_csv.loc[img_name, model_name] = sum(img_score)/len(img_score)
                
            cnt = cnt + 1
            
            # if (cnt == 2):
            #     break

        if len(similar_score) != 0:
            score_csv.loc[model_name, class_item] = sum(similar_score)/len(similar_score)
        else:
            score_csv.loc[model_name, class_item] = 0


score_csv.to_csv(text_img_similar_csv)

print(f"Results saved to {text_img_similar_csv}")

score_point_csv.to_csv(text_img_similar_point_csv)

print(f"Results saved to {text_img_similar_point_csv}")
            
            