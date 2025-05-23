from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import shutil
import pandas as pd
from tqdm import tqdm
import megfile
import datetime
from dreamsim import dreamsim
from scripts.utils.utils import split_2x2_grid

current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"
model, preprocess = dreamsim(pretrained=True, device=device)

def img_similar_score(img_1_path, img_2_path):
    img1 = preprocess(Image.open(img_1_path)).to(device)
    img2 = preprocess(Image.open(img_2_path)).to(device)
    distance = model(img1, img2)     
    return distance.item()

# image_dirname is advised to set "images"
image_dirname = "images"
model_names = ["gpt-4o"]
class_items = ["anime", "human", "object", "text", "reasoning"]
cache_dir = "tmp"
os.makedirs(cache_dir, exist_ok=True)
diversity_score_csv = f"results/diversity_score_{formatted_time}.csv"
os.makedirs(os.path.dirname(diversity_score_csv), exist_ok=True)
score_csv = pd.DataFrame(index=model_names, columns=class_items)

for class_item in class_items:
    print(f"We process {class_item} now.")
    for model_name in model_names:
        print(f"It is {model_name} time.")
        img_grid = (2, 2)
        image_dir = image_dirname + class_item + '/' + model_name
        img_list = megfile.smart_glob(image_dir + '/*')
        img_list = sorted(img_list)
        print(f"We fetch {len(img_list)} images.")
        diversity_score = []
        for idx, img_path in tqdm(enumerate(img_list), total=len(img_list), desc="Processing images"):
            split_img_list = split_2x2_grid(img_path, img_grid, cache_dir)
            img_name = img_path.split('/')[-1].split('.')[0]
            score = []
            for i in range(len(split_img_list)):
                for j in range(i+1, len(split_img_list)):
                    prob = img_similar_score(split_img_list[i], split_img_list[j])
                    score.append(prob)
            if len(score) != 0:
                diversity_score.append(sum(score)/len(score))
            else:
                diversity_score.append(0)

        if len(diversity_score) != 0:
            score_csv.loc[model_name, class_item] = sum(diversity_score)/len(diversity_score)
        else:
            score_csv.loc[model_name, class_item] = 0


score_csv.to_csv(diversity_score_csv)

print(f"Results saved to {diversity_score_csv}")

if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)