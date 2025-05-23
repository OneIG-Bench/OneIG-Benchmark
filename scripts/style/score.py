from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import megfile
import datetime
import pandas as pd
import shutil
import torch
torch.cuda.empty_cache()
import torchvision.transforms.functional as F
from torchvision import transforms
from tqdm import tqdm
from scripts.utils.utils import split_2x2_grid
from scripts.style.model import CSD_CLIP, convert_state_dict

current_time = datetime.datetime.now()

formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"

model = CSD_CLIP("vit_large", "default")

model_path = "scripts/style/models/checkpoint.pth"
checkpoint = torch.load(model_path, map_location="cpu")
state_dict = convert_state_dict(checkpoint['model_state_dict'])
model.load_state_dict(state_dict, strict=False)
model = model.cuda()

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
preprocess = transforms.Compose([
                transforms.Resize(size=224, interpolation=F.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

def style_embed(image_path):
    with torch.no_grad():
        image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to("cuda")
        _, _, style_output = model(image)
    return style_output

# image_dirname = 'images/anime'
image_dirname = 'images/anime'
model_names = ["gpt-4o"]
cache_dir = "tmp"
os.makedirs(cache_dir, exist_ok=True)

data_dir = "data.xlsx"
df = pd.read_excel(data_dir, dtype=str)

CSD_embed_parquet = "scripts/style/CSD_embed.parquet"
ref = pd.read_parquet(CSD_embed_parquet)

CSD_score_csv = f"results/CSD_score_{formatted_time}.csv"
os.makedirs(os.path.dirname(CSD_score_csv), exist_ok=True)
score_csv = pd.DataFrame(index=model_names, columns=["style"])

style_list = ['3d_rendering', 'abstract_expressionism', 'art_nouveau', 'baroque', 'celluloid', 'chibi', 'chinese_ink_painting', 'clay', 'comic', 'crayon', 'cubism', 'cyberpunk', 'fauvism', 'ghibli', 'graffiti', 'impressionism', 'lego', 'line_art', 'minimalism', 'pencil_sketch', 'pixar', 'pixel_art', 'pointillism', 'pop_art', 'rococo', 'stone_sculpture', 'ukiyo-e', 'watercolor', 'impasto']

for model_name in model_names:
    cnt = 0
    print(f"It is {model_name} time.")
    img_grid = (2, 2)
    image_dir = image_dirname + '/' + model_name
    img_list = megfile.smart_glob(image_dir + '/*')
    img_list = sorted(img_list)
    print(f"We fetch {len(img_list)} images.")
    CSD_score = []

    for idx, img_path in tqdm(enumerate(img_list), total=len(img_list), desc="Processing images"):
        id = img_path.split('/')[-1][:3]
        image_style =  str(df.loc[(df["category"] == "Anime_Stylization") & (df["id"] == id), "class"].values[0])
        if (image_style[:3] == "nan"):
            continue
        else:
            image_style = image_style.lower().replace(' ', '_')
        
        split_img_list = split_2x2_grid(img_path, img_grid, cache_dir)
        
        style_match_list = ref[ref["style"] == image_style]
        CSD_embeds = style_match_list["CSD_embed"].values    
        CSD_embeds = [torch.Tensor(CSD_embed).to(device) for CSD_embed in CSD_embeds]
        
        score = []
        for num, split_img_path in enumerate(split_img_list):
            embed = style_embed(split_img_path)
            max_style_score = 0
            for style_img_idx in range(len(CSD_embeds)):
                style_score_per = (embed @ CSD_embeds[style_img_idx].T).item()
                if style_score_per > max_style_score:
                    max_style_score = style_score_per
            score.append(max_style_score)
        
        if len(score) != 0:
            CSD_score.append(sum(score)/len(score))
        else:
            CSD_score.append(0)
            
    score_csv.loc[model_name, "style"] = sum(CSD_score) / len(CSD_score)

score_csv.to_csv(CSD_score_csv)

print(f"Results saved to {CSD_score_csv}")

if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
            