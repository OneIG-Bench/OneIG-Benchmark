from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import json
import shutil
import pandas as pd
from tqdm import tqdm
import megfile
import datetime
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import CLIPImageProcessor
from scripts.utils.utils import split_2x2_grid
from scripts.reasoning.llm2clip.llm2vec import LLM2Vec

current_time = datetime.datetime.now()

formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

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

# image_dirname is advised to set "images/reasoning"
image_dirname = 'images/reasoning'
model_names = ["gpt-4o"]
cache_dir = "tmp"
os.makedirs(cache_dir, exist_ok=True)

answer_dir = "scripts/reasoning/gt_answer"
reasoning_score_csv = f"results/reasoning_score_{formatted_time}.csv"
os.makedirs(os.path.dirname(reasoning_score_csv), exist_ok=True)
score_csv = pd.DataFrame(index=model_names, columns=["reasoning"])
os.makedirs(cache_dir, exist_ok=True)

for model_name in model_names:
    print(f"It is {model_name} time.")
    img_grid = (2, 2)
    image_dir = image_dirname + '/' + model_name
    img_list = megfile.smart_glob(image_dir + '/*')
    img_list = sorted(img_list)
    print(f"We fetch {len(img_list)} images.")
    reasoning_score = []
    for idx, img_path in tqdm(enumerate(img_list), total=len(img_list), desc="Processing images"):
        split_img_list = split_2x2_grid(img_path, img_grid, cache_dir)
        img_id = img_path.split('/')[-1][:3]
        print(img_id)
        answer_json_dir = answer_dir + '/' + img_id + ".json"
        with open(answer_json_dir, 'r', encoding='utf-8') as f:
            data = json.load(f)
            answer_text = data["prompts"]
        score = []
        for num, split_img_path in enumerate(split_img_list):
            prob = text_img_similarity_score(split_img_path, answer_text)
            score.append(prob)
            
        if len(score) != 0:
            reasoning_score.append(sum(score)/len(score))
        else:
            reasoning_score.append(0)

    if len(reasoning_score) != 0:
        score_csv.loc[model_name, "reasoning"] = sum(reasoning_score)/len(reasoning_score)
    else:
        score_csv.loc[model_name, "reasoning"] = 0


score_csv.to_csv(reasoning_score_csv)

print(f"Results saved to {reasoning_score_csv}")

if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
            
            