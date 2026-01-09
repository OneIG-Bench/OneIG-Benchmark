from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import megfile
import shutil
import pandas as pd
from tqdm import tqdm
from scripts.utils.utils import parse_args, split_2x2_grid, save2csv, on_rm_error

from scripts.text.text_utils import preprocess_string, clean_and_remove_hallucinations, levenshtein_distance, calculate_char_match_ratio
from scripts.utils.inference import Qwen2_5VLBatchInferencer

import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

def main():
    args = parse_args()
    cache_dir = f"tmp_{formatted_time}"
    os.makedirs(cache_dir, exist_ok=True)
    
    influencer = Qwen2_5VLBatchInferencer("Qwen/Qwen2.5-VL-7B-Instruct")
    
    if args.mode == "EN":
        text_csv_path = "scripts/text/text_content.csv"
        MAX_EDIT_DISTANCE = 100
    else:
        text_csv_path = "scripts/text/text_content_zh.csv"
        MAX_EDIT_DISTANCE = 50
    text_df = pd.read_csv(text_csv_path, dtype=str)

    text_score_csv = f"results/text_score_{args.mode}_{formatted_time}.csv"
    text_prompt_score_csv = f"results/text_prompt_score_{args.mode}_{formatted_time}.csv"
    os.makedirs(os.path.dirname(text_score_csv), exist_ok=True)
    
    score_csv = pd.DataFrame(index=args.model_names, columns=["ED", "CR", "WAC", "text score"])
    score_of_prompt_csv = pd.DataFrame(columns=args.model_names)

    for model_id, model_name in enumerate(args.model_names):
        
        print(f"It is {model_name} time.")
        
        img_grid = (args.image_grid[model_id], args.image_grid[model_id]) 
        
        edit_distances = []
        completion_ratios = []
        match_word_counts = []
        gt_word_counts = []
        
        for id, text_gt in tqdm(zip(text_df["id"], text_df["text_content"]), total=len(text_df), desc="Processing text"):
            word_count = len(text_gt.split())
            if (word_count > 60):
                max_new_tokens = 256
            else:
                max_new_tokens = 128
                
            text_gt_preprocessed = preprocess_string(text_gt)
            
            img_path = megfile.smart_glob(args.image_dirname + '/' + model_name + '/' +  id + '*')
            if len(img_path) != 1:
                score_of_prompt_csv.loc[id, model_name] = None
            else:
                split_img_list = split_2x2_grid(img_path[0], img_grid, cache_dir)    
                if  len(split_img_list) != 0:                 
                    ocr_results = influencer.infer_ocr(split_img_list, max_new_tokens)
                else:
                    score_of_prompt_csv.loc[id, model_name] = None
                
                text_ocr_list = clean_and_remove_hallucinations(ocr_results)
                
                ED_score = []
                CR_score = []
                WAC_score = []
                
                for text_ocr in text_ocr_list:
                    text_ocr_preprocessed = preprocess_string(text_ocr)
                    
                    edit_distance = levenshtein_distance(text_ocr_preprocessed, text_gt_preprocessed)
                    
                    completion_ratio = 1 if edit_distance == 0 else 0
                    
                    match_word_count, text_word_accuracy, gt_word_count = calculate_char_match_ratio(text_gt_preprocessed, text_ocr_preprocessed)
                    
                    edit_distances.append(edit_distance)
                    completion_ratios.append(completion_ratio)
                    match_word_counts.append(match_word_count)
                    gt_word_counts.append(gt_word_count)

                    ED_score.append(edit_distance)
                    CR_score.append(completion_ratio)
                    WAC_score.append(text_word_accuracy)

                score_of_prompt_csv.loc[id, model_name] = [(sum(ED_score)/len(ED_score)).item(), sum(CR_score)/len(CR_score), sum(WAC_score)/len(WAC_score)]

        ED = sum(edit_distances) / len(edit_distances)
        CR = sum(completion_ratios) / len(completion_ratios)
        WAC = sum(match_word_counts) / sum(gt_word_counts)
        
        score_csv.loc[model_name, "ED"] = ED
        score_csv.loc[model_name, "CR"] = CR
        score_csv.loc[model_name, "WAC"] = WAC
        score_csv.loc[model_name, "text score"] = 1 - min(MAX_EDIT_DISTANCE, ED) * (1 - CR) * (1 - WAC) / MAX_EDIT_DISTANCE

    save2csv(score_csv, text_score_csv)

    save2csv(score_of_prompt_csv, text_prompt_score_csv)

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, onerror=on_rm_error)

if __name__ == "__main__":
    main()