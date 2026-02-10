from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import megfile
import shutil
import pandas as pd
from tqdm import tqdm
from scripts.utils.utils import parse_args, split_mxn_grid, save2csv, on_rm_error

import json
from copy import deepcopy
from scripts.utils.inference import Qwen2_5VLBatchInferencer

import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

inferencer = Qwen2_5VLBatchInferencer("Qwen/Qwen2.5-VL-7B-Instruct")

def alignment_score(img_path, questions, dependencies, img_grid, cache_dir):
    score = {}
    
    if len(img_path) == 1:
        split_img_list = split_mxn_grid(img_path[0], img_grid, cache_dir)
        if len(split_img_list) == 0:
            return None    
    else:
        return None
    
    for id, question in questions.items():
        images_path = split_img_list
        batch_answer = inferencer.infer_semantic(images_path, question)
        score[id] = [float(ans == "Yes") for ans in batch_answer]
        
    filter_score = deepcopy(score)
    for img_idx in range(len(split_img_list)):
        for id, parent_ids in dependencies.items():
            any_parent_answered_no = False
            for parent_id in parent_ids:
                if parent_id == 0:
                    continue
                try:
                    if score[parent_id][img_idx] == 0:
                        any_parent_answered_no = True
                        break
                    else:
                        continue
                except:
                    print("The score is not a number.")
            if any_parent_answered_no:
                filter_score[id][img_idx] = 0

    sum_of_filter_score = [0] * len(split_img_list)
    for question_id in range(len(filter_score)):
        for img_idx in range(len(split_img_list)):
            sum_of_filter_score[img_idx] += filter_score[question_id + 1][img_idx]
    
    sum_of_filter_score = [img_score / len(filter_score) for img_score in sum_of_filter_score]
    
    return sum(sum_of_filter_score)  / len(sum_of_filter_score) 
    
def main():
    args = parse_args()
    
    cache_dir = f"tmp_{formatted_time}"
    os.makedirs(cache_dir, exist_ok=True)

    question_dependency_dir = "scripts/alignment"
    
    alignment_score_csv = f"results/alignment_score_{args.mode}_{formatted_time}.csv"
    alignment_prompt_score_csv = f"results/alignment_prompt_score_{args.mode}_{formatted_time}.csv"
    os.makedirs(os.path.dirname(alignment_score_csv), exist_ok=True)
    
    # save the alignment score of each method
    score_csv = pd.DataFrame(index=args.model_names, columns=["alignment"])
    # save the score of each prompt on each method to calculate average alignment score
    score_of_prompt_csv = pd.DataFrame(columns=args.model_names)
    
    for class_item in args.class_items:

        print(f"We process {class_item} now.")

        if args.mode == "EN":
            question_dependency_json_dir = question_dependency_dir + '/Q_D/' + class_item + '.json'
        else:
            question_dependency_json_dir = question_dependency_dir + '/Q_D/' + class_item + '_zh.json'
 
        with open(question_dependency_json_dir, "r", encoding="utf-8") as f:
            question_dependency = json.load(f)
        
        for key, item in tqdm(question_dependency.items(), desc=f"Processing {class_item}"):

            if isinstance(item["question"], str):
                item["question"] = {int(k): v for k, v in json.loads(item["question"]).items()}
            if isinstance(item["dependency"], str):
                item["dependency"] = {int(k): v for k, v in json.loads(item["dependency"]).items()}

            for model_id, model_name in enumerate(args.model_names):
                
                img_grid = (int(args.image_grid[model_id].split(',')[0]), int(args.image_grid[model_id].split(',')[-1]))
                 
                image_path = megfile.smart_glob(args.image_dirname + '/' + class_item + '/' + model_name + '/' + key + '*')
                
                result = alignment_score(image_path, item["question"], item["dependency"], img_grid, cache_dir)
                
                score_of_prompt_csv.loc[f"{class_item}_{key}", model_name] = result

    mean_values = score_of_prompt_csv.mean()
    score_csv["alignment"] = mean_values.values
    save2csv(score_csv, alignment_score_csv)
    
    score_of_prompt_csv = score_of_prompt_csv.sort_index()
    save2csv(score_of_prompt_csv, alignment_prompt_score_csv)

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, onerror=on_rm_error)
        
if __name__ == "__main__":
    main()