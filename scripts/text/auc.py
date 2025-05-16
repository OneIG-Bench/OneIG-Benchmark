import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
import time
import datetime

def preprocess_string(s):
    cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', s) 
    if contains_chinese(cleaned):
        pattern = re.compile(r'[\u4e00-\u9fa5a-zA-Z0-9]') 
        s = ''.join(pattern.findall(s))

        return s.strip()

    normalized = re.sub(r'\s+', ' ', cleaned)  

    return normalized.strip()

def levenshtein_distance(s1, s2):
    # 创建一个矩阵来存储距离
    # s1, s2 = preprocess_string(s1), preprocess_string(s2)
    matrix = np.zeros((len(s1) + 1, len(s2) + 1))

    # 初始化矩阵
    for i in range(len(s1) + 1):
        matrix[i][0] = i
    for j in range(len(s2) + 1):
        matrix[0][j] = j

    # 填充矩阵
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,      # 删除
                matrix[i][j - 1] + 1,      # 插入
                matrix[i - 1][j - 1] + cost  # 替换
            )

    return matrix[len(s1)][len(s2)]

from collections import Counter
def contains_chinese(text):
    """检测文本是否包含中文字符"""
    return bool(re.search('[\u4e00-\u9fff]', text))


def calculate_char_match_ratio(text_gt, ocr_str):
    # 判断是否为中文
    if contains_chinese(text_gt):
        # 中文字符匹配逻辑
        gt_counter = Counter(text_gt)
        ocr_counter = Counter(ocr_str)
        total_matches = sum((gt_counter & ocr_counter).values())
        ratio = total_matches / len(text_gt) if len(text_gt) > 0 else 0.0
    else:
        # 英文单词匹配逻辑
        words_gt = text_gt.split()
        words_ocr = ocr_str.split()
        gt_counter = Counter(words_gt)
        ocr_counter = Counter(words_ocr)
        total_matches = sum((gt_counter & ocr_counter).values())
        total_gt_words = len(words_gt)
        ratio = total_matches / total_gt_words if total_gt_words > 0 else 0.0
    
    
    return total_matches, ratio, sum(gt_counter.values())

def get_name_en_from_csv(csv_dir):
    # 创建一个空列表来存储所有的 "name_en"
    name_en_list = []

    # 遍历目录中的所有文件
    for filename in os.listdir(csv_dir):
        # 只处理 CSV 文件
        if filename == "text.csv":
            # 构建文件的完整路径
            file_path = os.path.join(csv_dir, filename)
            
            # 读取 CSV 文件
            try:
                df = pd.read_csv(file_path)
                # 如果 "name_en" 列存在，提取该列
                if "name_en" in df.columns:
                    name_en_list.extend(df["name_en"].dropna().tolist()) 
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    return name_en_list

if __name__ == "__main__":
    csv_file = "text_content.xlsx"
    # model_name = "Qwen2.5-VL-3B-Instruct"
    model_name = "Qwen2.5-VL-7B-Instruct"
    # model_name = "Qwen2.5-VL-72B-Instruct"
    
    ocr_parents_dir = ""
    csv_dir = ""
    name_en_list = get_name_en_from_csv(csv_dir)
    model_names = []
    ocr_point_csv = ""
    score_point_csv = pd.DataFrame(index=name_en_list, columns=model_names)

    reward_dict_model = {
        'tot_validate_img' : [],
        'edit_distance' : [],
        'normalized_edit_distance' : [],
        'ac_raito' : [],
        'text_count_raito_norm' : [],
        'tot_word_match_raito' : []
    }

    
    eval_model_name = []
    
    for model_chosen_name in model_names:
            
        ocr_dir = os.path.join(ocr_parents_dir, model_chosen_name)         

        print(f"evaluating {ocr_dir.split('/')[-1]} with {model_name}")

        df = pd.read_excel(csv_file)
        data_dict = df.to_dict(orient='list')
        ocr_count_all=0
        gt_count_all=0
        reward_dict = {
            'edit_distance': [],
            'normalized_edit_distance': [],
            'ac_raito': [],
            'ac_filelist': [],
            'text_count_raito_norm':[],
        }
        for i, (name, text_gt) in enumerate(zip(data_dict['name_en'], data_dict['text_content'])):
            json_file = os.path.join(ocr_dir, name + '.json')
            if not os.path.exists(json_file):
                continue
            with open(json_file, 'r') as f:
                ocr_res = json.load(f)
            
            ocr_list= [v[model_name][0] for k, v in ocr_res.items()]
            
            ocr_list = [text.replace("\n豆包AI", "").replace("\n豆包 AI", "").replace("豆包AI", "") for text in ocr_list]
            text_gt = preprocess_string(text_gt)
            
            edit_distance_per = 0
            normalized_edit_distance_per = 0
            ac_ratio = 0
            text_count_ratio_norm = 0                          
            for num, ocr in enumerate(ocr_list):
                if ocr == 'No text recoganized':
                    ocr = ''
                ocr = ocr.replace("No text recognized.\n", ' ').replace("\nNo text recognized.", ' ').replace("No text recognized.", ' ')
                ocr = preprocess_string(ocr)
                
                edit_distance = levenshtein_distance(ocr, text_gt)
                ac = 1 if edit_distance == 0 else 0
                normalized_edit_distance = min(edit_distance / len(preprocess_string(text_gt)), 1)

                text_count, text_acc, sum_gt = calculate_char_match_ratio(text_gt,ocr)
                ocr_count_all += text_count
                gt_count_all += sum_gt

                reward_dict['edit_distance'].append(edit_distance)
                reward_dict['normalized_edit_distance'].append(normalized_edit_distance)
                reward_dict['ac_raito'].append(ac)
                reward_dict['text_count_raito_norm'].append(text_acc)
                edit_distance_per += edit_distance
                normalized_edit_distance_per += normalized_edit_distance
                ac_ratio += ac
                text_count_ratio_norm += text_acc   
                if ac == 1:
                    reward_dict['ac_filelist'].append(name + f'_[{str(num)}]')
                
            if len(ocr_list) == 0:
                score_point_csv.loc[name, model_chosen_name] = None
            else:
                if model_chosen_name not in model_names:
                    continue
                else:
                    score_point_csv.loc[name, model_chosen_name] = [ edit_distance_per/len(ocr_list), normalized_edit_distance_per/len(ocr_list), ac_ratio/len(ocr_list), text_count_ratio_norm/len(ocr_list)]
                
                

        reward_dict_model['tot_validate_img'].append(len(reward_dict['edit_distance']))
        for k, v in reward_dict.items():
            print(f"total number of samples: {len(v)}")
            if k == 'ac_filelist':
                continue
            else:
                print(f"number of {k}", len(v))
                reward_dict_model[k].append(np.mean(v))
        
        save_file = ocr_parents_dir.split('/')[-1] + '/' + model_chosen_name + '_ac_list.json'
        if gt_count_all != 0:
            reward_dict_model["tot_word_match_raito"].append(ocr_count_all/gt_count_all)
        else:
            reward_dict_model["tot_word_match_raito"].append(0)
            
        eval_model_name.append(model_chosen_name)
        
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        with open(save_file, 'w') as f:
            json.dump(reward_dict["ac_filelist"], f, indent=2, ensure_ascii=False)
        pass

column_data = {'': eval_model_name}
for key in reward_dict_model.keys():
    column_data[key] = reward_dict_model[key]

print(column_data)
df_result = pd.DataFrame(column_data)

# Save the table to a CSV file
save_file_path = "evaluation_results.csv"
df_result.to_csv(save_file_path, index=False)
print(f"Results saved to {save_file_path}")

score_point_csv.to_csv(ocr_point_csv)
print(f"Results saved to {ocr_point_csv}")
save_file_path

