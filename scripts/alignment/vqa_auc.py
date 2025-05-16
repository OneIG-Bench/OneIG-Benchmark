
import os
import re
import megfile
import pandas as pd
import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from copy import deepcopy
from dsg.parse_utils import parse_dependency_output, parse_question_output
from tqdm import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
# print("Loading mPLUG-large")

# vqa_model = MPLUG()
model_names = []

cache_dir = ""
os.makedirs(cache_dir, exist_ok=True)

def qwen_execution(images_path, question):
    messages = []
    for image_path in images_path:
        message = [
            {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", 
                    "text": 
                    f"""
                        {question}. Please give your answer as either "Yes" or "No" with no additional explanations.
                    """
                },
            ],
            }
        ]
        messages.append(message)
        
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Inference: Generation of the output
    with torch.no_grad():
    # Batch Inference
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    # print(output_texts)

    return output_texts

def extract_section_content(input_text, start_section, end_section):
    # 正则表达式匹配开始部分到结束部分之间的内容
    pattern = re.compile(f"{start_section}\n(.*?)(?={end_section}|<end>)", re.DOTALL)
    match = pattern.search(input_text)
    
    if match:
        # 返回提取的内容
        return match.group(1).strip()
    return None

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

def modify_in_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()  # 读取所有行

    modified_lines = []
    for line in lines:
        # 如果这一行是 "question"，则替换为 "<question>"
        if line.strip() == "question":
            modified_lines.append("<question>\n")
        elif line.strip() == "dependency":
            modified_lines.append("<dependency>\n")
        else:
            modified_lines.append(line)  # 保持原来的内容

    # 将修改后的内容转换为一个字符串
    modified_text = ''.join(modified_lines)

    # 输出修改后的字符串
    return modified_text

# Function to read all .txt files in subdirectories in order
def process_all_txt_files(qa_path):
    txt_files = []
    data = {"img_name":[],}
    for model_name in model_names:
        data[f"{model_name}"] = []
        data[f"{model_name}_filter"] = []
    for root, dirs, files in os.walk(qa_path):
        # 判断当前目录是否在 'object' 文件夹下
        if 'object' in os.path.basename(root):  # 确保在 "object" 文件夹下
            # 只添加当前目录下的所有 .txt 文件
            txt_files.extend([os.path.join(root, f) for f in files if f.endswith('.txt')])
    
    print(txt_files)
    # Sort the files based on their order in the directory structure
    txt_files.sort()
    
    cnt_txt = 0

    for file in tqdm(txt_files, desc="Processing files", unit="file"):
        input_text = modify_in_file(file)       
        input_text += "<end>"   
        
        dependency_content = extract_section_content(input_text, "<dependency>", "<question>")
        question_content = extract_section_content(input_text, "<question>", "<end>")
        
        # print(question_content)
        qid2dependency = parse_dependency_output(dependency_content)
        qid2question = parse_question_output(question_content)
        
        cnt_txt += 1
        
        # print(qid2dependency)   
        # print(qid2question)
        
        file_name = file.split('/')[-1].split('.')[0]
        class_name = file.split('/')[-2]
        res = general_ability_score(file_name, class_name, qid2dependency, qid2question)
        
        data["img_name"].append(res['img'])
        for key, value in res.items():
            if (key != 'img'):
                data[f'{key}'].append(value['score'])
                data[f'{key}_filter'].append(value['filter_score'])
        for model_name in model_names:
            if len(data[f"{model_name}"]) != len(data["img_name"]):
                data[f"{model_name}"].append(None)
                data[f"{model_name}_filter"].append(None)
                
        len_check = []
        len_check.append(len(data["img_name"]))
        for model_name in model_names:
            len_check.append(len(data[f"{model_name}"]))
            len_check.append(len(data[f"{model_name}_filter"]))
        print("len_check: ", len_check)
    
    df = pd.DataFrame(data)        
    
    mean_values = df.drop(columns='img_name').mean()

    # Append the mean values as the last row
    df.loc['Average'] = ['Average'] + mean_values.tolist()

    # Save the dataframe to a CSV file
    csv_path = f'general_scores.csv'
    df.to_csv(csv_path, index=False)
  

def general_ability_score(file_name, class_name, qid2dependency, qid2question):
    qid2answer = {}
    qid2scores = {}
    
    oss_path = f"{class_name}/*/{file_name}*"
    
    img_path_list_init = megfile.smart_glob(oss_path)
    
    img_path_list = [img_path for img_path in img_path_list_init if img_path.split('/')[-2] in model_names]
    
    print(img_path_list)
    res = {}
    res["img"] = file_name
    cnt = 0
    for img_path in img_path_list:
        img_grid = (2, 2)
        split_img_list = split_2x2_grid(img_path, img_grid, cache_dir)
        model_name = img_path.split('/')[-2]
        res[model_name] = {}
        for id, question in qid2question.items():
            images_path = split_img_list
            answer = qwen_execution(images_path, question)
            qid2answer[id] = answer
            qid2scores[id] = [float(ans == "Yes") for ans in answer]
            # print(qid2scores[id])
        # 3) zero-out scores from invalid questions
        qid2scores_after_filtering = deepcopy(qid2scores)
        for img_idx in range(len(split_img_list)):
            for id, parent_ids in qid2dependency.items():
                # zero-out scores if parent questions are answered 'no'
                any_parent_answered_no = False
                for parent_id in parent_ids:
                    if parent_id == 0:
                        continue
                    try:
                        if qid2scores[parent_id][img_idx] == 0:
                            any_parent_answered_no = True
                            break
                        else:
                            continue
                    except:
                        print(qid2scores)
                        print(parent_id)
                if any_parent_answered_no:
                    qid2scores_after_filtering[id][img_idx] = 0.0
        # print(qid2scores)
        # print(qid2scores_after_filtering)
        if len(split_img_list) == 0:
            res[model_name]["score"] = None
            res[model_name]["filter_score"] = None       
        else:
            sum_of_score = 0
            sum_of_filter_score = 0
            for question_id in range(len(qid2scores)):
                for img_idx in range(len(split_img_list)):
                    # print("qid", qid2scores[question_id + 1])
                    sum_of_score += qid2scores[question_id + 1][img_idx]
                    sum_of_filter_score += qid2scores_after_filtering[question_id + 1][img_idx]
                              
            res[model_name]["score"] = sum_of_score / (len(qid2scores) * len(split_img_list))
            res[model_name]["filter_score"] = sum_of_filter_score  / (len(qid2scores_after_filtering) * len(split_img_list))     
        cnt += 1
        # if (cnt == 2):
        #     print(res)
        #     break
    return res

if __name__ == "__main__":
    # Call the function to read all .txt files and print the result
    qa_path = ""
    process_all_txt_files(qa_path)
            