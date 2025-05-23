import os
import json
import datetime
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from collections import defaultdict
import megfile
from tqdm import tqdm

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


def inference(model, processor, img_path, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path,},
                {"type": "text", "text": prompt,},
            ],
        }
    ]
  

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text


if __name__ == "__main__":
    # model_name = "Qwen2.5-VL-3B-Instruct"
    model_name = "Qwen2.5-VL-7B-Instruct"
    # model_name = "Qwen2.5-VL-72B-Instruct"
    
    image_dir = ""
    
    json_dir=""
    os.makedirs(json_dir, exist_ok=True)
    cache_dir = ""
    os.makedirs(cache_dir, exist_ok=True)

    
    # img_list = megfile.smart_glob(image_dir+'/**/*')
    
    img_list_webp = megfile.smart_glob(image_dir+'/**/*.webp')
    img_list_jpg = megfile.smart_glob(image_dir+'/**/*.jpg')
    img_list_png = megfile.smart_glob(image_dir+'/**/*.png')
    
    img_list = img_list_webp + img_list_jpg + img_list_png
    img_list = sorted(img_list)
    print(f"img_list: {len(img_list)}")

    print("=======")
    model_path = "Qwen2.5-VL-7B-Instruct"
    # model_path = f'{CKPT_HOME}/{model_name}' 
    # megfile.fs.fs_copy(model_path, cache_dir)
    print(f"model_path: {model_path}")
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map=torch.cuda.current_device(),
    )
    # default processer
    processor = AutoProcessor.from_pretrained(model_path)

    for i, img_path in tqdm(enumerate(img_list), total=len(img_list), desc="Processing images"):
        if("sana4_8" not in img_path) and ("kling" not in img_path) and ("recraft" not in img_path):
            continue
        print(img_path)
        img_grid = (2, 2)
        print(img_path)
        split_img_list = split_2x2_grid(img_path, img_grid, cache_dir)
        res_json = json_dir + img_path.split('/')[-2] + '/' + img_path.replace(".webp", ".json").replace(".png", ".json").split('/')[-1]
        if os.path.exists(res_json):
            continue
        else:
            os.makedirs(os.path.dirname(res_json), exist_ok=True)
            
        for num, split_img_path in enumerate(split_img_list):
            prompt = "Recognize the text in the image, only reply with the text content, but avoid repeating previously mentioned content. If no text is recognized, please reply with 'No text recognized'."
            # prompt = "识别图片中的文字，只回答文字内容，但避免重复之前提到的内容，如果没有识别到文字，请回复'未识别到文字'"
            # prompt = "识别图片中清晰可见的文字，只回答清楚的文字内容，跳过不清楚的文字内容，如果没有识别到文字，请回复‘未识别到文字’"
            ocr_res = inference(model, processor, split_img_path, prompt)             
            # print(f"ocr_res: {ocr_res}")
            # print(f"is_natural_ocr: {is_natural_ocr}")

            if os.path.exists(res_json):
                with open(res_json, "r") as f:
                    info = json.load(f)
            else:
                info = {}
            if str(num) not in info:
                info[str(num)] = {}
            info[str(num)][model_name] = ocr_res
            with open(res_json, "w") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
  
    os.system(f"rm -rf {cache_dir}")
    pass
