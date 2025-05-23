import os
import json
import torch
import logging
from llm2vec import LLM2Vec
from typing import List, Dict, Any
from transformers import AutoModel, AutoConfig, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG = {
    "llm_model_name": "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned",
    "flickr": {
        "ann_path": "eval_data/flickr30k/test.json",
        "root": "eval_data/flickr30k/",
        "save_filename": "flickr30k_8B_llm_features.dpt"
    },
    "coco": {
        "ann_path": "eval_data/coco/coco_karpathy_test.json",
        "root": "eval_data/coco/",
        "save_filename": "coco_8B_llm_features.dpt"
    },
    "sharegpt4v": {
        "path": "eval_data/sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json",
        "ann_path": "eval_data/sharegpt4v/validation_1k.json",
        "root": "eval_data/sharegpt4v/",
        "save_filename": "sv_8B_llm_features.dpt",
        "total_len": 1000
    },
    "urban1k": {
        "ann_path": "eval_data/Urban1k/test.json",
        "root": "eval_data/Urban1k",
        "save_filename": "urban1k_8B_llm_features.dpt"
    },
    "docci": {
        "path": "eval_data/docci/docci_descriptions.jsonlines",
        "ann_path": "eval_data/docci/test.json",
        "root": "eval_data/docci",
        "save_filename": "docci_8B_llm_features.dpt"
    }
}

def load_json(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON file {file_path}: {e}")
        raise

def save_embeddings(embeddings: torch.Tensor, save_path: str) -> None:
    try:
        torch.save(embeddings, save_path)
        logging.info(f"Embeddings saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save embeddings to {save_path}: {e}")
        raise

def process_multi_texts_dataset(data: List[Dict[str, Any]], llm_model: LLM2Vec, save_path: str) -> None:
    texts = [caption for item in data for caption in item['caption']]
    with torch.no_grad():
        embeddings = llm_model.encode(texts, convert_to_tensor=True, batch_size=196)
    
    texts_num = len(data[0]['caption'])
    embeddings = embeddings.view(-1, texts_num, embeddings.size(-1))
    save_embeddings(embeddings, save_path)
    
def process_dataset(texts: List, llm_model: LLM2Vec, save_path: str) -> None:
    with torch.no_grad():
        embeddings = llm_model.encode(texts, convert_to_tensor=True, batch_size=128)
    save_embeddings(embeddings, save_path)

def flickr(llm_model: LLM2Vec) -> None:
    config = CONFIG["flickr"]
    data = load_json(config["ann_path"])
    save_path = os.path.join(config["root"], config["save_filename"])
    process_multi_texts_dataset(data, llm_model, save_path)

def coco(llm_model: LLM2Vec) -> None:
    config = CONFIG["coco"]
    data = load_json(config["ann_path"])
    save_path = os.path.join(config["root"], config["save_filename"])
    process_multi_texts_dataset(data, llm_model, save_path)

def sharegpt4v(llm_model: LLM2Vec) -> None:
    config = CONFIG["sharegpt4v"]
    data = load_json(config["path"])[:config["total_len"]]
    captions = []
    for it in data:
        dic = {}
        dic['caption'] = it['conversations'][1]['value']
        dic['image'] = it['image']
        captions.append(dic)
    
    json.dump(captions, open(config['ann_path'], 'w'))
    
    texts = [item['caption'] for item in captions]
    save_path = os.path.join(config["root"], config["save_filename"])
    process_dataset(texts, llm_model, save_path)
    

def urban1k(llm_model: LLM2Vec) -> None:
    config = CONFIG["urban1k"]
    eval_data = []
    for i in range(1, 1001):
        caption_path = os.path.join(config["root"], f'caption/{i}.txt')
        image_path = os.path.join(config["root"], f'image/{i}.jpg')
        caption = open(caption_path, 'r').readlines()[0]
        eval_data.append({'caption': caption, 'image': image_path})
    
    json.dump(eval_data, open(config['ann_path'], 'w'))
    
    texts = [item['caption'] for item in eval_data]
    save_path = os.path.join(config["root"], config["save_filename"])
    process_dataset(texts, llm_model, save_path)

def docci(llm_model: LLM2Vec) -> None:
    config = CONFIG["docci"]
    data = open(config["path"], 'r').readlines()
    eval_data = []
    for line in data:
        dic = json.loads(line)
        if dic['split'] == "test":
            eval_data.append({'caption': dic['description'], 'image': dic['image_file']})
    
    json.dump(eval_data, open(config['ann_path'], 'w'))
    
    texts = [item['caption'] for item in eval_data]
    save_path = os.path.join(config["root"], config["save_filename"])
    process_dataset(texts, llm_model, save_path)

def main() -> None:
    llm_model_name = CONFIG["llm_model_name"]
    config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)
    llm_model = AutoModel.from_pretrained(
        llm_model_name,
        torch_dtype=torch.bfloat16,
        config=config,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = LLM2Vec(llm_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)
    
    flickr(model)
    coco(model)
    sharegpt4v(model)
    urban1k(model)
    docci(model)

if __name__ == '__main__':
    main()