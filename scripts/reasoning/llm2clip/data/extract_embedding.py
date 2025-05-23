import os
import json
import torch
import argparse
from glob import glob
import webdataset as wds
from llm2vec import LLM2Vec
from itertools import islice
from transformers import AutoModel, AutoConfig, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
_KEYS = ["shortSV_captions", "longSV_captions"]


def extrace_feature(tar_path, model):
    dataset = wds.WebDataset(tar_path)

    texts = {}
    keys = []
    for k in _KEYS:
        texts[k] = []

    for sample in islice(dataset, 0, 999999):
        key = sample["__key__"]
        keys.append(key)
        captions = json.loads(sample["json"])
        for key in texts.keys():
            texts[key].append(captions[key])

    text_embeddings = {}
    for caption_key in texts.keys():
        embeddings = model.encode(texts[caption_key], convert_to_tensor=True)
        text_embeddings[caption_key] = embeddings

    save_path = tar_path.replace(".tar", "_" + "text_embeddings" + ".tar")
    save_embeddings_to_tar(save_path, keys, text_embeddings)


def save_embeddings_to_tar(path, keys, embeddings):
    try:
        with wds.TarWriter(path) as dst:
            for i, key in enumerate(keys):
                sample = {}
                sample["__key__"] = key
                for k in _KEYS:
                    feautre = embeddings[k][i].numpy().tobytes()
                    sample[k] = feautre
                dst.write(sample)
        print("saved: ", path)
    except:
        print("failed to save: ", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embeddings from a dataset")
    parser.add_argument(
        "--data_path", type=str, default="cc3m", help="Path to the dataset"
    )
    args = parser.parse_args()

    llm_model_name = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"
    config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)
    llm_model = AutoModel.from_pretrained(
        llm_model_name,
        torch_dtype=torch.bfloat16,
        config=config,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model.config._name_or_path = (
        "meta-llama/Meta-Llama-3-8B-Instruct"  #  Workaround for LLM2VEC
    )
    model = LLM2Vec(
        llm_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512
    )

    paths = glob(args.data_path + "/*tar")
    paths = sorted(paths)

    for path in paths:
        extrace_feature(path, model)
    print("done")
