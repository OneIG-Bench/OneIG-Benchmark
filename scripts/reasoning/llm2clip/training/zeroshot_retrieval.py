import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from eva_clip import get_cast_dtype, get_tokenizer
from .precision import get_autocast

def evaluate(model, dataloader, tokenizer,  device, precision, distributed=False,recall_k_list=[1, 5, 10]):
    """
    Evaluate the model on the given dataset

    Parameters
    ----------
    
    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    device: cpu/cuda

    precision: floating point precision

    recall_k_list: list of int
        recall@k k's to use
    
    Returns
    -------
    
    dict of retrieval metrics
    """
    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []
    num_batches = dataloader.num_batches
    dataloader = dataloader_with_indices(dataloader)
    autocast = get_autocast(precision)
    cast_dtype = get_cast_dtype(precision)
    pbar = tqdm(total=num_batches)
    # for batch_images, batch_texts, inds in tqdm(dataloader):
    for batch_images, batch_texts, inds in dataloader:
        batch_images = batch_images.to(device, dtype=cast_dtype)
        # tokenize all texts in the batch
        if tokenizer:
            batch_texts_tok = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(device)
        else:
            batch_texts_tok = torch.tensor([text for i, texts in enumerate(batch_texts) for text in texts]).to(device, dtype=cast_dtype)
        # store the index of image for each text
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]
        # compute the embedding of images and texts
        with torch.no_grad(), autocast():
            if distributed:
                batch_images_emb = F.normalize(model.module.encode_image(batch_images), dim=-1)
                batch_texts_emb = F.normalize(model.module.encode_text(batch_texts_tok), dim=-1)
            else:
                batch_images_emb = F.normalize(model.encode_image(batch_images), dim=-1)
                batch_texts_emb = F.normalize(model.encode_text(batch_texts_tok), dim=-1)
        
        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)
        
        pbar.update(1)
        
    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)

    # get the score for each text and image pair
    try:
        scores  = texts_emb @ images_emb.t()
    except:
        scores = texts_emb.float() @ images_emb.t().float()

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k)>0).float().mean().item()

    return metrics

def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end

def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


def retrieval_global(model, dataloader, tokenizer,  device, precision, distributed=False):
    autocast = get_autocast(precision)
    cast_dtype = get_cast_dtype(precision)
    vis_embeds_list = []
    text_features_list = []
    # dataloader = dataloader_with_indices(dataloader)
    for images, texts in tqdm(dataloader):
        images = images.to(device, dtype=cast_dtype)
        texts = torch.tensor(texts)
        texts = texts.to(device, dtype=cast_dtype)
        with torch.no_grad(), autocast():
            if distributed:
                image_features = F.normalize(model.module.encode_image(images), dim=-1)
                text_features = F.normalize(model.module.encode_text(texts), dim=-1)
            else:
                image_features = F.normalize(model.encode_image(images), dim=-1)
                text_features = F.normalize(model.encode_text(texts), dim=-1)
            vis_embeds_list.append(image_features)
            text_features_list.append(text_features)
        image_features = torch.cat(vis_embeds_list, dim=0)
        text_features = torch.cat(text_features_list, dim=0)
    
    i, correct, total  = 0, 0, 0
    for i in range(text_features.shape[0]):
        text = text_features[i]
        sim = text @ image_features.T
        sim = sim.squeeze()
        correct_i = torch.argmax(sim)

        if i==correct_i:
            correct = correct + 1
        total = total + 1
    t2i = correct/total
            
    i, correct, total  = 0, 0, 0
    for i in range(image_features.shape[0]):
        img = image_features[i]
        sim = img @ text_features.T
        sim = sim.squeeze()
        correct_i = torch.argmax(sim)

        if i==correct_i:
            correct = correct + 1
        total = total + 1
    i2t = correct/total
    
    metrics = {}
    metrics[f"image_retrieval_recall@1"] = t2i
    metrics[f"text_retrieval_recall@1"] = i2t

    def compute_recall(features1, features2, k_values):
        recalls = {}
        for k in k_values:
            correct = 0
            total = 0
            for i in range(features1.shape[0]):
                sim = features1[i] @ features2.T
                sim = sim.squeeze()
                _, topk_indices = torch.topk(sim, k)
                
                if i in topk_indices:
                    correct += 1
                total += 1
            
            recall_at_k = correct / total
            recalls[f"recall@{k}"] = recall_at_k
        
        return recalls
    
    k_values = [1, 5, 10, 25, 50]
    t2i_recalls = compute_recall(text_features, image_features, k_values)
    for k, recall in t2i_recalls.items():
        metrics[f"image_retrieval_{k}"] = recall
    
    # Image to Text Retrieval
    i2t_recalls = compute_recall(image_features, text_features, k_values)
    for k, recall in i2t_recalls.items():
        metrics[f"text_retrieval_{k}"] = recall

    return metrics


def retrieval_eval(model, data, epoch, args):
    if args.zeroshot_frequency == 0:
        return {}
    logging.info('Starting zero-shot retrieval.')
    
    tokenizer = get_tokenizer(args.model)
    tokenizer = None
    model.to(args.device)
    collect_results = {}
    if 'DOCCI'  in data:
        logging.info('Starting DOCCI.')
        results = retrieval_global(model, data['DOCCI'].dataloader, tokenizer, args.device, args.precision)
        for key in results.keys():
            collect_results['DOCCI/'+ key] = results[key]
    if 'sharegpt4v'  in data:
        logging.info('Starting Sharegpt4v.')
        results = retrieval_global(model, data['sharegpt4v'].dataloader, tokenizer, args.device, args.precision)
        for key in results.keys():
            collect_results['Sharegpt4v/'+ key] = results[key]
    if 'Urban1k'  in data:
        logging.info('Starting Urban1k.')
        results = retrieval_global(model, data['Urban1k'].dataloader, tokenizer, args.device, args.precision)
        for key in results.keys():
            collect_results['Urban1k/'+ key] = results[key]
    if 'ret_flickr' in data:
        logging.info('Starting Flickr.')
        results = evaluate(model, data['ret_flickr'].dataloader, tokenizer, args.device, args.precision)
        for key in results.keys():
            collect_results['flickr/'+ key] = results[key]
    if 'ret_coco' in data:
        logging.info('Starting COCO.')
        results = evaluate(model, data['ret_coco'].dataloader, tokenizer, args.device, args.precision)
        for key in results.keys():
            collect_results['coco/'+ key] = results[key]
            
    logging.info('Finished zero-shot retrieval.')
    return collect_results