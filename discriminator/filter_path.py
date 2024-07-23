import json
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import cfg
from discriminator.model import Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
END_REL = "END"


@torch.no_grad()
def get_texts_embeddings(texts, tokenizer, model):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=cfg.MAX_LEN,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    embeddings = model(**inputs, return_dict=True).pooler_output
    return embeddings


@torch.no_grad()
def discriminate_path(path, word_list, MAX_HOP, tokenizer, retriever, discriminator):
    if not word_list or not path or path == [END_REL]:
        return {"score": 1, "label": -1}
    if path[-1] == END_REL:
        rel_list = path[:-1]
    else:
        rel_list = path.copy()
    mask_emb = torch.ones(MAX_HOP).to(device)
    if len(rel_list) < MAX_HOP:
        # Expand the path to MAX_HOP, and mask the expanded part
        mask_num = MAX_HOP - len(rel_list)
        mask_emb[-mask_num:] = 0
        rel_list += [""] * mask_num
    word_emb = get_texts_embeddings(word_list, tokenizer, retriever).unsqueeze(0)
    rel_emb = get_texts_embeddings(rel_list, tokenizer, retriever).unsqueeze(0)
    mask_emb = mask_emb.unsqueeze(0)

    logit = discriminator(word_emb, rel_emb, mask_emb)
    logit = F.softmax(logit, dim=-1)
    label = torch.argmax(logit, dim=-1)
    return {"score": logit[0][2].item(), "label": label.item()}


def additional_filter(TOP_K, RESERVED_COUNT):
    dataset = cfg.dataset
    if dataset == "webqsp":
        MAX_HOP = 2
    elif dataset == "cwq":
        MAX_HOP = 4
    else:
        raise NotImplementedError

    if TOP_K <= RESERVED_COUNT:
        return

    threshold = 1 / 3

    input_path = os.path.join(cfg.inference["output_dir"], f"K={TOP_K}.json")
    output_path = os.path.join(cfg.inference["output_dir"], f"K={TOP_K}_dc.json")
    notional_word_path = cfg.inference["input_path"]

    if os.path.exists(output_path):
        return

    retriever_path = cfg.retriever["final_model"]
    discriminator_path = cfg.discriminator["final_model"]

    print("[load model begin]")

    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model["roberta_base"])

    plm = AutoModel.from_pretrained(retriever_path)
    discriminator = Discriminator(plm.config.hidden_size)
    state_dict = torch.load(discriminator_path)
    discriminator.load_state_dict(state_dict)
    for param in plm.parameters():
        param.requires_grad = False
    for param in discriminator.parameters():
        param.requires_grad = False

    plm.eval()
    discriminator.eval()
    plm.to(device)
    discriminator.to(device)

    with open(input_path, "r") as f1, open(notional_word_path, "r") as f2:
        data_list = []
        for line1, line2 in zip(f1, f2):
            data = json.loads(line1)
            data["notional_word"] = json.loads(line2)["notional_word"]
            data_list.append(data)

    with open(output_path, "w") as f:
        for data in tqdm(data_list, desc="discriminate"):
            path_score_list = data["path_with_score"]
            path_score_list = sorted(
                path_score_list, key=lambda x: x["score"], reverse=True
            )
            path_list_new = []

            for i, path_score in enumerate(path_score_list):
                if i < RESERVED_COUNT:
                    path_list_new.append(path_score)
                    continue
                result = discriminate_path(
                    path_score["path"],
                    data["notional_word"],
                    MAX_HOP,
                    tokenizer,
                    plm,
                    discriminator,
                )
                if result["score"] > threshold and result["label"] != 0:
                    path_list_new.append(path_score)

            data["path_with_score"] = path_list_new
            del data["notional_word"]

            f.write(json.dumps(data) + "\n")
