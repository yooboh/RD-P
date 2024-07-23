import heapq
import json
import os
import random
from ast import List, Set

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import cfg
from knowledge_graph.knowledge_graph_cache import KnowledgeGraphCache

if cfg.dataset == "webqsp":
    MAX_HOP = 2
    WORD_NUM = 2
    ZERO_SAMPLE_NUM = 10
    PARTIAL_SAMPLE_NUM = 10
    partial_threshold = 50
elif cfg.dataset == "cwq":
    MAX_HOP = 4
    WORD_NUM = 3
    ZERO_SAMPLE_NUM = 10
    PARTIAL_SAMPLE_NUM = 5
    partial_threshold = 80
else:
    raise NotImplementedError

END_REL = "END OF REASONING PATH"
device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
random.seed(123)

filtered_rels = {
    "common.topic.webpage",
    "common.topic.image",
    "freebase.type_profile.equivalent_topic",
}


def random_walk_sampling(
    kg: KnowledgeGraphCache,
    heads: Set,
    excluded_rels: Set,
    current_path: tuple,
    sample_num: int,
    filtered_paths: Set,
    hop,
    termination_prob=0.35,
) -> Set:
    paths = set()
    max_try = sample_num * 5

    while sample_num and max_try:
        # path = current_path.copy()
        path = current_path
        head_list = list(heads)
        for _ in range(hop):
            head = random.choice(head_list)
            rels = list(kg.get_relation(head) - excluded_rels)
            if not rels:
                break
            rel = random.choice(rels)
            # path.append(rel)
            path = path + (rel,)
            if random.random() <= termination_prob:
                break
            head_list = list(kg.get_tail(head, rel))
        if path and path not in paths and path not in filtered_paths:
            paths.add(path)
            sample_num -= 1
        max_try -= 1
    return paths


def search_from_kg(kg: KnowledgeGraphCache, topic_entities, path_score_list) -> List:
    extended_path_score_list = []
    excluded_rels = set()
    sampled_paths = set()
    prefix_dict = dict()

    gold_paths = {
        tuple(path_score[0]) for path_score in path_score_list if path_score[1] == 2
    }

    all_paths = {tuple(path_score[0]) for path_score in path_score_list}

    for path in gold_paths:
        excluded_rels.update(path)
        for index in range(1, len(path)):
            prefix = tuple(path[:index])
            if prefix not in prefix_dict:
                prefix_dict[prefix] = {path[index]}
            else:
                prefix_dict[prefix].add(path[index])

    for entity in topic_entities:
        sampled_paths.update(
            random_walk_sampling(
                kg,
                {entity},
                excluded_rels,
                current_path=tuple(),
                sample_num=min(
                    int(ZERO_SAMPLE_NUM * (0.4 * len(gold_paths) + 0.6)), 50
                ),
                filtered_paths=all_paths,
                hop=MAX_HOP,
            )
        )
    extended_path_score_list.extend([[list(path), 0] for path in sampled_paths])

    sampled_paths.clear()
    excluded_rels.clear()

    for path in gold_paths:
        length = len(path)
        for entity in topic_entities:
            # if not kg.judge_src(entity, answers, list(path)):
            #     continue
            heads = kg.get_tail(entity, path[0])
            if not heads:
                continue
            for index in range(1, length):
                sampled_paths.add(tuple(path[:index]))
                sampled_paths.update(
                    random_walk_sampling(
                        kg,
                        heads,
                        excluded_rels=prefix_dict[tuple(path[:index])],
                        current_path=path[:index],
                        sample_num=PARTIAL_SAMPLE_NUM,
                        filtered_paths=all_paths,
                        hop=length - index,
                    )
                )
                tails = set()
                for head in heads:
                    tails.update(kg.get_tail(head, path[index]))
                if not tails:
                    break
                else:
                    heads = tails

    if len(sampled_paths) > partial_threshold:
        sampled_paths = random.sample(sampled_paths, partial_threshold)
    extended_path_score_list.extend([[list(path), 1] for path in sampled_paths])
    return extended_path_score_list


@torch.no_grad()
def get_texts_embeddings(texts, tokenizer, model):
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=cfg.MAX_LEN,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    embeddings = model(**inputs, return_dict=True).pooler_output
    return embeddings


@torch.no_grad()
def search_candidate_path(kg, candidate_paths, nums, max_hop, tokenizer, model):
    rel_score_list = []
    for index in range(len(candidate_paths)):
        if (
            candidate_paths[index]["path"]
            and candidate_paths[index]["path"][-1] == END_REL
            or len(candidate_paths[index]["path"]) >= max_hop
        ):
            continue
        score = candidate_paths[index]["score"]
        question = candidate_paths[index]["question"]
        rels = candidate_paths[index]["rels"]
        r_emb = get_texts_embeddings(rels, tokenizer, model)
        q_emb = get_texts_embeddings([question], tokenizer, model).expand_as(r_emb)

        new_scores = F.cosine_similarity(q_emb, r_emb, dim=-1) * score
        rel_score_list.extend(
            [[index, rel, new_score.item()] for rel, new_score in zip(rels, new_scores)]
        )
    topn = heapq.nlargest(nums, rel_score_list, key=lambda x: x[2])

    candidate_paths_new = []
    for index, rel, score in topn:
        if rel != END_REL:
            question_new = " ".join([candidate_paths[index]["question"], rel, "#"])
            heads_new = set.union(
                *[kg.get_tail(head, rel) for head in candidate_paths[index]["heads"]]
            )
            rels_new = set.union(*[kg.get_relation(head) for head in heads_new])
            rels_new = list(rels_new) + [END_REL]
            path_new = candidate_paths[index]["path"] + [rel]
            candidate_paths_new.append(
                {
                    "question": question_new,
                    "rels": rels_new,
                    "entity": candidate_paths[index]["entity"],
                    "heads": heads_new,
                    "path": path_new,
                    "score": score,
                }
            )
        else:
            candidate_paths_new.append(
                {
                    "path": candidate_paths[index]["path"] + [rel],
                    "entity": candidate_paths[index]["entity"],
                    "score": score,
                }
            )
    return candidate_paths_new


def hard_neg_sampling(
    kg: KnowledgeGraphCache,
    question,
    topic_entities,
    topic_entity_names,
    answers,
    filtered_list,
    tokenizer,
    retriever,
) -> List:
    TOP_K = 5
    HARD_NEG_NUM = 2

    final_path_list = []

    candidate_paths = []
    path_score_list = []

    for entity, entity_name in zip(topic_entities, topic_entity_names):
        rels = list(kg.get_relation(entity))
        if not rels:
            continue

        question = " ".join([question, "[SEP]", entity_name, "→"])
        question = question.replace("[SEP]", tokenizer.sep_token)
        candidate_paths.append(
            {
                "question": question,
                "rels": rels,
                "path": [],
                "entity": entity,
                "heads": {entity},
                "score": 1,
            }
        )

    nums = TOP_K
    counter = 0

    while counter < MAX_HOP and nums:
        candidate_paths = search_candidate_path(
            kg, candidate_paths, nums, MAX_HOP, tokenizer, retriever
        )
        for candidate in candidate_paths:
            if candidate["path"][-1] == END_REL or len(candidate["path"]) >= MAX_HOP:
                nums -= 1
                if candidate["path"][-1] == END_REL:
                    path_final = candidate["path"][:-1]
                else:
                    path_final = candidate["path"]
                path_score_list.append(
                    {
                        "path": path_final,
                        "entity": candidate["entity"],
                        "score": candidate["score"],
                    }
                )
        counter += 1

    for index in range(len(path_score_list)):
        path = path_score_list[index]["path"]
        ent = path_score_list[index]["entity"]
        if not path:
            continue
        elif path in filtered_list:
            final_path_list.extend([[path, 2]] * (HARD_NEG_NUM - 1))
            continue
        else:
            shortest_index = kg.get_shortest_index(ent, path, answers)
            if shortest_index != -1:
                final_path_list.extend([[path[:shortest_index], 2]] * HARD_NEG_NUM)
            elif any(any(elem in _path for _path in filtered_list) for elem in path):
                final_path_list.extend([[path, 1]] * HARD_NEG_NUM)
            else:
                final_path_list.extend([[path, 0]] * HARD_NEG_NUM)

    return final_path_list


def data_padding(sampled_path_score_list, notional_word):
    path_score_word_list = []
    for path_score in sampled_path_score_list:
        path = path_score[0]
        if len(path) < MAX_HOP:
            path.extend([""] * (MAX_HOP - len(path)))
        # while len(path) < MAX_HOP:
        #     path.extend(random.sample(path, min(MAX_HOP - len(path), len(path))))
        tmp_list = path + [path_score[1]]
        path_score_word_list.append(notional_word + tmp_list)
    return path_score_word_list


def get_discriminator_train_data():
    input_path = cfg.tmp_dir + "path.json"
    notional_word_path = cfg.processed_dataset + "train.json"
    output_dir = cfg.train_dir
    output_path = cfg.discriminator["train"]["input_path"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_path):
        return

    with open(input_path, "r") as f1, open(notional_word_path, "r") as f2:
        data_list = []
        for line1, line2 in zip(f1, f2):
            qa_path_score = json.loads(line1)
            qa_path_score["notional_word"] = json.loads(line2)["notional_word"]
            data_list.append(qa_path_score)

    kg = KnowledgeGraphCache()

    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model["roberta_base"])
    retriever = AutoModel.from_pretrained(cfg.retriever["final_model"])
    for param in retriever.parameters():
        param.requires_grad = False
    retriever = retriever.to(device)
    retriever.eval()

    column_names = (
        [f"word{i}" for i in range(WORD_NUM)]
        + [f"rel{i}" for i in range(MAX_HOP)]
        + ["label"]
    )
    pd.DataFrame(columns=column_names).to_csv(output_path, header=True, index=False)

    threshold = 0.1

    for qa_path_score in tqdm(data_list, desc="random-sampling"):
        path_and_score_list = qa_path_score["path_and_score"]
        if not path_and_score_list:
            continue
        topic_entities = qa_path_score["topic_entities"]
        topic_entity_names = qa_path_score["topic_entity_names"]
        notional_word = qa_path_score["notional_word"]
        question = qa_path_score["question"]
        answers = {item["kb_id"] for item in qa_path_score["answers"]}

        max_score = path_and_score_list[0]["score"]

        filtered_list = [
            path_and_score["path"]
            for path_and_score in path_and_score_list
            if path_and_score["path"]
            and (
                path_and_score["score"] > threshold
                or path_and_score["score"] >= max_score
            )
            and not any(rel in filtered_rels for rel in path_and_score["path"])
        ]

        if not filtered_list or not topic_entities or not notional_word or not answers:
            continue

        filtered_list = filtered_list[:25]

        hard_neg_paths = hard_neg_sampling(
            kg,
            question,
            topic_entities,
            topic_entity_names,
            answers,
            filtered_list,
            tokenizer,
            retriever,
        )

        sampled_path_score_list = [[path, 2] for path in filtered_list]
        if hard_neg_paths:
            sampled_path_score_list.extend(hard_neg_paths)

        random_sampled_list = search_from_kg(
            kg, topic_entities, sampled_path_score_list
        )
        if random_sampled_list:
            sampled_path_score_list.extend(random_sampled_list)

        sampled_path_score_list = data_padding(sampled_path_score_list, notional_word)
        df = pd.DataFrame(sampled_path_score_list)
        df.to_csv(output_path, mode="a", header=False, index=False)
