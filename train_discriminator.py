import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from config import cfg
from discriminator.model import Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))


class CustomTrainingArgs:
    def __init__(
        self,
        output_dir,
        learning_rate,
        per_device_train_batch_size,
        num_train_epochs,
        save_interval,
    ):
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.num_train_epochs = num_train_epochs
        self.save_interval = save_interval


class CustomDataset(Dataset):
    def __init__(self, data_dict, tokenizer):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        dict = {}
        for key in self.keys:
            if key == "label":
                dict[key] = self.data_dict[key][index]
            else:
                # dict[key] = {
                #     "input_ids": self.data_dict[key]["input_ids"][index],
                #     "attention_mask": self.data_dict[key]["attention_mask"][index],
                # }
                tokens = self.tokenizer(
                    self.data_dict[key][index],
                    padding="max_length",
                    truncation=True,
                    max_length=cfg.MAX_LEN,
                    return_tensors="pt",
                )
                dict[key] = {
                    "input_ids": tokens["input_ids"][0],
                    "attention_mask": tokens["attention_mask"][0],
                }
        return dict

    def __len__(self):
        return len(self.data_dict["label"])


def main():
    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M")

    dataset = cfg.dataset
    if dataset == "webqsp":
        training_args = CustomTrainingArgs(
            output_dir=cfg.discriminator["train"]["output_dir"] + time_string + "/",
            learning_rate=5e-4,
            per_device_train_batch_size=64,
            num_train_epochs=4,
            save_interval=2000,
        )
        val_flag = False
    elif dataset == "cwq":
        training_args = CustomTrainingArgs(
            output_dir=cfg.discriminator["train"]["output_dir"] + time_string + "/",
            learning_rate=2e-4,
            per_device_train_batch_size=64,
            num_train_epochs=1,
            save_interval=4000,
        )
        val_flag = False

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model["roberta_base"])
    train_file = cfg.discriminator["train"]["input_path"]

    log_file = training_args.output_dir + "/train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_file,
        filemode="w",
    )

    df = pd.read_csv(train_file, na_filter=False)

    if val_flag:
        split_point = int(len(df) * 0.9)
        train_df = df[:split_point]
        val_df = df[split_point:]
        val_data_dict = val_df.to_dict(orient="list")

        del val_df

        val_dataset = CustomDataset(val_data_dict, tokenizer)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            num_workers=8,
        )

        best_val_loss = np.inf
        patience = 3
        counter = 0
    else:
        train_df = df

    train_data_dict = train_df.to_dict(orient="list")

    del df, train_df

    word_num, rel_num = 0, 0
    for key in train_data_dict.keys():
        if key == "label":
            continue
        if key.startswith("word"):
            word_num += 1
        elif key.startswith("rel"):
            rel_num += 1
        else:
            raise ValueError(f"Invalid key: {key}")

    train_dataset = CustomDataset(train_data_dict, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        num_workers=8,
    )

    plm = AutoModel.from_pretrained(cfg.retriever["final_model"])
    model = Discriminator(plm.config.hidden_size, num_labels=3)
    plm.eval()
    for param in plm.parameters():
        param.requires_grad = False
    plm = plm.to(device)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    total_steps = training_args.num_train_epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps * 0.1,
        num_training_steps=total_steps,
    )

    print("Start training")
    print(f"Number of epochs: {training_args.num_train_epochs}")
    print(f"Number of training steps: {total_steps}")
    print(f"Number of batches per epoch: {len(train_dataloader)}")
    progress_bar = tqdm(range(total_steps), desc="Training", disable=False)
    current_steps = 0

    for epoch in range(training_args.num_train_epochs):
        # Training phase
        model.train()

        total_loss = 0
        recent_loss = 0
        num_batchs = 0
        for i, batch in enumerate(train_dataloader):
            word_list = []
            rel_list = []
            mask_list = []
            labels = batch["label"].to(device)

            """
                dim=0: the output shape is [num, batch_size, hidden_dim]. It returns num tensors of shape [batch_size, hidden_dim].
                dim=1: the output shape is [batch_size, num, hidden_dim]. It returns batch_size tensors of shape [num, hidden_dim].
            """
            for index in range(word_num):
                word = {k: v.to(device) for k, v in batch[f"word{index}"].items()}
                word_list.append(plm(**word, return_dict=True).pooler_output)
            word_emb = torch.stack(word_list, dim=1)

            for index in range(rel_num):
                rel = {k: v.to(device) for k, v in batch[f"rel{index}"].items()}
                rel_list.append(plm(**rel, return_dict=True).pooler_output)
                # Check if the attention_mask has only the first two as 1, if so, the mask is 0, otherwise it is 1
                mask = (rel["attention_mask"][:, :3].sum(dim=1) > 2).float().to(device)
                mask_list.append(mask)
            mask_emb = torch.stack(mask_list, dim=1)
            rel_emb = torch.stack(rel_list, dim=1)

            logits = model(word_emb, rel_emb, mask_emb)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            recent_loss += loss.item()
            num_batchs += 1
            current_steps += 1

            if i and i % 100 == 0:
                logging.info(
                    f"Iteration {i}. Average loss for the last 100 iterations: {recent_loss / 100}"
                )
                recent_loss = 0

            # backpropagation
            loss.backward()
            # update parameters
            optimizer.step()
            # update learning rate
            scheduler.step()
            # clear gradients
            optimizer.zero_grad()
            # update progress bar
            progress_bar.update(1)

            if current_steps % training_args.save_interval == 0:
                checkpoint_dir = (
                    training_args.output_dir + f"checkpoint-{current_steps}/"
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_dir + "discriminator.pt")
                if val_flag:
                    # Validation phase
                    model.eval()
                    total_preds = []
                    total_labels = []
                    val_loss = 0
                    num_batchs_eval = 0
                    with torch.no_grad():
                        for i, batch in enumerate(val_dataloader):
                            word_list = []
                            rel_list = []
                            labels = batch["label"].to(device)
                            for index in range(word_num):
                                word = {
                                    k: v.to(device)
                                    for k, v in batch[f"word{index}"].items()
                                }
                                word_list.append(
                                    plm(**word, return_dict=True).pooler_output
                                )
                            word_emb = torch.stack(word_list, dim=1)
                            mask_list = []
                            for index in range(rel_num):
                                rel = {
                                    k: v.to(device)
                                    for k, v in batch[f"rel{index}"].items()
                                }
                                rel_list.append(
                                    plm(**rel, return_dict=True).pooler_output
                                )
                                mask = (
                                    (rel["attention_mask"][:, :3].sum(dim=1) > 2)
                                    .float()
                                    .to(device)
                                )
                                mask_list.append(mask)
                            mask_emb = torch.stack(mask_list, dim=1)
                            rel_emb = torch.stack(rel_list, dim=1)

                            logits = model(word_emb, rel_emb, mask_emb)
                            loss = F.cross_entropy(logits, labels)
                            val_loss += loss.item()
                            num_batchs_eval += 1

                            preds = torch.argmax(logits, dim=1)
                            total_preds.extend(preds.cpu().numpy())
                            total_labels.extend(labels.cpu().numpy())

                    avg_loss_eval = val_loss / num_batchs_eval

                    label_index = 2
                    precision = precision_score(total_labels, total_preds, average=None)
                    recall = recall_score(total_labels, total_preds, average=None)

                    logging.info(
                        f"Current Iteration {current_steps}. Validation Loss: {avg_loss_eval}"
                    )
                    logging.info(
                        f"Precision: {precision[label_index]}, Recall: {recall[label_index]}"
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        counter = 0
                    else:
                        counter += 1
                        if counter >= patience:
                            print("Early stopping")
                            return
                    model.train()

        avg_loss = total_loss / num_batchs
        # print(f"Epoch {epoch+1}, Loss: {avg_loss}, num_batchs: {num_batchs}")
        logging.info(f"Epoch {epoch+1}, Loss: {avg_loss}")
        # if epoch < training_args.num_train_epochs - 1:
        #     output_dir = training_args.output_dir + f"epoch_{epoch+1}/"
        # else:
        #     output_dir = training_args.output_dir + "final_model/"
        # os.makedirs(output_dir, exist_ok=True)
        if epoch == training_args.num_train_epochs - 1:
            output_dir = training_args.output_dir + "final_model/"
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), output_dir + "discriminator.pt")

        # if val_flag:
        #     # Validation phase
        #     model.eval()
        #     total_preds = []
        #     total_labels = []
        #     val_loss = 0
        #     num_batchs = 0
        #     with torch.no_grad():
        #         for i, batch in enumerate(val_dataloader):
        #             word_list = []
        #             rel_list = []
        #             labels = batch["label"].to(device)
        #             for index in range(word_num):
        #                 word = {
        #                     k: v.to(device) for k, v in batch[f"word{index}"].items()
        #                 }
        #                 word_list.append(plm(**word, return_dict=True).pooler_output)
        #             word_emb = torch.stack(word_list, dim=1)
        #             mask_list = []
        #             for index in range(rel_num):
        #                 rel = {k: v.to(device) for k, v in batch[f"rel{index}"].items()}
        #                 rel_list.append(plm(**rel, return_dict=True).pooler_output)
        #                 mask = (
        #                     (rel["attention_mask"][:, :3].sum(dim=1) > 2)
        #                     .float()
        #                     .to(device)
        #                 )
        #                 mask_list.append(mask)
        #             mask_emb = torch.stack(mask_list, dim=1)
        #             rel_emb = torch.stack(rel_list, dim=1)

        #             logits = model(word_emb, rel_emb, mask_emb)
        #             loss = F.cross_entropy(logits, labels, weight=label_weights)
        #             val_loss += loss.item()
        #             num_batchs += 1

        #             preds = torch.argmax(logits, dim=1)
        #             total_preds.extend(preds.cpu().numpy())
        #             total_labels.extend(labels.cpu().numpy())

        #     avg_loss = val_loss / num_batchs

        #     label_index = 2
        #     precision = precision_score(total_labels, total_preds, average=None)
        #     recall = recall_score(total_labels, total_preds, average=None)

        #     logging.info(f"Epoch {epoch+1}, Validation Loss: {avg_loss}")
        #     logging.info(
        #         f"Precision: {precision[label_index]}, Recall: {recall[label_index]}"
        #     )
        #     if val_loss < best_val_loss:
        #         best_val_loss = val_loss
        #         counter = 0
        #     else:
        #         counter += 1
        #         if counter >= patience:
        #             print("Early stopping")
        #             break

    print("Training finished")


if __name__ == "__main__":
    main()
