# 导入所需的库
import argparse
import os

import torch
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np

from imdb_dataset import load_imdb_data

import warnings
warnings.filterwarnings('ignore')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--expn", default=0, type=int)
    parser.add_argument("--dataset_path", default="datasets/aclImdb", type=str)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--save_path", default="logs/imdb", type=str)

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    return parser


def main():

    args = get_parser().parse_args()

    # 加载IMDb数据集
    train_texts, train_labels, test_texts, test_labels = load_imdb_data(args.dataset_path)

    print("load imdb ok")

    if args.train:
        train(train_texts, train_labels, args)
    elif args.test:
        test(test_texts, test_labels, args)
    else:
        train(train_texts, train_labels, args)
        test(test_texts, test_labels, args)


def train(reviews, labels, args):

    if args.expn:
        reviews = reviews[:args.expn]
        labels = labels[:args.expn]

    # 初始化BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    print("Saving tokenizer")
    tokenizer.save_pretrained(args.save_path+"/tokenizer")

    # 将文本转换为令牌，并进行截断和填充
    max_length = args.max_length  # 128
    input_ids = []
    attention_masks = []

    for review in tqdm(reviews):
        encoded_dict = tokenizer.encode_plus(
            review,  # 输入文本
            add_special_tokens=True,  # 添加特殊令牌
            max_length=max_length,  # 截断文本
            pad_to_max_length=True,  # 填充文本
            return_attention_mask=True,  # 创建 attention masks
            return_tensors='pt'  # 返回 PyTorch tensors 格式
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    print("tokenizer.encode_plus ok")

    # 将输入和标签转换为 PyTorch tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # 创建 TensorDataset 对象
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # 划分训练集和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    batch_size = args.batch_size  # 32

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),  # 随机采样数据进行训练
        batch_size=batch_size
    )

    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),  # 顺序采样数据进行验证
        batch_size=batch_size
    )

    print("dataloader ok")

    # 初始化BertForSequenceClassification模型
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # 使用预训练模型
        num_labels=2,  # 二元分类
        output_attentions=False,
        output_hidden_states=False,
    )

    print("model ok")

    # 定义优化器和学习率调度程序
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,
                      eps=1e-8
                      )

    epochs = args.epochs  # 5
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    print("optimizer ok")

    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    for epoch_i in range(epochs):

        # 训练阶段
        print(f'Epoch {epoch_i + 1}/{epochs} - Training...')

        # 初始化损失和准确度变量
        total_loss = 0.0
        total_acc = 0.0
        total_f1 = 0.0

        model.train()

        # 对数据进行迭代
        for step, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
            # 将输入和标签放到GPU上
            batch = tuple(t.to(device) for t in batch)

            # 解包输入和标签
            b_input_ids, b_input_mask, b_labels = batch

            # 清除之前的梯度
            model.zero_grad()

            # 前向传递
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            """
                outputs: SequenceClassifierOutput
                {
                    loss: [1×1],
                    logits: [n×2]
                }
            """

            # 计算损失和梯度
            loss = outputs.loss
            loss.backward()

            # 更新模型参数
            optimizer.step()

            # 更新学习率
            scheduler.step()

            # 累加损失和准确度
            total_loss += loss.item()
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_acc += accuracy_score(label_ids, np.argmax(logits, axis=1))
            total_f1 += f1_score(label_ids, np.argmax(logits, axis=1))

        # 计算训练损失、准确度和 F1 值
        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_acc = total_acc / len(train_dataloader)
        avg_train_f1 = total_f1 / len(train_dataloader)

        print(f'Training loss: {avg_train_loss:.3f}, accuracy: {avg_train_acc:.3f}, F1: {avg_train_f1:.3f}')

        # 验证阶段
        print("Evaluating...")

        # 初始化损失和准确度变量
        total_val_acc = 0.0
        total_val_f1 = 0.0

        model.eval()

        # 对数据进行迭代
        for batch in validation_dataloader:
            # 将输入和标签放到GPU上
            batch = tuple(t.to(device) for t in batch)

            # 解包输入和标签
            b_input_ids, b_input_mask, b_labels = batch

            # 禁用梯度计算
            with torch.no_grad():
                # 前向传递
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

                # 计算损失和准确度
                logits = outputs.logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                total_val_acc += accuracy_score(label_ids, np.argmax(logits, axis=1))
                total_val_f1 += f1_score(label_ids, np.argmax(logits, axis=1))

        # 计算验证损失、准确度和
        # 计算验证损失、准确度和 F1 值
        avg_val_acc = total_val_acc / len(validation_dataloader)
        avg_val_f1 = total_val_f1 / len(validation_dataloader)

        print(f'Validation, accuracy: {avg_val_acc:.3f}, F1: {avg_val_f1:.3f}')

    print("Saving imdb model")
    model.save_pretrained(args.save_path+"/model")


def test(test_texts, test_labels, args):

    if args.expn:
        test_texts = test_texts[:args.expn]
        test_labels = test_labels[:args.expn]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(args.save_path+"/tokenizer")
    model = BertForSequenceClassification.from_pretrained(args.save_path+"/model").to(device)

    batch_size = args.batch_size

    print("Start Tokenizing")
    test_inputs = []
    for batch_i in tqdm(range(0, len(test_texts), batch_size)):
        batch = test_texts[batch_i:batch_i+batch_size]
        batch_input = tokenizer(batch, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
        batch_input = {k: v.to(device) for k, v in batch_input.items()}
        test_inputs.append(batch_input)

    predictions = []
    print("Start Testing")
    with torch.no_grad():
        # 进行预测
        for batch in tqdm(test_inputs):
            outputs = model(**batch)
            # 获取预测结果
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=-1)
            predictions.append(batch_predictions.detach().cpu().numpy())
    predictions = np.concatenate(predictions)
    test_labels = np.array(test_labels)

    acc = accuracy_score(predictions, test_labels)
    f1 = f1_score(predictions, test_labels)

    print(f'Test, accuracy: {acc:.3f}, F1: {f1:.3f}')


if __name__ == '__main__':
    main()
