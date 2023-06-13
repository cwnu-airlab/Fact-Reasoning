import argparse 
import json
import logging
import evaluate 
import torch 
import os 
import numpy as np

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)
from datasets import Dataset, DatasetDict

"""
Train the reader model for BoolQ dataset.
The reader model is a sequence classification model on pretrained BERT.
label
- 0 -> No
- 1 -> Yes

Input Fromat

[CLS] question [SEP] context [SEP]
""" 

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


def preprocess_data(data):
    dd = data["data"]
    cleaned_data = []
    for datapoint in dd:
        paras = datapoint["paragraphs"]
        for para in paras:
            context = para["context"]
            qas_list = para["qas"]
            for qas in qas_list:
                question = qas["question"]
                answer = qas["answers"]["text"]
                is_impossible = qas["is_impossible"]
                text = f"{question} [SEP] {context}"
                label = None 
                if answer == "No":
                    label = 0
                else:
                    label = 1
                if not is_impossible:
                    cleaned_data.append({"text":text,"label": label})
    return cleaned_data     

def load_dataset(train_file_path, dev_file_path):
    with open(train_file_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(dev_file_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    train_data_cleaned = preprocess_data(train_data)
    dev_data_cleaned = preprocess_data(dev_data)

    # Number of positive and negative labels 
    num_pos = 0
    num_neg = 0
    for data in train_data_cleaned:
        if data["label"] == 0:
            num_neg += 1
        else:
            num_pos += 1
    logger.info(f"Train data: {num_pos} positive, {num_neg} negative")

    train_dataset  = Dataset.from_list(train_data_cleaned)
    dev_dataset = Dataset.from_list(dev_data_cleaned)

    dataset = DatasetDict({"train":train_dataset, "validation":dev_dataset})
    return dataset



def main(args):
    # Load dataset 
    

    logger.info("Loading korean boolQA dataset..")
    dataset = load_dataset(args.train_file_path, args.dev_file_path)
    logger.info("Dataset loaded")
    logger.info(f"Train: {len(dataset['train'])}, Dev: {len(dataset['validation'])}")
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

    # Dataset statistic
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    logger.info(f"Loading model: {args.model_name_or_path}")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=2)

    logger.info("Tokenizing dataset..")
    tokenized_KBQA = dataset.map(preprocess_function, batched=True)

    metric_accuracy = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
        f1 = metric_f1.compute(predictions=predictions, references=labels)
        return {"accuracy":accuracy, "f1":f1}


    num_gpu_available = torch.cuda.device_count()
    num_workers = int(4 * num_gpu_available)
    logger.info(f"Let's use {num_gpu_available} GPUs!")

    logger.info("Start training..")

    local_rank = int(os.environ["LOCAL_RANK"])
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        seed=args.seed,
        load_best_model_at_end=True,
        dataloader_num_workers=num_workers,
        local_rank=local_rank,
        ddp_find_unused_parameters=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_KBQA["train"],
        eval_dataset=tokenized_KBQA["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", type=str, default="/home/deokhk/project/CBNU_implicit/dataset/korean_boolQA/TL_text_entailment.json")
    parser.add_argument("--dev_file_path", type=str, default="/home/deokhk/project/CBNU_implicit/dataset/korean_boolQA/VL_text_entailment.json")
    parser.add_argument("--output_dir", type=str, default="/home/deokhk/project/CBNU_trained_models/kor_BoolQA")

    # Add training args 
    parser.add_argument("--model_name_or_path", type=str, default="klue/roberta-base")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Other arguments
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()
    main(args)