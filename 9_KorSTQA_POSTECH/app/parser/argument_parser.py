# coding=utf-8
#/usr/bin/env python3
import os
import sys
import argparse
import torch
import json
import logging
import random
import numpy as np

from os.path import join

logger = logging.getLogger(__name__)


def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def json_to_argv(json_file):
    j = json.load(open(json_file))
    argv = []
    for k, v in j.items():
        new_v = str(v) if v is not None else None
        argv.extend(['--' + k, new_v])
    return argv

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomposition_model_path", default="/home/deokhk/project/CBNU_trained_models/kor_QD/kobart/best_checkpoint")
    parser.add_argument("--reader_model_path", default="monologg/koelectra-base-v3-finetuned-korquad")
    parser.add_argument("--final_boolqa_model_path", default="/home/deokhk/project/CBNU_trained_models/kor_BoolQA/checkpoint-3755/")

    # Retriever-related arguments 
    parser.add_argument("--passage_path", help="Path to the passages to be retrieved", default="/home/deokhk/project/CBNU_implicit/dataset/strategyqa/kor_translated/strategyqa_train_paras_kor.json")
    parser.add_argument("--pretrained_embedding_model_name", help="Name of the pretrained embedding model", default="distiluse-base-multilingual-cased-v1")

    # Other options 
    parser.add_argument("--decomp_model_device", default=0)
    parser.add_argument("--reader_model_device", default=0)
    parser.add_argument("--final_reader_model_device", default=0)

    return parser 