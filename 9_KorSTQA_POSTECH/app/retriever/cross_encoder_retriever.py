import argparse 
import math
import json 
import logging
import os
import sys 

from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder import CrossEncoder
from korean_qa.retriever.semantic_search_retriever import evaluate_retriever, load_retriever_test_dataset
from tqdm import tqdm 
from typing import List 

class CrossEncoderRetriever(object):
    def __init__(self, passages_list:List, pretrained_embedding_model_name_or_path:str):
        self.passages_list = passages_list

        print("Loading corpus..")
        self.corpus = [doc["content"] for doc in self.passages_list]
        print("Loaded corpus")

        print(f"Loading pretrained cross encoder model {pretrained_embedding_model_name_or_path}")
        self.cross_encoder =  CrossEncoder(pretrained_embedding_model_name_or_path, device="cuda:0")
        print("Loaded the model")

    
    def query_index(self, query:str, top_k:int):
        """ 
        query embedding 과 유사한  top_k document를 return한다.
        """
        q_p_list = [[query, passage] for passage in self.corpus]
        scores = self.cross_encoder.predict(q_p_list)
        score_list = scores.tolist()
        passage_with_similarity_tuples = []
        for passage, similarity in zip(self.corpus, score_list):
            passage_with_similarity_tuples.append((passage, similarity))

        passage_with_similarity_tuples.sort(key=lambda x: x[1],reverse=True)

        matched_top_k_passages = [x[0] for x in passage_with_similarity_tuples][:top_k]

        return matched_top_k_passages



def main(args):
    with open(args.passage_path, 'r') as f: 
        passage_list = json.load(f)
    
    doc_to_document_passages = dict()
    doc_title_to_context = dict()
    mypassage_list = []
    for passage_title, passage_content in passage_list.items(): 
        passage_content["title"] = passage_title 
        content = passage_content["content"]
        doc_to_document_passages[content] = passage_content 
        doc_title_to_context[passage_title] = content
        mypassage_list.append(passage_content)

    retriever = CrossEncoderRetriever(mypassage_list, args.pretrained_embedding_model_name_or_path)

    retriever_test_data = load_retriever_test_dataset(doc_title_to_context)
    evaluate_retriever(retriever, retriever_test_data, doc_to_document_passages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_embedding_model_name_or_path", default="/home/deokhk/project/CBNU_trained_models/kor_reranker", help="Path to the pretrained cross-encoder model")
    parser.add_argument("--passage_path", help="Path to the passages to be retrieved", default="/home/deokhk/project/CBNU_implicit/dataset/strategyqa/kor_translated/strategyqa_train_paras_kor.json")

    args = parser.parse_args()
    main(args)