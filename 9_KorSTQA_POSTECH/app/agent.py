import argparse 
import logging
import os 
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import json 
import torch 
import sys 
from transformers import (
    BartForConditionalGeneration, 
    PreTrainedTokenizerFast,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from parser.argument_parser import default_parser,json_to_argv

from retriever.semantic_search_retriever import SemanticRetriever
from reader.reader_util import extract_integer_numbers, ends_with_jong, replace_str
from util import load_korean_passage
from typing import List 

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

torch.manual_seed(42)
#random.seed(42)
#np.random.seed(42)

class Service:
    task = [
        {
            'name': "implicit_question_answering",
            'description': 'Question & Answering module for implicit multi-hop reasoning'
        }
    ]

    def __init__(self):
        self.qa_pipeline = STPipeLine() 
        self.qa_pipeline.load_models()

    @classmethod
    def get_task_list(cls):
        return json.dumps(cls.task), 200
    
    def do(self, content):
        question = content["question"]
        ret = self.qa_pipeline(question)
        try:
            question = content["question"]
            ret = self.qa_pipeline(question)
            if 'error' in ret.keys():
                return json.dumps(ret), 400
            return json.dumps(ret), 200
        except Exception as e:
            return json.dumps(
                {
                    'error': "{}".format(e)
                }
            ), 400


class STPipeLine(object):
    def __init__(self):

        # Initialize argument parser for initializing parameters 
        parser = default_parser()
        argv = json_to_argv("predict.json")
        args = parser.parse_args(argv)

        self.decomposition_model_path = args.decomposition_model_path
        self.reader_model_path = args.reader_model_path
        self.final_boolqa_model_path = args.final_boolqa_model_path
        self.retriever_model_path = args.pretrained_embedding_model_name 

        self.passage_path = args.passage_path 

        self.decomposition_model_device = args.decomp_model_device
        self.reader_model_device = args.reader_model_device
        self.final_reader_model_device = args.final_reader_model_device 
        self.comparision_operator = ["먼저인가", "나중인가", "동시에인가", "같은가", "다른가"]

    def load_models(self):
        """
        Load models
        """
        logger.info("Loading retriever...")
        passages, doc_to_passages, _ = load_korean_passage(self.passage_path)
        self.retriever = SemanticRetriever(passages, self.retriever_model_path)
        self.doc_to_passages = doc_to_passages

        self.retriever.build_corpus_embedding()


        logger.info("Loading decomposition model...")
        self.decomposition_tokenizer = PreTrainedTokenizerFast.from_pretrained('hyunwoongko/kobart')
        self.decomposition_tokenizer.add_tokens(['#1', '#2', '#3', '#4', '#5'], special_tokens=True)

        self.decomposition_model = BartForConditionalGeneration.from_pretrained(self.decomposition_model_path)
        self.decomposition_model.to(self.decomposition_model_device)

        self.final_reader_tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
        self.final_reader_model = AutoModelForSequenceClassification.from_pretrained(self.final_boolqa_model_path, num_labels=2)

        logger.info("Loading reader model..")
        self.reader_model_pipe = pipeline('question-answering', model=self.reader_model_path) 
        self.final_reader_pipe = pipeline('text-classification', model=self.final_reader_model, tokenizer=self.final_reader_tokenizer)


    def decompose_question(self, question)->List:
        """
        Decompose question
        """
        input_ids = self.decomposition_tokenizer.encode(question, return_tensors='pt')
        input_ids = input_ids.to(self.decomposition_model_device)
        generated_ids = self.decomposition_model.generate(
            input_ids=input_ids,
            penalty_alpha=0.6,
            top_k=4,
            max_length=128,
        )
        generated_ids = generated_ids[0].cpu().tolist()
        decomposed_question_str = self.decomposition_tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        # Postprocessing 
        for r in (("<s>", ""), ("</s>", ""), ("<pad>", ""), ("<unk>", "")):
            decomposed_question_str = decomposed_question_str.replace(*r)
        
        splitted = decomposed_question_str.split("?")
        
        decomposed_questions = [(x+"?").strip() for x in splitted if x != ""]

        return decomposed_questions
    
    def retrieve_passage(self, question):
        """
        Retrieve passages.
        Given question, return the most similar passages
        """
        top_1_res = self.retriever.query_index(question, 1)
        top_1_psg = self.doc_to_passages[top_1_res[0]]

        return top_1_psg

    @staticmethod
    def fill_answer_placeholder(decomp_question, subquestion_answers):
        """
        Fill the answer placeholder in the decomposed question
        Here, we consider josa.
        We replace josa either "은/는" or "이/가" depending on the last latter of the answer
        """
        for idx, placeholder in enumerate(['#1', '#2', '#3', '#4', '#5']):
            if placeholder in decomp_question and len(subquestion_answers) > idx:
                # First replace the josa
                placeholder_idx = decomp_question.find(placeholder)
                if placeholder_idx+2 <= len(decomp_question):
                    # Position of the josa 
                    cur_josa = decomp_question[placeholder_idx+2]
                    if cur_josa == "은" or cur_josa == "는":
                        if ends_with_jong(subquestion_answers[idx]):
                            decomp_question = replace_str(decomp_question, placeholder_idx+2, "은")
                        else:
                            decomp_question = replace_str(decomp_question, placeholder_idx+2, "는")
                    elif cur_josa == "이" or cur_josa == "가":
                        if ends_with_jong(subquestion_answers[idx]):
                            decomp_question = replace_str(decomp_question, placeholder_idx+2, "이")
                        else:
                            decomp_question = replace_str(decomp_question, placeholder_idx+2, "가")
                decomp_question = decomp_question.replace(placeholder, subquestion_answers[idx])
        return decomp_question

    def __call__(self, question):
        # 1. 질문 분해
        decomposed_questions = self.decompose_question(question)

        # 2. 패시지 검색 & QA 수행 
        qap_list = []
        subquestion_answers = []
        subsequent_passages = []
        for idx, decomp_question in enumerate(decomposed_questions):
            decomp_question = self.fill_answer_placeholder(decomp_question, subquestion_answers)
            if idx == len(decomposed_questions)-1:
                # Final questions. Should be answered yes or no! 
                # context is concatenated passages
                # If it asks for decomposing numbers.. then we extract number from the string and just compare those
                extracted_numbers = extract_integer_numbers(decomp_question)
                comparison_operator_in_question = [True for x in self.comparision_operator if x in decomp_question] 

                if len(extracted_numbers) >=2 and (True in comparison_operator_in_question):
                    # If it asks for decomposing numbers.. then we extract number from the string and just compare those
                    if "먼저인가" in decomp_question:
                        comp_1 = extracted_numbers[0]
                        comp_2 = extracted_numbers[1]

                        if comp_1 < comp_2:
                            predicted_answer = "예"
                        else:
                            predicted_answer = "아니요" 
                    elif "나중인가" in decomp_question:
                        comp_1 = extracted_numbers[0]
                        comp_2 = extracted_numbers[1]

                        if comp_1 > comp_2:
                            predicted_answer = "예"
                        else:
                            predicted_answer = "아니요" 
                    elif "동시에인가" in decomp_question or "같은가" in decomp_question:
                        comp_1 = extracted_numbers[0]
                        comp_2 = extracted_numbers[1]

                        if comp_1 == comp_2:
                            predicted_answer = "예"
                        else:
                            predicted_answer = "아니요"
                    elif "다른가" in decomp_question:
                        comp_1 = extracted_numbers[0]
                        comp_2 = extracted_numbers[1]

                        if comp_1 != comp_2:
                            predicted_answer = ""
                        else:
                            predicted_answer = "아니요"
                    else:
                        pass    
                    passage_context = " ".join(subsequent_passages)
                else:
                    top_1_psg = self.retrieve_passage(decomp_question)
                    passage_context = top_1_psg["content"]
                    """
                    if subsequent_passages != []:
                        passage_context = passage_context + " " + " ".join(subsequent_passages)
                    For denoise, we don't use subsequent passages
                    """
                    inputs = f"{decomp_question} [SEP] {passage_context}"
                    res = self.final_reader_pipe(inputs)
                    label = res[0]["label"]
                    if label == "LABEL_1":
                        predicted_answer = "예"
                    else:
                        predicted_answer = "아니요"
                    subquestion_answers.append(predicted_answer)
            else:
                top_1_psg = self.retrieve_passage(decomp_question)
                passage_context = top_1_psg["content"]
                # Answer questions 
                subsequent_passages.append(passage_context)
                predicted_answer_dict = self.reader_model_pipe(question=decomp_question, context=passage_context)
                predicted_answer = predicted_answer_dict["answer"]
                subquestion_answers.append(predicted_answer)

            qap_list.append({"decomposed question": decomp_question, "answer": predicted_answer, "passage": passage_context})

        final_answer = predicted_answer
        output_dict = {
            "question": question,
            "final_answer":final_answer,
            "qap_list": qap_list
        }
        return output_dict
    
if __name__=='__main__':
    
    data = { "question": "조지 워싱턴 자신의 연설을 CD에 라이브로 녹음할 수 있었습니까?" }

    api_service = Service()
    predict, status = api_service.do(data)
    print(json.loads(predict))
