import argparse
import json
import os 

from typing import List
from itertools import chain
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize 
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import torch
from tqdm import tqdm 

"""
I don't know why but this one is not working..
(Especially the reranker part.)
"""

def simple_tokenizer(sent):
    return sent.split(" ")

def normalized_tokenizer(text):
    stop_words = "아 휴 아이구 아이쿠 아이고 어 나 우리 저희 따라 의해 을 를 에 의 가 으로 로 에게 뿐이다 의거하여 근거하여 입각하여 기준으로 예하면 예를 들면 예를 들자면 저 소인 소생 저희 지말고 하지마 하지마라 다른 물론 또한 그리고 비길수 없다 해서는 안된다 뿐만 아니라 만이 아니다 만은 아니다 막론하고 관계없이 그치지 않다 그러나 그런데 하지만 든간에 논하지 않다 따지지 않다 설사 비록 더라도 아니면 만 못하다 하는 편이 낫다 불문하고 향하여 향해서 향하다 쪽으로 틈타 이용하여 타다 오르다 제외하고 이 외에 이 밖에 하여야 비로소 한다면 몰라도 외에도 이곳 여기 부터 기점으로 따라서 할 생각이다 하려고하다 이리하여 그리하여 그렇게 함으로써 하지만 일때 할때 앞에서 중에서 보는데서 으로써 로써 까지 해야한다 일것이다 반드시 할줄알다 할수있다 할수있어 임에 틀림없다 한다면 등 등등 제 겨우 단지 다만 할뿐 딩동 댕그 대해서 대하여 대하면 훨씬 얼마나 얼마만큼 얼마큼 남짓 여 얼마간 약간 다소 좀 조금 다수 몇 얼마 지만 하물며 또한 그러나 그렇지만 하지만 이외에도 대해 말하자면 뿐이다 다음에 반대로 반대로 말하자면 이와 반대로 바꾸어서 말하면 바꾸어서 한다면 만약 그렇지않으면 까악 툭 딱 삐걱거리다 보드득 비걱거리다 꽈당 응당 해야한다 에 가서 각 각각 여러분 각종 각자 제각기 하도록하다 와 과 그러므로 그래서 고로 한 까닭에 하기 때문에 거니와 이지만 대하여 관하여 관한 과연 실로 아니나다를가 생각한대로 진짜로 한적이있다 하곤하였다 하 하하 허허 아하 거바 와 오 왜 어째서 무엇때문에 어찌 하겠는가 무슨 어디 어느곳 더군다나 하물며 더욱이는 어느때 언제 야 이봐 어이 여보시오 흐흐 흥 휴 헉헉 헐떡헐떡 영차 여차 어기여차 끙끙 아야 앗 아야 콸콸 졸졸 좍좍 뚝뚝 주룩주룩 솨 우르르 그래도 또 그리고 바꾸어말하면 바꾸어말하자면 혹은 혹시 답다 및 그에 따르는 때가 되어 즉 지든지 설령 가령 하더라도 할지라도 일지라도 지든지 몇 거의 하마터면 인젠 이젠 된바에야 된이상 만큼 어찌됏든 그위에 게다가 점에서 보아 비추어 보아 고려하면 하게될것이다 일것이다 비교적 좀 보다더 비하면 시키다 하게하다 할만하다 의해서 연이서 이어서 잇따라 뒤따라 뒤이어 결국 의지하여 기대여 통하여 자마자 더욱더 불구하고 얼마든지 마음대로 주저하지 않고 곧 즉시 바로 당장 하자마자 밖에 안된다 하면된다 그래 그렇지 요컨대 다시 말하자면 바꿔 말하면 즉 구체적으로 말하자면 시작하여 시초에 이상 허 헉 허걱 바와같이 해도좋다 해도된다 게다가 더구나 하물며 와르르 팍 퍽 펄렁 동안 이래 하고있었다 이었다 에서 로부터 까지 예하면 했어요 해요 함께 같이 더불어 마저 마저도 양자 모두 습니다 가까스로 하려고하다 즈음하여 다른 다른 방면으로 해봐요 습니까 했어요 말할것도 없고 무릎쓰고 개의치않고 하는것만 못하다 하는것이 낫다 매 매번 들 모 어느것 어느 로써 갖고말하자면 어디 어느쪽 어느것 어느해 어느 년도 라 해도 언젠가 어떤것 어느것 저기 저쪽 저것 그때 그럼 그러면 요만한걸 그래 그때 저것만큼 그저 이르기까지 할 줄 안다 할 힘이 있다 너 너희 당신 어찌 설마 차라리 할지언정 할지라도 할망정 할지언정 구토하다 게우다 토하다 메쓰겁다 옆사람 퉤 쳇 의거하여 근거하여 의해 따라 힘입어 그 다음 버금 두번째로 기타 첫번째로 나머지는 그중에서 견지에서 형식으로 쓰여 입장에서 위해서 단지 의해되다 하도록시키다 뿐만아니라 반대로 전후 전자 앞의것 잠시 잠깐 하면서 그렇지만 다음에 그러한즉 그런즉 남들 아무거나 어찌하든지 같다 비슷하다 예컨대 이럴정도로 어떻게 만약 만일 위에서 서술한바와같이 인 듯하다 하지 않는다면 만약에 무엇 무슨 어느 어떤 아래윗 조차 한데 그럼에도 불구하고 여전히 심지어 까지도 조차도 하지 않도록 않기 위하여 때 시각 무렵 시간 동안 어때 어떠한 하여금 네 예 우선 누구 누가 알겠는가 아무도 줄은모른다 줄은 몰랏다 하는 김에 겸사겸사 하는바 그런 까닭에 한 이유는 그러니 그러니까 때문에 그 너희 그들 너희들 타인 것 것들 너 위하여 공동으로 동시에 하기 위하여 어찌하여 무엇때문에 붕붕 윙윙 나 우리 엉엉 휘익 윙윙 오호 아하 어쨋든 만 못하다 하기보다는 차라리 하는 편이 낫다 흐흐 놀라다 상대적으로 말하자면 마치 아니라면 쉿 그렇지 않으면 그렇지 않다면 안 그러면 아니었다면 하든지 아니면 이라면 좋아 알았어 하는것도 그만이다 어쩔수 없다 하나 일 일반적으로 일단 한켠으로는 오자마자 이렇게되면 이와같다면 전부 한마디 한항목 근거로 하기에 아울러 하지 않도록 않기 위해서 이르기까지 이 되다 로 인하여 까닭으로 이유만으로 이로 인하여 그래서 이 때문에 그러므로 그런 까닭에 알 수 있다 결론을 낼 수 있다 으로 인하여 있다 어떤것 관계가 있다 관련이 있다 연관되다 어떤것들 에 대해 이리하여 그리하여 여부 하기보다는 하느니 하면 할수록 운운 이러이러하다 하구나 하도다 다시말하면 다음으로 에 있다 에 달려 있다 우리 우리들 오히려 하기는한데 어떻게 어떻해 어찌됏어 어때 어째서 본대로 자 이 이쪽 여기 이것 이번 이렇게말하자면 이런 이러한 이와 같은 요만큼 요만한 것 얼마 안 되는 것 이만큼 이 정도의 이렇게 많은 것 이와 같다 이때 이렇구나 것과 같이 끼익 삐걱 따위 와 같은 사람들 부류의 사람들 왜냐하면 중의하나 오직 오로지 에 한하다 하기만 하면 도착하다 까지 미치다 도달하다 정도에 이르다 할 지경이다 결과에 이르다 관해서는 여러분 하고 있다 한 후 혼자 자기 자기집 자신 우에 종합한것과같이 총적으로 보면 총적으로 말하면 총적으로 대로 하다 으로서 참 그만이다 할 따름이다 쿵 탕탕 쾅쾅 둥둥 봐 봐라 아이야 아니 와아 응 아이 참나 년 월 일 령 영 일 이 삼 사 오 육 륙 칠 팔 구 이천육 이천칠 이천팔 이천구 하나 둘 셋 넷 다섯 여섯 일곱 여덟 아홉 령 영 이 있 하 것 들 그 되 수 이 보 않 없 나 사람 주 아니 등 같 우리 때 년 가 한 지 대하 오 말 일 그렇 위하 때문 그것 두 말하 알 그러나 받 못하 일 그런 또 문제 더 사회 많 그리고 좋 크 따르 중 나오 가지 씨 시키 만들 지금 생각하 그러 속 하나 집 살 모르 적 월 데 자신 안 어떤 내 내 경우 명 생각 시간 그녀 다시 이런 앞 보이 번 나 다른 어떻 여자 개 전 들 사실 이렇 점 싶 말 정도 좀 원 잘 통하 놓" 
    word_tokens = word_tokenize(text)
    stop_words=stop_words.split(' ')

    result = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            result.append(w) 
    return result 



def check_match(evidence_list, retrieved_result):
    """
    evidence_list에 속하는 evidence passage중 하나라도 retrieved_result에 포함되어 있으면 true, 아니면 false를 반환함.
    """
    matched = False
    for evidence in evidence_list:
        if evidence in retrieved_result:
            matched = matched or True
        else: 
            matched = matched or False  
    return matched 

def measure_retriever_accuracy(retriever, retriever_test_data, n:int, doc_to_document_passages):
    """
    Top-n retriever accuracy를 측정함.
    """
    assert n>=2, "N should be larger than 2!"

    matched_question_count = 0
    retrieved_results = []
    for datapoint in tqdm(retriever_test_data, desc="Measure retriever accuracy.."):
        question = datapoint["question"]
        evidence_list = datapoint["evidence_list"]
        top_n_res = retriever.query_index(question, n)
        top_n_res_passage_info_added = [doc_to_document_passages[res] for res in top_n_res]
        retrieved_results.append({"question": question, "retrieved_result":top_n_res_passage_info_added})
        if check_match(evidence_list, top_n_res):
            matched_question_count +=1 
    
    accuracy = matched_question_count / len(retriever_test_data)
    return accuracy 

def evaluate_retriever(retriever, retriever_test_data, doc_to_document_passages):
    """
    Top-2, Top-5, Top-10 retriever accuracy를 측정함
    각 question type별로 Top-2, Top-5, Top-10 retriever accuracy도 측정해주자. 
    """
    top_2_retriever_accuracy = measure_retriever_accuracy(retriever, retriever_test_data, 2, doc_to_document_passages)
    top_5_retriever_accuracy = measure_retriever_accuracy(retriever, retriever_test_data, 5, doc_to_document_passages)
    top_10_retriever_accuracy = measure_retriever_accuracy(retriever, retriever_test_data, 10, doc_to_document_passages)
    top_50_retriever_accuracy = measure_retriever_accuracy(retriever, retriever_test_data, 50, doc_to_document_passages)
    top_100_retriever_accuracy = measure_retriever_accuracy(retriever, retriever_test_data, 100, doc_to_document_passages)

    print("============================")
    print(f"Top-2 retriever accuracy: {top_2_retriever_accuracy}")
    print(f"Top-5 retriever accuracy: {top_5_retriever_accuracy}")
    print(f"Top-10 retriever accuracy: {top_10_retriever_accuracy}")
    print(f"Top-50 retriever accuracy: {top_50_retriever_accuracy}")
    print(f"Top-100 retriever accuracy: {top_100_retriever_accuracy}")
    print("============================")

def batch_replace(sequence, replaced_list, to):
    for w in replaced_list:
        sequence = sequence.replace(w, to)
    return sequence

def fix_incoherent_placeholder(decomp_seq):
    replaced_decomposition = batch_replace(decomp_seq, ["1위", "1번","1등"],"#1")
    replaced_decomposition = batch_replace(replaced_decomposition, ["2위", "2번","2등"],"#2")
    replaced_decomposition = batch_replace(replaced_decomposition, ["3위", "3번","3등"],"#3")
    replaced_decomposition = batch_replace(replaced_decomposition, ["4위", "4번","4등"],"#4")
    return replaced_decomposition

def load_retriever_test_dataset(doc_title_to_context):
    with open("/home/deokhk/project/CBNU_implicit/dataset/strategyqa/kor_translated/strategyqa_train_kor.json", 'r') as f: 
        stqa_kor = json.load(f)
    
    stqa_kor_replaced = []
    for datapoint in stqa_kor:
        replaced_decomposition_list = []
        decomposition_list = datapoint["decomposition"]
        for decomposition in decomposition_list:
            replaced_decomposition = decomposition.replace("# 1", "#1")
            replaced_decomposition = replaced_decomposition.replace("# 2", "#2")
            replaced_decomposition = replaced_decomposition.replace("# 3", "#3")
            replaced_decomposition = replaced_decomposition.replace("# 4", "#4")
            replaced_decomposition = fix_incoherent_placeholder(replaced_decomposition)
            replaced_decomposition_list.append(replaced_decomposition)
        datapoint["decomposition"] = replaced_decomposition_list
        stqa_kor_replaced.append(datapoint)

    data_cleaned = []
    for datapoint in stqa_kor_replaced:
        decomposition_list = datapoint["decomposition"]
        evidences = datapoint["evidence"][0]
        for subquestion, evidence_list in zip(decomposition_list, evidences):
            subq_to_evidence = dict()
            operation_word_list = ["#1", "#2", "#3", "#4", "#5"]
            operation_question_flag = False 
            for word in operation_word_list:
                if word in subquestion:
                    operation_question_flag = True 
            
            if operation_question_flag:
                continue 

            if "operation" in evidence_list or "no_evidence" in evidence_list:
                continue 
            subq_to_evidence["question"] = subquestion
            evidence_list_as_context = []
            for evidence_title in evidence_list[0]:
                try:
                    evidence_list_as_context.append(doc_title_to_context[evidence_title])
                except KeyError:
                    breakpoint()

            subq_to_evidence["evidence_list"] = evidence_list_as_context
            data_cleaned.append(subq_to_evidence)
    return data_cleaned 


class Retriever_with_Reranker(object):
    def __init__(self, passages_list:List, pretrained_embedding_model_name, pretrained_reranker_path):
        self.passages_list = passages_list
        self.corpus_embeddings = None

        print("Loading corpus..")
        self.corpus = [doc["content"] for doc in self.passages_list]
        print("Loaded corpus")

        print(f"Loading pretrained embedding model {pretrained_embedding_model_name}")
        self.embedder = SentenceTransformer(pretrained_embedding_model_name)
        print("Loaded pretrained model")

        print(f"Loaded pretrained cross-encoder reranker from {pretrained_reranker_path}")
        self.reranker = CrossEncoder(pretrained_reranker_path, device="cuda:0")

    def build_corpus_embedding(self):
        """
        Build corpus embeddings
        """
        print("Building corpus embeddings..")
        
        self.corpus_embeddings = self.embedder.encode(self.corpus, convert_to_tensor=True)

        print("Corpus embedding created!")
    
    def query_index(self, query:str, top_k:int):
        """ 
        query embedding 과 유사한  top_k document를 return한다.
        이후 reranker로 reranking을 수행한다.
        """
        # First retrieve
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)

        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=100)

        matched_passages = [self.corpus[idx] for idx in top_results[1]]
        # Second rerank
        q_p_list = [[query, passage] for passage in matched_passages]
        scores = self.reranker.predict(q_p_list)
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

    retriever = Retriever_with_Reranker(mypassage_list, args.pretrained_embedding_model_name, args.pretrained_reranker_model_path)
    retriever.build_corpus_embedding()


    retriever_test_data = load_retriever_test_dataset(doc_title_to_context)
    retriever_test_data = retriever_test_data[0:20]
    evaluate_retriever(retriever, retriever_test_data, doc_to_document_passages)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--passage_path", help="Path to the passages to be retrieved", default="/home/deokhk/project/CBNU_implicit/dataset/strategyqa/kor_translated/strategyqa_train_paras_kor.json")
    parser.add_argument("--pretrained_embedding_model_name", help="Name of the pretrained embedding model", default="distiluse-base-multilingual-cased-v1")
    parser.add_argument("--pretrained_reranker_model_path", help="Path to the pretrained reranker model", default="/home/deokhk/project/CBNU_trained_models/kor_reranker")

    args = parser.parse_args()
    main(args)