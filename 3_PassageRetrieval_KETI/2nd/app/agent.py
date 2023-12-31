
import json
import random

import torch

from transformers import AutoTokenizer

from mhop_dataset import QueryPassageFormatter
from passage_loader import SearchManager

from mhop_retriever import *
from mhop_retriever import load_model
from modeling_retriever import T5MeanBiEncoderRetriever

torch.manual_seed(42)
random.seed(42)

class Service:
    task = [
        {
            'name': "passage_retrieval",
            'description': 'dummy task'
        }
    ]

    def __init__(self):
        self.retriever = Retriever()
    
    @classmethod
    def get_task_list(cls):
        return json.dumps(cls.task), 200
    
    def do(self, content):
        try:
            print(content['question'],flush=True)
            ret = self.retriever.do_search(content)
            if 'error' in ret.keys():
                return json.dumps(ret), 400
            return json.dumps(ret), 200
        except Exception as e:
            return json.dumps(
                {
                    'error': "{}".format(e)
                }
            ), 400


# {
#   "question":{
#     "text":"데드풀 감독이랑 킬러의 보디가드 감독이 같은 사람이야?",
#     "language":"kr",
#     "domain":"common-sense"
#   },
#   "max_num_retrieved":10,
#   "max_hop":5,
#   "num_retrieved":-1
# }


class Retriever(object):
    def __init__(self):
        self.config = json.load(open("config.json", "r"))

        self.search_manager = {
            comb: self.load_search_manager(self.config[comb]) for comb in ["kr_common-sense", "en_common-sense", "kr_law"]
        }
    
    @staticmethod
    def load_search_manager(model_cfg):
        if model_cfg["model_name"] == "T5MeanBiEncoderRetriever":
            model = T5MeanBiEncoderRetriever.from_pretrained(model_cfg["hf_path"])
        else:
            model_class = load_model(model_cfg["model_name"])
            model = model_class.from_pretrained(model_cfg["hf_path"])
        model.eval()

        pre_trained_tokenizer = model_cfg.get("pre_trained_tokenizer", "KETI-AIR/ke-t5-base")
        tokenizer = AutoTokenizer.from_pretrained(pre_trained_tokenizer)
        qp_formatter = QueryPassageFormatter(
            tokenizer,
            max_q_len=70,
            max_q_sp_len=350,
            max_c_len=350,
            remove_question_mark=False,
            add_cls_token=True,
            add_sep_token=True,
            add_token_type_ids=False,
            cls_token_id=-1,
            sep_token_id=-1,
        )
        search_manager = SearchManager(
            qp_formatter,
            model,
            page_info_path=None,
            exhaustive_scorer=False,
            normalize=False,
            use_gpu=model_cfg["use_gpu"],
            model_gpu=model_cfg["model_gpu"],
            scorer_gpu=model_cfg["scorer_gpu"],
            squeeze=model_cfg.get("squeeze", False)
        )
        if model_cfg["use_gpu"]:
            search_manager.cuda()
        
        return search_manager
    
    def do_search(self, content):
        question = content.get('question', None)
        context = content.get('context', None)
        if question is None:
            return {
                'error': "There is no question!!!"
            }
        elif context is None:
            return {
                'error': "You have to pass context set. But got Null context."
            }
        else:
            context = self.convert_context(context)
            question_txt = question.get('text', '')
            question_ln = question.get('language', 'kr')
            question_domain = question.get('domain', 'common-sense')
            max_num_retrieved = content.get('max_num_retrieved', 10)
            max_hop = content.get('max_hop', 2)
            num_retrieved = content.get('num_retrieved', -1)

            if question_txt == '':
                return {
                    'error': "Empty question string!!!"
                }

            query = {
                "question": question_txt,
                "context": context,
            }

            comb_str = "{}_{}".format(question_ln, question_domain)
            if comb_str in self.search_manager:
                return self.search(
                    comb_str, 
                    query, 
                    max_num_retrieved=max_num_retrieved,
                    max_hop=max_hop,
                    num_retrieved=num_retrieved)
            else:
                return {
                    'error': f"The requested combination of language and question domain is currently unsupported. \
                        (kr, common-sense), (en, common-sense) are currently supported on this service. \
                            But got ({question_ln},{question_domain})"
                }
    
    @staticmethod
    def convert_context(context):
        return [{'title': item[0], 'text':' '.join(item[1])} for item in context]

    
    @torch.no_grad()
    def search(self, comb_str, query, max_num_retrieved=10, max_hop=2, num_retrieved=-1):
        max_num_retrieved = max(4, max_num_retrieved)
        retrived_items = self.search_manager[comb_str].search(query, top_k=max_num_retrieved, n_hop=max_hop, num_cands_per_topk=2)

        if num_retrieved > 1:
            max_num_retrieved = num_retrieved

        top_n_docs = self.get_n_docs(retrived_items, max_num_retrieved=max_num_retrieved)
        
        return {
            'num_retrieved_doc': len(top_n_docs),
            'retrieved_doc': top_n_docs,
            'top_n_candidates': retrived_items[:max_num_retrieved]
        }

    
    def get_n_docs(self, items, max_num_retrieved=10):
        title_set = set()
        result_set = []

        for item in items:
            for ctx in item:
                if ctx['title'] not in title_set:
                    result_set.append(ctx)
                    title_set.add(ctx['title'])

                if len(title_set) >= max_num_retrieved:
                    return result_set
        return result_set


if __name__ == "__main__":

    example_content = {
        'question': {
                'text': '매거리 컨트리 파크가 위치한 계획된 정착지의 건설은 몇 년도에 시작되었는가?',
                'language': 'kr',
                'domain': 'common-sense',
                
            }, 
        'context': [
            [
                '코스스턴 레이크스 컨트리 파크', 
                [
                    '코스메스턴 레이크스 컨트리 파크는 글래모건 시의 베일이 소유하고 관리하는 영국의 공공 컨트리 파크이다.', 
                    '그것은 카디프에서 7.3마일 (11.7 킬로미터) 떨어진 글래모건의 베일 페나스와 설리 사이에 위치해 있습니다.', 
                    '2013년 5월 1일 시골공원은 지역자연보호구역 LNR로 지정되었다.', 
                    '부품은 특별한 과학적 관심의 장소입니다.', 
                    '공원, 방문객 센터, 카페는 일년 내내 문을 연다.'
                ]
            ], 
            [
                '크레이가본', 
                [
                    '크레이거본( )은 북아일랜드 아마 주의 주도이다.', 
                    '그것의 건설은 1965년에 시작되었고 북아일랜드의 초대 총리인 제임스 크레이그의 이름을 따서 지어졌습니다.', 
                    '그것은 러건과 포트다운을 통합한 새로운 직선 도시의 심장부가 될 예정이었지만, 이 계획은 버려졌고 제안된 작업의 절반도 되지 않았다.', 
                    '오늘날 지역 주민들 사이에서 "크레이가본"은 두 마을 사이의 지역을 가리킨다.', 
                    '두 개의 인공 호수 옆에 지어졌으며 넓은 주거 지역(브라운로우), 두 번째로 작은 지역(맨더빌), 그리고 실질적인 쇼핑 센터, 법원 및 구의회 본부를 포함하는 중심 지역(하이필드)으로 구성되어 있다.', 
                    '야생동물의 안식처인 호수는 산책로가 있는 삼림지대로 둘러싸여 있다.', 
                    '이 지역에는 수상 스포츠 센터, 애완 동물원과 골프 코스, 스키 슬로프도 있습니다.', 
                    '대부분의 크레이가본에서는 자동차가 보행자와 완전히 분리되어 있으며, 회전교차가 광범위하게 사용된다.'
                ]
            ], 
            [
                '우드게이트 밸리 컨트리 파크', 
                [
                    '우드게이트 밸리 컨트리 파크는 버밍엄의 바틀리 그린과 퀸튼 지역에 있는 컨트리 파크입니다.', 
                    '그것은 서튼 공원과 리키 힐스 컨트리 파크 다음으로 세 번째로 큰 버밍엄 컨트리 파크입니다.', 
                    '이 공원은 야생동물 서식지로 유지되고 있지만 농장 동물들도 있다.'
                ]
            ], 
            [
                '룽푸산 컨트리 파크', 
                [
                    '룽푸산 컨트리 파크( , 1998년 12월 18일 설립)는 홍콩 중서구에 위치한 컨트리 파크이다.', 
                    "사용하지 않는 파인우드 배터리뿐만 아니라 파인우드 가든 피크닉 지역 등 '룽푸산'의 초목이 우거진 경사면을 뒤덮어 홍콩 섬의 주택가와 상권을 조망할 수 있는 경관을 제공한다.", 
                    '중층부 및 서부 지구의 주택가와 인접한 Lung Fu Shan 지역은 대중, 특히 아침 산책객과 피크닉객들이 집중적으로 이용하고 있다.', 
                    '그것은 Pok Fu Lam 컨트리 공원의 북쪽에 위치해 있습니다.', 
                    'Lung Fu Shan 컨트리 파크의 동쪽에는 Hatton Road, 남쪽에는 Harlech Road가 있고 북쪽과 서쪽에는 Supply Department에서 건설한 덮개 도관이 있습니다.', 
                    '이 컨트리 파크의 면적은 약 47헥타르로 홍콩에서 가장 작은 컨트리 파크(특별 구역 제외)이다.', '이곳은 설립 날짜에 따르면 가장 최신의 컨트리 파크이기도 하다.'
                ]
            ], 
            [
                '마헤리 컨트리 파크', 
                [
                    'Maghery Country Park는 북아일랜드 아마 주 County Amagh의 Maghery 마을에 있는 공원입니다.', 
                    '그것은 30에이커의 면적에 걸쳐 있고 5킬로미터의 삼림 산책로와 피크닉 장소를 포함하고 있으며 조류 관찰, 낚시, 그리고 산책을 위해 사용된다.', 
                    '코니 아일랜드는 해안에서 1km 떨어져 있고 주말에는 공원에서 보트 여행을 할 수 있습니다.', 
                    '이곳은 중요한 지역 편의 시설이자 관광 명소이며 크레이가본 구의회에서 관리하고 있다.'
                ]
            ], 
            [
                '킹피셔 컨트리 파크', 
                [
                    '킹피셔 컨트리 파크는 영국에 있는 시골 공원입니다.', 
                    '그것은 웨스트 미들랜즈 주 이스트 버밍엄에 위치해 있습니다.', 
                    '처음에 버밍엄 시의회에 의해 프로젝트 킹피셔로 지정되었던 이 공원은 2004년 7월에 공식적으로 컨트리 공원으로 선언되었습니다.', 
                    '이 컨트리파크는 M6 고속도로의 스몰 히스(버밍엄)에서 첼름슬리 우드(솔리헐)까지 11km의 리버 콜 강 연장을 따라 위치해 있습니다.', 
                    '이것은 지역 자연 보호 구역입니다.'
                ]
            ], 
            [
                '플로버 코브 컨트리 파크', 
                [
                    '플로버 코브 컨트리 파크(, )는 홍콩 동북부 신대륙에 위치한 컨트리 파크이다.', 
                    '최초의 컨트리 파크는 1978년 4월 7일에 설립되었으며, 행정 구역인 북구와 타이포 구의 4,594 헥타르의 자연 지형을 포함한다.', 
                    '1979년 6월 1일 더블 헤이븐 섬과 핑차우 섬을 포함하는 확장 코브 컨트리 파크로 지정되었다.'
                ]
            ], 
            [
                '팻신렝 컨트리 파크', 
                [
                    '팻신렝 컨트리 파크(, )는 홍콩 동북부 신구에 위치한 컨트리 파크이다.', 
                    '1978년 8월 18일에 설립된 이 시골 공원은 3,125헥타르의 자연 지형을 차지하고 있다.', 
                    '그것은 팻 신 렝 산맥과 웡 렝, 핑펑 샨, 흐린 언덕 그리고 콰이 타우 렝을 포함한 다른 돌기들로 구성되어 있습니다.', 
                    '호크타우저수지와 라우수이흥저수지도 시골공원 내에 있다.'
                ]
            ], 
            [
                '코니 아일랜드, 러프 니', 
                [
                    '코니 아일랜드는 북아일랜드의 러프 니에 있는 섬이다.', 
                    '아마흐 카운티의 마헤리에서 약 1km 떨어진 곳에 위치해 있으며 숲이 우거져 있고 면적은 약 9에이커에 달한다.', 
                    '그것은 Lough Neagh의 남서쪽 모퉁이에 있는 Blackwater 강과 Bann 강 어귀 사이에 있다.', 
                    '섬으로의 보트 여행은 Maghery Country Park 또는 Kinnego Marina에서 주말에 이용할 수 있습니다.', 
                    '이 섬은 내셔널 트러스트가 소유하고 있으며 크레이거본 자치구가 그들을 대신하여 관리하고 있다.', 
                    '코니 아일랜드 플랫은 섬에 인접한 바위 돌출부입니다.', 
                    '새뮤얼 루이스가 코니 섬을 아마그 카운티의 유일한 섬이라고 불렀지만, 아마그의 Lough Neagh 구역에는 Croaghan 섬과 Padian, Rathlin 섬, Derrywaragh Island의 변두리 사례도 포함되어 있습니다.'
                ]
            ], 
            [
                '발록 컨트리 파크', 
                [
                    '발록 컨트리 파크는 스코틀랜드 로몬드호의 남쪽 끝에 있는 200에이커의 컨트리 파크이다.', 
                    '1980년 컨트리파크로 인정받았고, 스코틀랜드 최초의 국립공원인 로몬드호와 트로삭스 국립공원에 있는 유일한 컨트리파크이다.', 
                    '발록 컨트리 파크는 자연 산책로, 안내 산책로, 벽으로 둘러싸인 정원, 그리고 로치가 보이는 피크닉 잔디밭을 특징으로 합니다.', 
                    '원래 19세기 초 글래스고와 선박은행의 파트너인 존 뷰캐넌에 의해 개발되었으며 1851년 땅을 매입한 데니스툰-브라운에 의해 정원이 크게 개선되었다.', 
                    'Buchanan은 또한 현재 공원 방문객의 중심 역할을 하고 있는 Balloch Castle을 건설했다.'
                ]
            ]
        ], 
        'max_num_retrieved': 6
    }
    # ['마헤리 컨트리 파크', '0'], ['크레이가본', '1']
    
    
    
    example_content = {
        'question': {
            "text": '연소근로자의 동의나 노동부 장관의 인가없이 야간이나 휴일근로할 할 경우 벌칙이 있나요',
            'language': 'kr',
            'domain': 'law',
        },
        'max_num_retrieved': 6,
        'context': [
            [
                '근로기준법 0113조',
                [
                    '제113조(벌칙) 제45조를 위반한 자는 1천만원 이하의 벌금에 처한다.'
                ]
            ],
            [
                '목재의 지속가능한 이용에 관한 법률 0004조',
                [
                    '제4조(책무)\n\t① 국가 및 지방자치단체는 목재문화의 진흥과 목재교육의 활성화 및 목재제품의 체계적·안정적 공급에 필요한 시책을 수립·시행하여 목재의 지속가능한 이용이 증진되도록 노력하여야 한다. <개정 2017.3.21>\n\t② 산림청장은 국내 또는 원산국의 목재수확 관계 법령을 준수하여 생산(이하 "합법벌채"라 한다)된 목재 또는 목재제품이 유통·이용될 수 있도록 필요한 시책을 수립·시행하여야 한다. <신설 2017.3.21>\n\t③ 목재생산업자는 합법벌채된 목재 또는 목재제품이 수입·유통 및 생산·판매되도록 노력하여야 한다. <신설 2017.3.21>\n'
                ]
            ],
            [
                '공직자윤리법의 시행에 관한 헌법재판소 규칙 0009조',
                [
                    '제9조의4(재산형성과정 소명 요구 등)\n\t① 위원회는 등록의무자가 다음 각 호의 어느 하나에 해당하는 경우에는 법 제8조제13항에 따라 재산형성과정의 소명을 요구할 수 있다. <개정 2020.10.13>\t\t1. 직무와 관련하여 부정한 재산증식을 의심할 만한 상당한 사유가 있는 경우\n\t\t2. 법 제8조의2제6항에 따른 다른 법령을 위반하여 부정하게 재물 또는 재산상 이익을 얻었다는 혐의를 입증하기 위한 경우\n\t\t3. 재산상의 문제로 사회적 물의를 일으킨 경우\n\t\t4. 등록의무자의 보수 수준 등을 고려할 때 특별한 사유 없이 재산의 뚜렷한 증감이 있는 경우\n\t\t5. 제1호부터 제4호까지의 규정에 상당하는 사유로 위원회가 소명 요구를 의결한 경우\n\n\t② 재산형성과정의 소명을 요구받은 사람은 특별한 사유가 없으면 요구받은 날부터 20일 이내에 별지 제3호의5서식의 소명서 및 증빙자료를 위원회에 제출하여야 한다. <개정 2020.10.13>\n\t③ 재산형성과정의 소명을 요구받은 사람은 분실ㆍ멸실 및 훼손 등의 사유로 증빙자료를 제출할 수 없는 경우에는 위원회에 그 사실을 소명하고, 거래시기ㆍ거래상대방 및 거래목적 등을 주요내용으로 하는 증빙자료를 대체할 수 있는 별지 제3호의6서식의 소명서(이하 "증빙자료대체소명서"라 한다)를 위원회에 제출하여야 한다. <개정 2020.10.13>\n\t④ 위원회는 증빙자료대체소명서의 내용에 대한 사실관계를 검증하는 과정에서 추가소명 또는 증빙자료 제출을 요구할 수 있다.\n'
                ]
            ],
            [
                '행정심판법 0054조',
                [
                    '제54조(전자정보처리조직을 이용한 송달 등)\n\t① 피청구인 또는 위원회는 제52조제1항에 따라 행정심판을 청구하거나 심판참가를 한 자에게 전자정보처리조직과 그와 연계된 정보통신망을 이용하여 재결서나 이 법에 따른 각종 서류를 송달할 수 있다. 다만, 청구인이나 참가인이 동의하지 아니하는 경우에는 그러하지 아니하다.\n\t② 제1항 본문의 경우 위원회는 송달하여야 하는 재결서 등 서류를 전자정보처리조직에 입력하여 등재한 다음 그 등재 사실을 국회규칙, 대법원규칙, 헌법재판소규칙, 중앙선거관리위원회규칙 또는 대통령령으로 정하는 방법에 따라 전자우편 등으로 알려야 한다.\n\t③ 제1항에 따른 전자정보처리조직을 이용한 서류 송달은 서면으로 한 것과 같은 효력을 가진다.\n\t④ 제1항에 따른 서류의 송달은 청구인이 제2항에 따라 등재된 전자문서를 확인한 때에 전자정보처리조직에 기록된 내용으로 도달한 것으로 본다. 다만, 제2항에 따라 그 등재사실을 통지한 날부터 2주 이내(재결서 외의 서류는 7일 이내)에 확인하지 아니하였을 때에는 등재사실을 통지한 날부터 2주가 지난 날(재결서 외의 서류는 7일이 지난 날)에 도달한 것으로 본다.\n\t⑤ 서면으로 심판청구 또는 심판참가를 한 자가 전자정보처리조직의 이용을 신청한 경우에는 제52조ㆍ제53조 및 이 조를 준용한다.\n\t⑥ 위원회, 피청구인, 그 밖의 관계 행정기관 간의 서류의 송달 등에 관하여는 제52조ㆍ제53조 및 이 조를 준용한다.\n\t⑦ 제1항 본문에 따른 송달의 방법이나 그 밖에 필요한 사항은 국회규칙, 대법원규칙, 헌법재판소규칙, 중앙선거관리위원회규칙 또는 대통령령으로 정한다.\n'
                ]
            ],
            [
                '출입국관리법 0036조',
                [
                    '제36조(체류지 변경의 신고)\n\t① 제31조에 따라 등록을 한 외국인이 체류지를 변경하였을 때에는 대통령령으로 정하는 바에 따라 전입한 날부터 15일 이내에 새로운 체류지의 시ㆍ군ㆍ구 또는 읍ㆍ면ㆍ동의 장이나 그 체류지를 관할하는 지방출입국ㆍ외국인관서의 장에게 전입신고를 하여야 한다. <개정 2014.3.18, 2016.3.29, 2018.3.20, 2020.6.9>\n\t② 외국인이 제1항에 따른 신고를 할 때에는 외국인등록증을 제출하여야 한다. 이 경우 시ㆍ군ㆍ구 또는 읍ㆍ면ㆍ동의 장이나 지방출입국ㆍ외국인관서의 장은 그 외국인등록증에 체류지 변경사항을 적은 후 돌려주어야 한다. <개정 2014.3.18, 2016.3.29>\n\t③ 제1항에 따라 전입신고를 받은 지방출입국ㆍ외국인관서의 장은 지체 없이 새로운 체류지의 시ㆍ군ㆍ구 또는 읍ㆍ면ㆍ동의 장에게 체류지 변경 사실을 통보하여야 한다. <개정 2014.3.18, 2016.3.29>\n\t④ 제1항에 따라 직접 전입신고를 받거나 제3항에 따라 지방출입국ㆍ외국인관서의 장으로부터 체류지 변경통보를 받은 시ㆍ군ㆍ구 또는 읍ㆍ면ㆍ동의 장은 지체 없이 종전 체류지의 시ㆍ군ㆍ구 또는 읍ㆍ면ㆍ동의 장에게 체류지 변경신고서 사본을 첨부하여 외국인등록표의 이송을 요청하여야 한다. <개정 2014.3.18, 2016.3.29>\n\t⑤ 제4항에 따라 외국인등록표 이송을 요청받은 종전 체류지의 시ㆍ군ㆍ구 또는 읍ㆍ면ㆍ동의 장은 이송을 요청받은 날부터 3일 이내에 새로운 체류지의 시ㆍ군ㆍ구 또는 읍ㆍ면ㆍ동의 장에게 외국인등록표를 이송하여야 한다. <개정 2016.3.29>\n\t⑥ 제5항에 따라 외국인등록표를 이송받은 시ㆍ군ㆍ구 또는 읍ㆍ면ㆍ동의 장은 신고인의 외국인등록표를 정리하고 제34조제2항에 따라 관리하여야 한다. <개정 2016.3.29>\n\t⑦ 제1항에 따라 전입신고를 받은 시ㆍ군ㆍ구 또는 읍ㆍ면ㆍ동의 장이나 지방출입국ㆍ외국인관서의 장은 대통령령으로 정하는 바에 따라 그 사실을 지체 없이 종전 체류지를 관할하는 지방출입국ㆍ외국인관서의 장에게 통보하여야 한다. <개정 2014.3.18, 2016.3.29>\n'
                ]
            ],
            [
                '지방자치단체 보조금 관리에 관한 법률 0038조',
                [
                    '제38조(벌칙) 다음 각 호의 어느 하나에 해당하는 자는 5년 이하의 징역 또는 5천만원 이하의 벌금에 처한다.\n\t\t\t1. 제13조를 위반하여 지방보조금을 다른 용도에 사용한 자\n\t\t2. 제21조제2항을 위반하여 지방자치단체의 장의 승인 없이 중요재산에 대하여 금지된 행위를 한 자\n\n'
                ]
            ],
            [
                '가사소송법 0038조',
                [
                    '제38조(증거 조사) 가정법원은 필요하다고 인정할 경우에는 당사자 또는 법정대리인을 당사자 신문(訊問) 방식으로 심문(審問)할 수 있고, 그 밖의 관계인을 증인 신문 방식으로 심문할 수 있다.\n'
                ]
            ],
            [
                '수산업협동조합법 0176조',
                [
                    '제176조(벌칙)\n\t① 조합등 또는 중앙회의 임직원이 다음 각 호의 어느 하나에 해당하는 행위로 조합등 또는 중앙회에 손실을 끼쳤을 때에는 10년 이하의 징역 또는 1억원 이하의 벌금에 처한다. <개정 2014.10.15, 2015.2.3, 2016.5.29>\t\t1. 조합등 또는 중앙회의 사업 목적 외의 용도로 자금을 사용하거나 대출하는 행위\n\t\t2. 투기의 목적으로 조합등 또는 중앙회의 재산을 처분하거나 이용하는 행위\n\n\t② 제1항의 징역형과 벌금형은 병과(倂科)할 수 있다.\n'
                ]
            ],
            [
                '출입국관리법 시행령 0059조',
                [
                    '제59조(신문조서)\n\t① 법 제48조제3항에 따른 용의자신문조서에는 다음 각 호의 사항을 적어야 한다.\t\t1. 국적ㆍ성명ㆍ성별ㆍ생년월일ㆍ주소 및 직업\n\t\t2. 출입국 및 체류에 관한 사항\n\t\t3. 용의사실의 내용\n\t\t4. 그 밖에 범죄경력 등 필요한 사항\n\n\t② 출입국관리공무원은 법 제48조제6항 또는 제7항에 따라 통역이나 번역을 하게 한 때에는 통역하거나 번역한 사람으로 하여금 조서에 간인(間印)한 후 서명 또는 기명날인하게 하여야 한다.\n'
                ]
            ],
            [
                '관광진흥법 시행령 0017조',
                [
                    '제17조(의견 청취) 위원장은 위원회의 심의사항과 관련하여 필요하다고 인정하면 관계인 또는 안전ㆍ소방 등에 대한 전문가를 출석시켜 그 의견을 들을 수 있다.\n'
                ]
            ],
            [
                '공직자윤리법의 시행에 관한 중앙선거관리위원회 규칙 0001조',
                [
                    '제1조(목적) 이 규칙은 「공직자윤리법」에서 중앙선거관리위원회규칙에 위임된 사항과 그 밖에 그 법의 시행에 관하여 필요한 사항을 규정함을 목적으로 한다. <개정 2006.1.24, 2009.2.19>\n'
                ]
            ],
            [
                '자본시장과 금융투자업에 관한 법률 시행령 0204조',
                [
                    '제204조(안정조작의 방법 등)\n\t① 제203조에 따른 투자매매업자는 법 제176조제3항제1호에 따라 그 증권의 투자설명서에 다음 각 호의 사항을 모두 기재한 경우만 안정조작을 할 수 있다. 다만, 제203조제2호의 경우에는 인수계약의 내용에 이를 기재하여야 한다.\t\t1. 안정조작을 할 수 있다는 뜻\n\t\t2. 안정조작을 할 수 있는 증권시장의 명칭\n\n\t② 제203조에 따른 투자매매업자는 투자설명서나 인수계약의 내용에 기재된 증권시장 외에서는 안정조작을 하여서는 아니 된다.\n\t③ 제203조에 따른 투자매매업자는 안정조작을 할 수 있는 기간(이하 "안정조작기간"이라 한다) 중에 최초의 안정조작을 한 경우에는 지체 없이 다음 각 호의 사항을 기재한 안정조작신고서(이하 "안정조작신고서"라 한다)를 금융위원회와 거래소에 제출하여야 한다.\t\t1. 안정조작을 한 투자매매업자의 상호\n\t\t2. 다른 투자매매업자와 공동으로 안정조작을 한 경우에는 그 다른 투자매매업자의 상호\n\t\t3. 안정조작을 한 증권의 종목 및 매매가격\n\t\t4. 안정조작을 개시한 날과 시간\n\t\t5. 안정조작기간\n\t\t6. 안정조작에 의하여 그 모집 또는 매출을 원활하게 하려는 증권의 모집 또는 매출가격과 모집 또는 매출가액의 총액\n\t\t7. 안정조작을 한 증권시장의 명칭\n\n\t④ 제203조에 따른 투자매매업자는 다음 각 호에서 정하는 가격을 초과하여 안정조작의 대상이 되는 증권(이하 "안정조작증권"이라 한다)을 매수하여서는 아니 된다.\t\t1. 안정조작개시일의 경우\t\t\t가. 최초로 안정조작을 하는 경우: 안정조작개시일 전에 증권시장에서 거래된 해당 증권의 직전 거래가격과 안정조작기간의 초일 전 20일간의 증권시장에서의 평균거래가격 중 낮은 가격. 이 경우 평균거래가격의 계산방법은 금융위원회가 정하여 고시한다.\n\t\t\t나. 최초 안정조작 이후에 안정조작을 하는 경우: 그 투자매매업자의 안정조작 개시가격\n\n\t\t2. 안정조작개시일의 다음 날 이후의 경우: 안정조작 개시가격(같은 날에 안정조작을 한 투자매매업자가 둘 이상 있는 경우에는 이들 투자매매업자의 안정조작 개시가격 중 가장 낮은 가격)과 안정조작을 하는 날 이전에 증권시장에서 거래된 해당 증권의 직전거래가격 중 낮은 가격\n\n\t⑤ 제203조에 따른 투자매매업자는 안정조작을 한 증권시장마다 안정조작개시일부터 안정조작종료일까지의 기간 동안 안정조작증권의 매매거래에 대하여 해당 매매거래를 한 날의 다음 날까지 다음 각 호의 사항을 기재한 안정조작보고서(이하 "안정조작보고서"라 한다)를 작성하여 금융위원회와 거래소에 제출하여야 한다.\t\t1. 안정조작을 한 증권의 종목\n\t\t2. 매매거래의 내용\n\t\t3. 안정조작을 한 투자매매업자의 상호\n\n\t⑥ 금융위원회와 거래소는 안정조작신고서와 안정조작보고서를 다음 각 호에서 정하는 날부터 3년간 비치하고, 인터넷 홈페이지 등을 이용하여 공시하여야 한다.\t\t1. 안정조작신고서의 경우: 이를 접수한 날\n\t\t2. 안정조작보고서의 경우: 안정조작 종료일의 다음 날\n\n\t⑦ 법 제176조제3항제1호에서 "대통령령으로 정하는 날"이란 모집되거나 매출되는 증권의 모집 또는 매출의 청약기간의 종료일 전 20일이 되는 날을 말한다. 다만, 20일이 되는 날과 청약일 사이의 기간에 모집가액 또는 매출가액이 확정되는 경우에는 그 확정되는 날의 다음 날을 말한다.\n\t⑧ 제1항부터 제7항까지에서 규정한 사항 외에 안정조작신고서ㆍ안정조작보고서의 서식과 작성방법 등에 관하여 필요한 사항은 금융위원회가 정하여 고시한다.\n'
                ]
            ],
            [
                '민사집행법 0182조',
                [
                    '제182조(사건의 이송)\n\t① 압류된 선박이 관할구역 밖으로 떠난 때에는 집행법원은 선박이 있는 곳을 관할하는 법원으로 사건을 이송할 수 있다.\n\t② 제1항의 규정에 따른 결정에 대하여는 불복할 수 없다.\n'
                ]
            ],
            [
                '구직자 취업촉진 및 생활안정지원에 관한 법률 0024조',
                [
                    '제24조(소멸시효)\n\t① 구직촉진수당등을 지급받거나 제28조에 따라 반환받을 권리는 3년간 행사하지 아니하면 시효로 소멸한다.\n\t② 제1항에 따른 소멸시효는 수급자 또는 고용노동부장관의 청구로 중단된다.\n'
                ]
            ],
            [
                '검찰압수물사무규칙 0067조',
                [
                    '제67조(사건종결전 처분의 정리) 사건종결전에 이 절에 정한 환부등의 처분을 한 때에는 압수물사무담당직원은 압수표에 그 뜻을 기재하고 소속과장의 확인을 받아야 한다. 이 경우 압수물 총목록 및 압수조서등에도 그 뜻을 기재하여야 한다.\n'
                ]
            ],
            [
                '지방공무원법 0075조',
                [
                    '제75조의2(적극행정의 장려)\n\t① 지방자치단체의 장은 소속 공무원의 적극행정(공무원이 불합리한 규제의 개선 등 공공의 이익을 위해 업무를 적극적으로 처리하는 행위를 말한다. 이하 이 조에서 같다)을 장려하기 위하여 조례로 정하는 바에 따라 계획을 수립ㆍ시행할 수 있다. 이 경우 대통령령으로 정하는 인사상 우대 및 교육의 실시 등의 사항을 포함하여야 한다.\n\t② 적극행정 추진에 관한 다음 각 호의 사항을 심의하기 위하여 지방자치단체의 장 소속으로 적극행정위원회를 둔다. 다만, 적극행정위원회를 두기 어려운 경우에는 인사위원회(시ㆍ도에 복수의 인사위원회를 두는 경우 제1인사위원회를 말한다)가 적극행정위원회의 기능을 대신할 수 있다.\t\t1. 제1항에 따른 계획 수립에 관한 사항\n\t\t2. 공무원이 불합리한 규제의 개선 등 공공의 이익을 위해 업무를 적극적으로 추진하기 위하여 해당 업무의 처리 기준, 절차, 방법 등에 관한 의견 제시를 요청한 사항\n\t\t3. 그 밖에 적극행정 추진을 위하여 필요하다고 대통령령으로 정하는 사항\n\n\t③ 공무원이 적극행정을 추진한 결과에 대하여 해당 공무원의 행위에 고의 또는 중대한 과실이 없다고 인정되는 경우에는 대통령령으로 정하는 바에 따라 징계의결등을 하지 아니한다.\n\t④ 교육부장관 또는 행정안전부장관은 공직사회의 적극행정 문화 조성을 위하여 필요한 사업을 발굴하고 추진할 수 있다.\n\t⑤ 적극행정위원회의 구성ㆍ운영 및 적극행정을 한 공무원에 대한 인사상 우대 등 적극행정을 장려하기 위하여 필요한 사항은 대통령령으로 정한다.\n'
                ]
            ],
            [
                '체육시설의 설치ㆍ이용에 관한 법률 시행규칙 0007조',
                [
                    '제7조(대중골프장업의 세분) 영 제7조제2항에 따라 대중골프장업의 종류를 다음 각 호와 같이 세분 한다.\n\t\t\t1. 정규 대중골프장업\n\t\t2. 일반 대중골프장업\n\t\t3. 간이골프장업\n\n'
                ]
            ],
            [
                '부동산등기규칙 0152조',
                [
                    '제152조(가처분등기 이후의 등기의 말소)\n\t① 소유권이전등기청구권 또는 소유권이전등기말소등기(소유권보존등기말소등기를 포함한다. 이하 이 조에서 같다)청구권을 보전하기 위한 가처분등기가 마쳐진 후 그 가처분채권자가 가처분채무자를 등기의무자로 하여 소유권이전등기 또는 소유권말소등기를 신청하는 경우에는, 법 제94조제1항에 따라 가처분등기 이후에 마쳐진 제3자 명의의 등기의 말소를 단독으로 신청할 수 있다. 다만, 다음 각 호의 등기는 그러하지 아니하다.\t\t1. 가처분등기 전에 마쳐진 가압류에 의한 강제경매개시결정등기\n\t\t2. 가처분등기 전에 마쳐진 담보가등기, 전세권 및 저당권에 의한 임의경매개시결정등기\n\t\t3. 가처분채권자에게 대항할 수 있는 주택임차권등기등\n\n\t② 가처분채권자가 제1항에 따른 소유권이전등기말소등기를 신청하기 위하여는 제1항 단서 각 호의 권리자의 승낙이나 이에 대항할 수 있는 재판이 있음을 증명하는 정보를 첨부정보로서 등기소에 제공하여야 한다.\n'
                ]
            ],
            [
                '물품관리법 시행령 0027조',
                [
                    '제27조(회계 간의 관리전환)\n\t① 법 제22조제2항에서 "대통령령으로 정하는 관리전환의 경우"란 다음 각 호의 어느 하나에 해당하는 경우를 말한다.\t\t1. 6개월 이내에 반환하는 조건으로 물품을 관리전환하는 경우\n\t\t2. 제26조제2호에 따른 물품을 관리전환하는 경우\n\t\t3. 각 중앙관서의 장이 조달청장과 협의하여 무상으로 관리전환하기로 정한 경우\n\n\t② 법 제22조제2항에 따라 관리전환을 유상으로 정리할 때의 가액은 해당 물품의 대장가격(臺帳價格)으로 한다. 다만, 대장가격으로 정리하기 곤란할 때에는 시가(時價)로 정리할 수 있다.\n'
                ]
            ],
            [
                '노동조합 및 노동관계조정법 시행령 0020조',
                [
                    '제20조(방산물자 생산업무 종사자의 범위) 법 제41조제2항에서 "주로 방산물자를 생산하는 업무에 종사하는 자"라 함은 방산물자의 완성에 필요한 제조ㆍ가공ㆍ조립ㆍ정비ㆍ재생ㆍ개량ㆍ성능검사ㆍ열처리ㆍ도장ㆍ가스취급 등의 업무에 종사하는 자를 말한다.\n'
                ]
            ],
            [
                '혁신의료기기 지원 및 관리 등에 관한 규칙 0011조',
                [
                '제11조(혁신의료기기소프트웨어의 변경허가 또는 변경인증) 법 제24조제4항에 따라 혁신의료기기소프트웨어의 변경허가 또는 변경인증을 받으려는 자는 그 변경이 있은 날부터 30일 이내에 「의료기기법 시행규칙」 제26조제3항에 따른 변경허가(변경인증) 신청서(전자문서로 된 신청서를 포함한다)에 다음 각 호의 자료(전자문서를 포함한다)를 첨부하여 식품의약품안전처장 또는 「의료기기법」 제42조에 따른 한국의료기기안전정보원(이하 "정보원"이라 한다)에 제출해야 한다.\n\t\t\t1. 변경사실을 확인할 수 있는 서류\n\t\t2. 「의료기기법」 제6조제5항에 따른 기술문서와 임상시험자료(혁신의료기기소프트웨어의 안전성ㆍ유효성에 영향을 미치는 경우로서 식품의약품안전처장이 정하여 고시하는 변경사항만 해당한다)\n\t\t3. 제15조에 따른 시설과 제조 및 품질관리체계의 기준에 적합함을 증명하는 자료(제조소 또는 영업소 등이 변경되는 경우로서 식품의약품안전처장이 정하여 고시하는 변경사항만 해당한다)\n\n'
                ]
            ]
        ],
    }
    
    
    
    example_content = {
        'question': {
            "text": '저는 부산에 있는 상가건물을 보증금 6,000만원, 월세 100만원에 임차하여 장사를 하고 있습니다. 그런데 임대인이 1년의 계약기간이 만료한 지 3개월이 지난 최근에 보증금 2,000만원을 더 올려주지 않으면 가게를 비우라고 합니다. 지금은 장사가 잘 되는 때라 점포를 그냥 비워 주자니 아까운 상황인바, 사람들 말로는 약정기간이 만료하여도 임대인으로부터 재계약조건에 관한 아무런 통지를 받지 않았다면 자동갱신된 것으로 보아 계속 점포를 사용할 수 있다고 하는데사실인가요?',
            'language': 'kr',
            'domain': 'law',
        },
        'max_num_retrieved': 6,
        'context': [
            [
                '상가건물 임대차보호법 0010조',
                [
                    '제10조의8(차임연체와 해지) 임차인의 차임연체액이 3기의 차임액에 달하는 때에는 임대인은 계약을 해지할 수 있다.'
                ]
            ],
            ['군사법원법 0257조', ['제257조의2(압수물의 환부, 가환부)\n\t① 군검사는 사본을 확보한 경우 등 압수를 계속할 필요가 없다고 인정되는 압수물 및 증거에 사용할 압수물에 대하여 공소제기 전이라도 소유자, 소지자, 보관자 또는 제출인의 청구가 있는 때에는 환부 또는 가환부하여야 한다.\n\t② 제1항의 청구에 대하여 군검사가 이를 거부하는 경우에는 신청인은 해당 군검사의 소속 보통검찰부에 대응한 군사법원에 압수물의 환부 또는 가환부 결정을 청구할 수 있다.\n\t③ 제2항의 청구에 대하여 군사법원이 환부 또는 가환부를 결정하면 군검사는 신청인에게 압수물을 환부 또는 가환부하여야 한다.\n\t④ 군사법경찰관의 환부 또는 가환부 처분에 관하여는 제1항부터 제3항까지의 규정을 준용한다. 이 경우 군사법경찰관은 군검사의 동의를 받아야 한다.\n']], 
            ['행정심판법 시행령 0011조', ['제11조(수당 등의 지급) 위원회(소위원회 또는 전문위원회를 포함한다)의 회의에 출석하거나 안건을 검토한 위원에게는 예산의 범위에서 출석수당, 안건검토수당 및 여비를 지급한다. 다만, 공무원인 위원이 소관 업무와 직접 관련되어 출석하거나 안건을 검토한 경우에는 그러하지 아니하다.\n']], 
            ['상가건물 임대차보호법 0021조', ['제21조(주택임대차분쟁조정위원회 준용) 조정위원회에 대하여는 이 법에 규정한 사항 외에는 주택임대차분쟁조정위원회에 관한 「주택임대차보호법」 제14조부터 제29조까지의 규정을 준용한다. 이 경우 "주택임대차분쟁조정위원회"는 "상가건물임대차분쟁조정위원회"로 본다.\n']], 
            ['구직자 취업촉진 및 생활안정지원에 관한 법률 0014조', ['제14조(구직활동지원 프로그램)\n\t① 고용노동부장관은 수급자의 취업활동계획에 따라 일자리 소개 및 이력서 작성ㆍ면접 기법 등 구직활동에 필요한 프로그램(이하 "구직활동지원 프로그램"이라 한다)을 제공하여야 한다.\n\t② 구직활동지원 프로그램의 구체적인 내용 및 방법 등에 관하여 필요한 사항은 고용노동부령으로 정한다.\n']], 
            ['행정심판법 시행규칙 0005조', ['제5조(문서의 서식)\n\t① 위원회 및 위원회 위원장의 결정은 별지 제6호서식 및 별지 제7호서식에 따른다.\n\t② 제1항, 제2조 및 제3조에 따른 서식 외에 위원회 또는 법 제36조제2항에 따라 증거조사를 하는 자가 행정심판에 관하여 사용하는 문서의 서식은 다음 각 호와 같다. <개정 2017.10.19>\t\t1. 법 제21조제1항 및 「행정심판법 시행령」(이하 "영"이라 한다) 제18조에 따른 심판참가 요구서: 별지 제8호서식\n\t\t2. 법 제32조제1항 및 영 제24조제1항에 따른 보정요구서: 별지 제9호서식\n\t\t3. 법 제36조 및 영 제25조제3항부터 제5항까지의 규정에 따른 증거조사조서: 별지 제10호서식\n\t\t4. 법 제36조제1항에 따른 증거자료 영치증명서: 별지 제11호서식\n\t\t5. 법 제36조제1항에 따른 감정의뢰서: 별지 제12호서식\n\t\t6. 법 제36조제1항에 따른 감정통지서: 별지 제13호서식\n\t\t7. 법 제36조제2항 및 영 제25조제5항에 따른 증거조사 촉탁서: 별지 제14호서식\n\t\t8. 법 제40조제2항에 따른 서면심리 통지서: 별지 제15호서식\n\t\t9. 법 제46조에 따른 재결서: 별지 제16호서식\n\t\t9. 2. 법 제50조의2제5항에 따른 집행문: 별지 제16호의2서식\n\t\t10. 영 제28조에 따른 회의록: 별지 제17호서식\n\n\t③ 청구인, 행정심판 피청구인(이하 "피청구인"이라 한다), 참가인 또는 관계인이 행정심판에 관하여 사용하는 문서의 서식은 다음 각 호와 같다. <개정 2012.9.20, 2017.10.19, 2018.11.1>\t\t1. 법 제10조 및 영 제12조에 따른 제척ㆍ기피 신청서: 별지 제18호서식\n\t\t2. 법 제15조제1항에 따른 선정대표자 선정서: 별지 제19호서식\n\t\t3. 법 제15조제5항에 따른 선정대표자 해임서: 별지 제20호서식\n\t\t4. 법 제16조제3항에 따른 청구인 지위 승계 신고서: 별지 제21호서식\n\t\t5. 법 제16조제5항에 따른 청구인 지위 승계 허가신청서: 별지 제22호서식\n\t\t6. 법 제16조제8항 및 영 제14조제1항, 법 제17조제6항 및 영 제15조제3항, 법 제20조제6항 및 영 제17조, 법 제29조제7항 및 영 제21조 등에 따른 위원회 결정에 대한 이의신청서: 별지 제23호서식\n\t\t7. 법 제17조제2항ㆍ제5항 및 영 제15조에 따른 피청구인 경정신청서: 별지 제24호서식\n\t\t8. 법 제18조에 따른 대리인 선임서(위임장): 별지 제25호서식\n\t\t9. 법 제18조제1항ㆍ제2항 및 영 제16조에 따른 대리인 선임 허가신청서: 별지 제26호서식\n\t\t10. 법 제18조제3항에 따른 대리인 해임서: 별지 제27호서식\n\t\t10. 2. 법 제18조의2제1항 및 영 제16조의2제2항에 따른 국선대리인 선임 신청서: 별지 제27호의2서식\n\t\t11. 법 제19조제2항에 따른 대표자 등의 자격상실 신고서: 별지 제28호서식\n\t\t12. 법 제20조제2항에 따른 심판참가 허가신청서: 별지 제29호서식\n\t\t13. 법 제28조 및 영 제20조에 따른 행정심판 청구서: 별지 제30호서식\n\t\t14. 법 제29조에 따른 청구변경신청서: 별지 제31호서식\n\t\t15. 법 제30조제5항에 따른 집행정지결정 취소신청서: 별지 제32호서식\n\t\t16. 법 제30조제5항 및 영 제22조제1항에 따른 집행정지신청서: 별지 제33호서식\n\t\t17. 법 제31조제2항에 따른 임시처분 신청서: 별지 제34호서식\n\t\t18. 법 제31조제2항에 따른 임시처분 취소신청서: 별지 제35호서식\n\t\t19. 법 제32조제2항에 따른 심판청구 보정서: 별지 제36호서식\n\t\t20. 법 제34조제1항 및 제2항에 따른 증거서류 등 제출서: 별지 제37호서식\n\t\t21. 법 제36조제1항 및 영 제25조제1항에 따른 증거조사 신청서: 별지 제38호서식\n\t\t22. 법 제40조제1항 단서 및 영 제27조에 따른 구술심리 신청서: 별지 제39호서식\n\t\t23. 법 제42조제1항ㆍ제3항 및 영 제30조, 법 제15조제3항에 따른 심판청구 취하서: 별지 제40호서식\n\t\t24. 법 제42조제2항ㆍ제3항 및 영 제30조에 따른 심판참가신청 취하서: 별지 제41호서식\n\t\t25. 법 제50조제1항에 따른 의무이행심판 인용재결 이행신청서: 별지 제42호서식\n\t\t25. 2. 법 제50조의2제1항 및 영 제33조의2제1항에 따른 간접강제신청서: 별지 제42호의2서식\n\t\t25. 3. 법 제50조의2제2항 및 영 제33조의2제1항에 따른 간접강제결정 변경신청서: 별지 제42호의3서식\n\t\t25. 4. 법 제50조의2제5항에 따른 집행문부여 신청서: 별지 제42호의4서식\n\t\t25. 5. 법 제50조의2제6항에 따라 준용되는 「민사집행법」 제31조에 따른 승계집행문부여 신청서: 별지 제42호의5서식\n\t\t26. 법 제55조에 따른 증거서류 등 반환신청서: 별지 제43호서식\n\t\t27. 영 제31조제1항에 따른 재결경정신청서: 별지 제44호서식\n\n']], 
            ['국가를 당사자로 하는 소송에 관한 법률 0010조', ['제10조 (임의변제의 절차 등) 국가소송에서 금전 지급을 목적으로 하는 사건이 국가의 패소로 확정되어 국가에서 임의변제를 하려는 경우 그 지급기관, 지급절차, 지급방법, 그 밖에 필요한 사항은 대통령령으로 정한다.\n']], 
            ['형법 0196조', ['제196조(미수범) 제192조제2항, 제193조제2항과 전조의 미수범은 처벌한다.\n']], 
            ['국민건강보험법 시행규칙 0014조', ['제14조(본인부담액 경감 인정)\n\t① 영 별표 2 제3호라목에 따라 본인부담액을 경감받을 수 있는 요건을 갖춘 희귀난치성질환자등은 본인부담액 경감 인정을 받으려면 경감 인정 신청서(전자문서를 포함한다)에 다음 각 호의 서류(전자문서를 포함한다)를 첨부하여 특별자치도지사ㆍ시장ㆍ군수ㆍ구청장에게 제출하여야 한다. <개정 2013.9.30, 2015.7.24, 2015.12.31>\t\t1. 영 별표 2 제3호라목에 따른 부양의무자(이하 "부양의무자"라 한다)와의 관계를 확인할 수 있는 가족관계등록부의 증명서(세대별 주민등록표 등본으로 부양의무자와의 관계를 확인할 수 없는 경우만 해당한다)\n\t\t2. 임대차계약서(주택을 임대하거나 임차하고 있는 사람만 해당한다)\n\t\t3. 요양기관이 발급한 진단서 1부(6개월 이상 치료를 받고 있거나 6개월 이상 치료가 필요한 사람만 해당한다)\n\n\t② 제1항에 따른 신청인의 가족, 친족, 이해관계인 또는 「사회복지사업법」 제14조에 따른 사회복지 전담공무원은 신청인이 신체적ㆍ정신적인 이유로 신청을 할 수 없는 경우에는 신청인을 대신하여 제1항에 따른 신청을 할 수 있다. 이 경우 다음 각 호의 구분에 따른 서류를 제시하거나 제출하여야 한다.\t\t1. 신청인의 가족ㆍ친족 또는 이해관계인: 신청인과의 관계를 증명하는 서류\n\t\t2. 사회복지 전담공무원: 공무원임을 증명하는 신분증\n\n\t③ 제1항과 제2항에 따른 신청을 받은 특별자치도지사ㆍ시장ㆍ군수ㆍ구청장은 신청인이 제15조에 따른 기준에 해당하는지를 확인하여 부득이한 사유가 없으면 그 결과를 신청일부터 30일 이내에 공단에 통보하여야 한다. 다만, 다음 각 호의 어느 하나에 해당하는 경우에는 신청일부터 60일 이내에 통보할 수 있다. <개정 2015.12.31>\t\t1. 부양의무자의 소득 조사에 시간이 걸리는 특별한 사유가 있는 경우\n\t\t2. 제1항에 따른 경감 인정 신청서를 제출한 희귀난치성질환자등 또는 부양의무자가 같은 항 또는 관계 법령에 따른 조사나 자료제출 요구를 거부ㆍ방해 또는 기피하는 경우\n\n\t④ 공단은 제3항에 따른 확인 결과를 통보받았을 때에는 부득이한 사유가 없으면 통보를 받은 날부터 7일 이내에 영 별표 2 제3호라목에 따른 인정 여부를 결정하여 그 결과를 신청인에게 통보하여야 한다. <개정 2015.12.31>\n\t⑤ 제1항부터 제4항까지에서 규정한 사항 외에 본인부담액의 경감 인정 절차 등에 관하여 필요한 사항은 보건복지부장관이 정한다.\n']], 
            ['개인정보 보호법 시행령 0004조', ['제4조의2(영리업무의 금지) 법 제7조제1항에 따른 개인정보 보호위원회(이하 "보호위원회"라 한다)의 위원은 법 제7조의6제1항에 따라 영리를 목적으로 다음 각 호의 어느 하나에 해당하는 업무에 종사해서는 안 된다.\n\t\t\t1. 법 제7조의9제1항에 따라 보호위원회가 심의ㆍ의결하는 사항과 관련된 업무\n\t\t2. 법 제40조제1항에 따른 개인정보 분쟁조정위원회(이하 "분쟁조정위원회"라 한다)가 조정하는 사항과 관련된 업무\n\n']], 
            ['공공데이터의 제공 및 이용 활성화에 관한 법률 0021조', ['제21조(공공데이터 포털의 운영)\n\t① 행정안전부장관은 공공데이터의 효율적 제공을 위하여 통합제공시스템(이하 "공공데이터 포털"이라 한다)을 구축ㆍ관리하고 활용을 촉진하여야 한다. <개정 2014.11.19, 2017.7.26>\n\t② 행정안전부장관은 공공기관의 장에게 공공데이터 포털의 구축과 운영에 필요한 공공데이터의 연계, 제공 등의 협력을 요청할 수 있다. 이 경우 요청을 받은 공공기관의 장은 특별한 사유가 없는 한 이에 따라야 한다. <개정 2014.11.19, 2017.7.26>\n\t③ 그 밖에 공공데이터 포털의 구축ㆍ관리 및 활용촉진 등 필요한 사항은 대통령령으로 정한다.\n']], 
            ['특정조달을 위한 국가를 당사자로 하는 계약에 관한 법률 시행령 특례규정 0032조', ['제32조 삭제 <2013.7.15>\n']], 
            ['민법 0701조', ['제701조(준용규정) 제682조, 제684조 내지 제687조 및 제688조제1항, 제2항의 규정은 임치에 준용한다.\n']], 
            ['방송통신위원회의 설치 및 운영에 관한 법률 시행령 0004조', ['제4조(결격사유) 법 제10조제2항 및 법 제19조제2항에 따른 방송ㆍ통신 관련 사업에 종사하는 자의 범위는 다음 각 호와 같다. <개정 2010.10.1>\n\t\t\t1. 「방송법」 제2조제2호ㆍ제5호ㆍ제8호ㆍ제11호ㆍ제13호에 따른 방송사업 등에 종사하는 자\n\t\t2. 「전기통신사업법」 제5조제2항에 따른 기간통신사업에 종사하는 자\n\n']], 
            ['소득세법 시행령 0091조', ['제91조(재고자산 평가방법)\n\t① 법 제39조의 규정을 적용함에 있어서의 재고자산(유가증권을 제외한다)의 평가방법은 다음 각호의 1에 해당하는 것으로 한다.\t\t1. 원가법\n\t\t2. 저가법\n\n\t② 제1항제1호의 원가법을 적용하는 경우에는 다음 각호의 1에 해당하는 평가방법에 의한다. <개정 1998.12.31>\t\t1. 개별법\n\t\t2. 선입선출법\n\t\t3. 후입선출법\n\t\t4. 총평균법\n\t\t5. 이동평균법\n\t\t6. 매출가격환원법\n\n\t③ 제1항 및 제2항에 따라 재고자산을 평가하는 경우에는 해당 자산을 다음 각 호의 구분에 따라 종류별ㆍ사업장별로 각각 다른 방법으로 평가할 수 있다. <개정 2010.2.18>\t\t1. 제품과 상품(건물건설업 또는 부동산 개발 및 공급업을 경영하는 사업자가 매매를 목적으로 소유하는 부동산을 포함한다)\n\t\t2. 반제품과 재공품\n\t\t3. 원재료\n\t\t4. 저장품\n\n\t④ 삭제 <2010.2.18>\n']], 
            ['댐건설 및 주변지역지원 등에 관한 법률 시행령 0009조', ['제9조 삭제 <2004.7.30>\n']], 
            ['독립유공자예우에 관한 법률 시행령 0018조', ['제18조(여유자금의 운용)\n\t① 기금의 여유자금은 다음 각 호와 같이 운용할 수 있다.\t\t1. 국채, 공채, 그 밖의 유가증권 매입\n\t\t2. 금융기관에 예탁\n\t\t3. 공공자금 관리기금에 예탁\n\t\t4. 그 밖에 기금수익을 위한 사업\n\n\t② 기금출납공무원이 제1항제2호에 따라 기금을 예탁할 때에는 금융기관에 기금출납공무원 예탁금 계좌를 설치하고 예탁하여야 한다.\n']], 
            ['상법 0731조', ['제731조(타인의 생명의 보험)\n\t① 타인의 사망을 보험사고로 하는 보험계약에는 보험계약 체결시에 그 타인의 서면(「전자서명법」 제2조제2호에 따른 전자서명이 있는 경우로서 대통령령으로 정하는 바에 따라 본인 확인 및 위조ㆍ변조 방지에 대한 신뢰성을 갖춘 전자문서를 포함한다)에 의한 동의를 얻어야 한다. <개정 1991.12.31, 2017.10.31, 2020.6.9>\n\t② 보험계약으로 인하여 생긴 권리를 피보험자가 아닌 자에게 양도하는 경우에도 제1항과 같다. <개정 1991.12.31>\n']], 
            ['건설산업기본법 시행령 0045조', ['제45조(건설사업자의 실태조사 등)\n\t① 국토교통부장관 또는 지방자치단체의 장(제86조제1항에 따라 위임받은 사무의 처리를 위하여 필요한 경우에 한정한다)은 법 제49조제1항에 따라 소속 공무원에게 경영실태를 조사하게 하거나 자재ㆍ시설을 검사하게 하는 때에는 그 사유를 미리 건설사업자에게 통보해야 한다. <개정 2007.12.28, 2008.2.29, 2013.3.23, 2020.2.18>\n\t② 법 제49조제1항의 규정에 의한 조사 또는 검사를 하는 공무원이 준수하여야 할 사항에 관하여 필요한 사항은 국토교통부령으로 정할 수 있다. <개정 2008.2.29, 2013.3.23>\n\t③ 법 제49조제7항에 따른 경영실태의 조사는 법 제10조에 따른 건설업 등록기준에의 적합 여부에 관한 조사를 그 내용으로 한다. <신설 2016.8.4>\n\t④ 국토교통부장관 또는 지방자치단체의 장은 법 제49조제7항에 따라 경영실태의 조사를 할 때에는 조사의 기간, 내용 및 사유를 해당 건설사업자에게 통보해야 한다. <신설 2016.8.4, 2020.2.18>\n\t⑤ 국토교통부장관은 제3항 및 제4항에서 규정한 사항 외에 경영실태의 조사에 필요한 사항을 정하여 고시할 수 있다. <신설 2016.8.4>\n']], 
            ['공무원직장협의회의 설립ㆍ운영에 관한 법률 시행령 0001조', ['제1조(목적) 이 영은 공무원직장협의회의 설립ㆍ운영에 관한 법률에서 위임된 사항과 그 시행에 관하여 필요한 사항을 규정함을 목적으로 한다. <개정 2020.5.19>\n']], 
            ['교통약자의 이동편의 증진법 시행령 0009조', ['제9조(지방교통약자 이동편의 증진계획의 경미한 변경) 법 제7조제10항 단서에서 "대통령령으로 정하는 경미한 사항을 변경하는 경우"란 제5조 각 호의 어느 하나에 해당하는 경우를 말한다.\n']]
        ],
    }

    

    test_model = Retriever()
    ret = test_model.do_search(example_content)
    print(ret)




# {'num_retrieved_doc': 6, 
# 'retrieved_doc': [
#     {'title': '룽푸산 컨트리 파크', 'text': "룽푸산 컨트리 파크( , 1998년 12월 18일 설립)는 홍콩 중서구에 위치한 컨트리 파크이다. 사용하지 않는 파인우드 배터리뿐만 아니라 파인우드 가든 피크닉 지역 등 '룽푸산'의 초목이 우거진 경사면을 뒤덮어 홍콩 섬의 주택가와 상권을 조망할 수 있는 경관을 제공한다. 중층부 및 서부 지구의 주택가와 인접한 Lung Fu Shan 지역은 대중, 특히 아침 산책객과 피크닉객들이 집중적으로 이용하고 있다. 그것은 Pok Fu Lam 컨트리 공원의 북쪽에 위치해 있습니다. Lung Fu Shan 컨트리 파크의 동쪽에는 Hatton Road, 남쪽에는 Harlech Road가 있고 북쪽과 서쪽에는 Supply Department에서 건설한 덮개 도관이 있습니다. 이 컨트리 파크의 면적은 약 47헥타르로 홍콩에서 가장 작은 컨트리 파크(특별 구역 제외)이다. 이곳은 설립 날짜에 따르면 가장 최신의 컨트리 파크이기도 하다."}, 
#     {'title': '플로버 코브 컨트리 파크', 'text': '플로버 코브 컨트리 파크(, )는 홍콩 동북부 신대륙에 위치한 컨트리 파크이다. 최초의 컨트리 파크는 1978년 4월 7일에 설립되었으며, 행정 구역인 북구와 타이포 구의 4,594 헥타르의 자연 지형을 포함한다. 1979년 6월 1일 더블 헤이븐 섬과 핑차우 섬을 포함하는 확장 코브 컨트리 파크로 지정되었다.'}, 
#     {'title': '팻신렝 컨트리 파크', 'text': '팻신렝 컨트리 파크(, )는 홍콩 동북부 신구에 위치한 컨트리 파크이다. 1978년 8월 18일에 설립된 이 시골 공원은 3,125헥타르의 자연 지형을 차지하고 있다. 그것은 팻 신 렝 산맥과 웡 렝, 핑펑 샨, 흐린 언덕 그리고 콰이 타우 렝을 포함한 다른 돌기들로 구성되어 있습니다. 호크타우저수지와 라우수이흥저수지도 시골공원 내에 있다.'}, 
#     {'title': '우드게이트 밸리 컨트리 파크', 'text': '우드게이트 밸리 컨트리 파크는 버밍엄의 바틀리 그린과 퀸튼 지역에 있는 컨트리 파크입니다. 그것은 서튼 공원과 리키 힐스 컨트리 파크 다음으로 세 번째로 큰 버밍엄 컨트리 파크입니다. 이 공원은 야생동물 서식지로 유지되고 있지만 농장 동물들도 있다.'}, 
#     {'title': '킹피셔 컨트리 파크', 'text': '킹피셔 컨트리 파크는 영국에 있는 시골 공원입니다. 그것은 웨스트 미들랜즈 주 이스트 버밍엄에 위치해 있습니다. 처음에 버밍엄 시의회에 의해 프로젝트 킹피셔로 지정되었던 이 공원은 2004년 7월에 공식적으로 컨트리 공원으로 선언되었습니다. 이 컨트리파크는 M6 고속도로의 스몰 히스(버밍엄)에서 첼름슬리 우드(솔리헐)까지 11km의 리버 콜 강 연장을 따라 위치해 있습니다. 이것은 지역 자연 보호 구역입니다.'}, 
#     {'title': '발록 컨트리 파크', 'text': '발록 컨트리 파크는 스코틀랜드 로몬드호의 남쪽 끝에 있는 200에이커의 컨트리 파크이다. 1980년 컨트리파크로 인정받았고, 스코틀랜드 최초의 국립공원인 로몬드호와 트로삭스 국립공원에 있는 유일한 컨트리파크이다. 발록 컨트리 파크는 자연 산책로, 안내 산책로, 벽으로 둘러싸인 정원, 그리고 로치가 보이는 피크닉 잔디밭을 특징으로 합니다. 원래 19세기 초 글래스고와 선박은행의 파트너인 존 뷰캐넌에 의해 개발되었으며 1851년 땅을 매입한 데니스툰-브라운에 의해 정원이 크게 개선되었다. Buchanan은 또한 현재 공원 방문객의 중심 역할을 하고 있는 Balloch Castle을 건설했다.'}], 
#     'top_n_candidates': [
#         [
#             {'title': '룽푸산 컨트리 파크', 'text': "룽푸산 컨트리 파크( , 1998년 12월 18일 설립)는 홍콩 중서구에 위치한 컨트리 파크이다. 사용하지 않는 파인우드 배터리뿐만 아니라 파인우드 가든 피크닉 지역 등 '룽푸산'의 초목이 우거진 경사면을 뒤덮어 홍콩 섬의 주택가와 상권을 조망할 수 있는 경관을 제공한다. 중층부 및 서부 지구의 주택가와 인접한 Lung Fu Shan 지역은 대중, 특히 아침 산책객과 피크닉객들이 집중적으로 이용하고 있다. 그것은 Pok Fu Lam 컨트리 공원의 북쪽에 위치해 있습니다. Lung Fu Shan 컨트리 파크의 동쪽에는 Hatton Road, 남쪽에는 Harlech Road가 있고 북쪽과 서쪽에는 Supply Department에서 건설한 덮개 도관이 있습니다. 이 컨트리 파크의 면적은 약 47헥타르로 홍콩에서 가장 작은 컨트리 파크(특별 구역 제외)이다. 이곳은 설립 날짜에 따르면 가장 최신의 컨트리 파크이기도 하다."}, 
#             {'title': '플로버 코브 컨트리 파크', 'text': '플로버 코브 컨트리 파크(, )는 홍콩 동북부 신대륙에 위치한 컨트리 파크이다. 최초의 컨트리 파크는 1978년 4월 7일에 설립되었으며, 행정 구역인 북구와 타이포 구의 4,594 헥타르의 자연 지형을 포함한다. 1979년 6월 1일 더블 헤이븐 섬과 핑차우 섬을 포함하는 확장 코브 컨트리 파크로 지정되었다.'}], 
#         [
#             {'title': '룽푸산 컨트리 파크', 'text': "룽푸산 컨트리 파크( , 1998년 12월 18일 설립)는 홍콩 중서구에 위치한 컨트리 파크이다. 사용하지 않는 파인우드 배터리뿐만 아니라 파인우드 가든 피크닉 지역 등 '룽푸산'의 초목이 우거진 경사면을 뒤덮어 홍콩 섬의 주택가와 상권을 조망할 수 있는 경관을 제공한다. 중층부 및 서부 지구의 주택가와 인접한 Lung Fu Shan 지역은 대중, 특히 아침 산책객과 피크닉객들이 집중적으로 이용하고 있다. 그것은 Pok Fu Lam 컨트리 공원의 북쪽에 위치해 있습니다. Lung Fu Shan 컨트리 파크의 동쪽에는 Hatton Road, 남쪽에는 Harlech Road가 있고 북쪽과 서쪽에는 Supply Department에서 건설한 덮개 도관이 있습니다. 이 컨트리 파크의 면적은 약 47헥타르로 홍콩에서 가장 작은 컨트리 파크(특별 구역 제외)이다. 이곳은 설립 날짜에 따르면 가장 최신의 컨트리 파크이기도 하다."}, 
#             {'title': '팻신렝 컨트리 파크', 'text': '팻신렝 컨트리 파크(, )는 홍콩 동북부 신구에 위치한 컨트리 파크이다. 1978년 8월 18일에 설립된 이 시골 공원은 3,125헥타르의 자연 지형을 차지하고 있다. 그것은 팻 신 렝 산맥과 웡 렝, 핑펑 샨, 흐린 언덕 그리고 콰이 타우 렝을 포함한 다른 돌기들로 구성되어 있습니다. 호크타우저수지와 라우수이흥저수지도 시골공원 내에 있다.'}], 
#         [
#             {'title': '우드게이트 밸리 컨트리 파크', 'text': '우드게이트 밸리 컨트리 파크는 버밍엄의 바틀리 그린과 퀸튼 지역에 있는 컨트리 파크입니다. 그것은 서튼 공원과 리키 힐스 컨트리 파크 다음으로 세 번째로 큰 버밍엄 컨트리 파크입니다. 이 공원은 야생동물 서식지로 유지되고 있지만 농장 동물들도 있다.'}, 
#             {'title': '킹피셔 컨트리 파크', 'text': '킹피셔 컨트리 파크는 영국에 있는 시골 공원입니다. 그것은 웨스트 미들랜즈 주 이스트 버밍엄에 위치해 있습니다. 처음에 버밍엄 시의회에 의해 프로젝트 킹피셔로 지정되었던 이 공원은 2004년 7월에 공식적으로 컨트리 공원으로 선언되었습니다. 이 컨트리파크는 M6 고속도로의 스몰 히스(버밍엄)에서 첼름슬리 우드(솔리헐)까지 11km의 리버 콜 강 연장을 따라 위치해 있습니다. 이것은 지역 자연 보호 구역입니다.'}], 
#         [   
#             {'title': '우드게이트 밸리 컨트리 파크', 'text': '우드게이트 밸리 컨트리 파크는 버밍엄의 바틀리 그린과 퀸튼 지역에 있는 컨트리 파크입니다. 그것은 서튼 공원과 리키 힐스 컨트리 파크 다음으로 세 번째로 큰 버밍엄 컨트리 파크입니다. 이 공원은 야생동물 서식지로 유지되고 있지만 농장 동물들도 있다.'}, 
#             {'title': '발록 컨트리 파크', 'text': '발록 컨트리 파크는 스코틀랜드 로몬드호의 남쪽 끝에 있는 200에이커의 컨트리 파크이다. 1980년 컨트리파크로 인정받았고, 스코틀랜드 최초의 국립공원인 로몬드호와 트로삭스 국립공원에 있는 유일한 컨트리파크이다. 발록 컨트리 파크는 자연 산책로, 안내 산책로, 벽으로 둘러싸인 정원, 그리고 로치가 보이는 피크닉 잔디밭을 특징으로 합니다. 원래 19세기 초 글래스고와 선박은행의 파트너인 존 뷰캐넌에 의해 개발되었으며 1851년 땅을 매입한 데니스툰-브라운에 의해 정원이 크게 개선되었다. Buchanan은 또한 현재 공원 방문객의 중심 역할을 하고 있는 Balloch Castle을 건설했다.'}], 
#         [   
#             {'title': '플로버 코브 컨트리 파크', 'text': '플로버 코브 컨트리 파크(, )는 홍콩 동북부 신대륙에 위치한 컨트리 파크이다. 최초의 컨트리 파크는 1978년 4월 7일에 설립되었으며, 행정 구역인 북구와 타이포 구의 4,594 헥타르의 자연 지형을 포함한다. 1979년 6월 1일 더블 헤이븐 섬과 핑차우 섬을 포함하는 확장 코브 컨트리 파크로 지정되었다.'}, 
#             {'title': '팻신렝 컨트리 파크', 'text': '팻신렝 컨트리 파크(, )는 홍콩 동북부 신구에 위치한 컨트리 파크이다. 1978년 8월 18일에 설립된 이 시골 공원은 3,125헥타르의 자연 지형을 차지하고 있다. 그것은 팻 신 렝 산맥과 웡 렝, 핑펑 샨, 흐린 언덕 그리고 콰이 타우 렝을 포함한 다른 돌기들로 구성되어 있습니다. 호크타우저수지와 라우수이흥저수지도 시골공원 내에 있다.'}], 
#         [   
#             {'title': '플로버 코브 컨트리 파크', 'text': '플로버 코브 컨트리 파크(, )는 홍콩 동북부 신대륙에 위치한 컨트리 파크이다. 최초의 컨트리 파크는 1978년 4월 7일에 설립되었으며, 행정 구역인 북구와 타이포 구의 4,594 헥타르의 자연 지형을 포함한다. 1979년 6월 1일 더블 헤이븐 섬과 핑차우 섬을 포함하는 확장 코브 컨트리 파크로 지정되었다.'}, 
#             {'title': '룽푸산 컨트리 파크', 'text': "룽푸산 컨트리 파크( , 1998년 12월 18일 설립)는 홍콩 중서구에 위치한 컨트리 파크이다. 사용하지 않는 파인우드 배터리뿐만 아니라 파인우드 가든 피크닉 지역 등 '룽푸산'의 초목이 우거진 경사면을 뒤덮어 홍콩 섬의 주택가와 상권을 조망할 수 있는 경관을 제공한다. 중층부 및 서부 지구의 주택가와 인접한 Lung Fu Shan 지역은 대중, 특히 아침 산책객과 피크닉객들이 집중적으로 이용하고 있다. 그것은 Pok Fu Lam 컨트리 공원의 북쪽에 위치해 있습니다. Lung Fu Shan 컨트리 파크의 동쪽에는 Hatton Road, 남쪽에는 Harlech Road가 있고 북쪽과 서쪽에는 Supply Department에서 건설한 덮개 도관이 있습니다. 이 컨트리 파크의 면적은 약 47헥타르로 홍콩에서 가장 작은 컨트리 파크(특별 구역 제외)이다. 이곳은 설립 날짜에 따르면 가장 최신의 컨트리 파크이기도 하다."}]]}


