## 9_Korean_StrategyQA_POSTECH
## Example

1. question decomposition model folder과 final reader model folder각각 다운받고, **./app/pretrained_model/** 폴더 안에 넣어주십시오.
* question decomposition model : https://drive.google.com/file/d/1oknVa6Du6p3gVvVbQ6UwzSwfDhmmLhqu/view?usp=sharing
* Korean model file : https://drive.google.com/file/d/15Uw75SR3yvU9-dYjSFIkg1GHxw4a9zgU/view?usp=sharing

2. passages 다운 받으시고, **./app/data/** 폴더 안에 넣어주십시오.
* passages: https://drive.google.com/file/d/17moFS0b6Q4A9BBYC10xQ_UjfOwCjBm4o/view?usp=sharing

3. 다음 커맨드로 도커 이미지를 빌드하고 실행시키십시오.
```bash
    docker build --tag stqa_postech . 
    docker run --gpus all --rm -it -p 5001:5000 --name stqa stqa_postech
```

4. test the app
```bash
    python3 test.py
```


### input 형식

|Key|Value|Explanation|
|-----|----|----------|
|question|str|질문 문장|

question은 암시적 추론을 요구하는 multi-hop 질문입니다. 
예시는 아래와 같습니다.

```
    {
        "question": "조지 워싱턴 자신의 연설을 CD에 라이브로 녹음할 수 있었습니까?"
    }

```

### output 형식
|Key|Value|Explanation|
|-----|----|-----------|
|question|str|질문|
|final_answer|str|정답|
|qap_list|List[dict]|분해된 각 질문/정답/근거 passage의 List|
|ㄴ decomposed_question|str|분해된 하위 질문|
|ㄴ answer|str|하위 질문의 정답|
|ㄴ passage|str|하위 질문에 대하여 검색된 근거단락|

이 때 qap_list의 마지막 원소에 대응되는 passage의 경우,
해당 원소의 decomposed question이 새로운 정보를 요구하는 경우에는 새롭게 검색된 근거단락,
그렇지 않은 경우에는 이전 원소들의 모든 passage가 concat된 형태로 정의됩니다.

decomposed question의 새로운 정보 요구 유무는 해당 question이 단순 수치의 비교를 요구하는 질문인 경우 X, 그렇지 않은 경우 O로 판단하였습니다.

output 형식의 예시는 다음과 같습니다.
```
{
    "question": "조지 워싱턴 자신의 연설을 CD에 라이브로 녹음할 수 있었습니까?",
    "final_answer": "아니요",
    "qap_list": [
        {
            "decomposed question": "조지 워싱턴은 언제 죽었습니까?",
            "answer": "1799년 12월 14일",
            "passage": "워싱턴의 죽음은 예상보다 더 빨리 찾아왔다. 임종 당시 그는 자신의 비서인 토비아스 리어에게 생매장을 두려워하여 매장되기 3일 전에 기다리라고 지시했습니다. Lear에 따르면 그는 1799년 12월 14일 토요일 밤 10시에서 11시 사이에 침대 발치에 앉아 있는 Martha와 함께 평화롭게 사망했습니다. 그의 마지막 말은 그의 매장에 대해 Lear와의 대화에서 나온 &quot;&#39;Tis well&quot;이었습니다. 그는 67세였다."
        },
        {
            "decomposed question": "CD는 언제 발명되었습니까?",
            "answer": "1982년",
            "passage": "1982년 기술 도입 당시 CD는 일반적으로 10MB를 저장할 수 있는 개인용 컴퓨터 하드 드라이브보다 훨씬 더 많은 데이터를 저장할 수 있었습니다. 2010년까지 하드 드라이브는 일반적으로 CD 천 개만큼의 저장 공간을 제공했지만 가격은 상품 수준으로 떨어졌습니다. 2004년에는 오디오 CD, CD-ROM 및 CD-R의 전 세계 판매량이 약 300억 개에 달했습니다. 2007년까지 전 세계적으로 2,000억 장의 CD가 판매되었습니다."
        },
        {
            "decomposed question": "1982년이 1799년 12월 14일보다 먼저인가?",
            "answer": "아니요",
            "passage": "워싱턴의 죽음은 예상보다 더 빨리 찾아왔다. 임종 당시 그는 자신의 비서인 토비아스 리어에게 생매장을 두려워하여 매장되기 3일 전에 기다리라고 지시했습니다. Lear에 따르면 그는 1799년 12월 14일 토요일 밤 10시에서 11시 사이에 침대 발치에 앉아 있는 Martha와 함께 평화롭게 사망했습니다. 그의 마지막 말은 그의 매장에 대해 Lear와의 대화에서 나온 &quot;&#39;Tis well&quot;이었습니다. 그는 67세였다. 1982년 기술 도입 당시 CD는 일반적으로 10MB를 저장할 수 있는 개인용 컴퓨터 하드 드라이브보다 훨씬 더 많은 데이터를 저장할 수 있었습니다. 2010년까지 하드 드라이브는 일반적으로 CD 천 개만큼의 저장 공간을 제공했지만 가격은 상품 수준으로 떨어졌습니다. 2004년에는 오디오 CD, CD-ROM 및 CD-R의 전 세계 판매량이 약 300억 개에 달했습니다. 2007년까지 전 세계적으로 2,000억 장의 CD가 판매되었습니다."
        }
    ]
}```
