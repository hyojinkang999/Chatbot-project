# Chatbot project   
  
* [개요](#1.-개요)  
* [개발 환경](#2.-개발-환경)  
* [데이터셋](#3.-데이터셋)
* [전처리](#4.-데이터-전처리)  
* [모델 학습 및 예측](#5.-모델-학습-및-예측)  
    * [Transformer](#5-1.-Transformer)
    * [KoGPT2](#5-2.-GPT2)
    * [KoBART](#5-3.-BART)
* [모델 선정](#6.-모델-선정)
* [Demo](#7.-Demo)  
* [멤버](#8.-멤버)     
  
  
  
  
## 1. 개요 
  
최근 5년(2017~2021년), 우울증과 불안 장애 진료 현황 분석 결과, 환자 수가 급증하는 추세   
  
<img width="500" alt="스크린샷 2022-12-28 오후 3 49 31" src="https://user-images.githubusercontent.com/114709620/209770654-0a217608-820a-4ddc-bcf9-cb02f0622ae7.png">
  
**⇢ 비대면 심리 치료로 우울감과 불안감을 조금이나마 덜고자 심리 케어 AI 챗봇 구현**  
  
<img width="500" alt="스크린샷 2022-12-28 오후 3 52 14" src="https://user-images.githubusercontent.com/114709620/209771578-30461e81-01b1-4e6f-8fdb-fcee25606e08.png">
  

  
## 2. 개발 환경

     Google Colab Pro
     
     huggingface-hub==0.10.1
     datasets==2.6.1
     tokenizers==0.13.2
     torch==1.12.1+cu113
     torchvision==0.13.1+cu113
     transformers==4.24.0
     tqdm==4.64.1
     scikit-learn==1.0.2
     sentencepiece==0.1.97


  
## 3. 데이터셋  
  
  - 출처 :  
  
    - [AI Hub 웰니스 대화 스크립트 데이터셋](https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-006)
    - [AI Hub 감성 대화 말뭉치](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86)  
  
  - 질의 별 3개 내외의 답변 존재 
  - 육체・정신적 불안감과 우울감을 담은 30,000개의 대화 쌍    
  
  
<img width="653" alt="스크린샷 2022-12-28 오후 4 04 18" src="https://user-images.githubusercontent.com/114709620/209772265-4fb47a0c-e6b2-49d1-af96-e234a74b5c09.png">  
  
  
<img width="208" alt="대화 내용의 범주 " src="https://user-images.githubusercontent.com/114709620/209914011-21a576ad-a10e-4b0d-b73d-784aad6bfe0c.png">    
  
  
  
## 4. 데이터 전처리  
  
#### 4-1. 결측치 제거  
    답변이 NaN인 대화 쌍 제거   
    
#### 4-2. 맞춤법 교정  
    py-hanspell(맞춤법 검사기) 사용  
  
#### 4-3. 구두점 분리  
    Tokenizing을 위해 구두점(.,?!)과 텍스트 분리    
  
#### 4-4. 띄어쓰기 교정  
    pykospacing 패키지 사용      
  
    
  
<img width="756" alt="스크린샷 2022-12-29 오후 3 56 44" src="https://user-images.githubusercontent.com/114709620/209915295-40d51cc9-8908-4854-a2a8-bf6b01a77f66.png">    
  
  
  
## 5. 모델 학습 및 예측    

Model | Pretraind Model | Code
-------|----------------|------|
Transformer | Transformer 라이브러리  | code
GPT2 | [skt/kogpt2-base-v2](https://huggingface.co/skt/kogpt2-base-v2) | code 
BART | [cosmoquester/bart-ko-small](https://huggingface.co/cosmoquester/bart-ko-small) | code   
  
  
  
#### 5-1. Transformer  
  
- BART, BERT, GPT2의 기반이 되는 Encoder-Decoder Model  
     
    <img width="400" alt="스크린샷 2022-12-30 오후 3 17 50" src="https://user-images.githubusercontent.com/114709620/210040558-fb4d233d-7c60-405c-b45e-b9fdfd404cb9.png">    
  
  
  
  
- 하이퍼 파라미터:  
  
        Num_layers = 6  
        D_model = 512  
        Num_heads = 8  
        DFF = 1024  
        Dropout = 0.2    
  
  
  
  
- 예측 결과:  
  
    <img width="567" alt="스크린샷 2022-12-30 오후 3 21 39" src="https://user-images.githubusercontent.com/114709620/210040784-e221fe45-73f0-4907-b23e-cf93aea6cb29.png">    
  
  
  
#### 5-2. GPT2  
  
- 순차적 예측 학습을 통해 텍스트 생성에 강점을 지님  
  
    <img width="400" alt="스크린샷 2022-12-30 오후 3 22 35" src="https://user-images.githubusercontent.com/114709620/210040830-c0e60491-a599-4ffc-b5a7-a2f1c552943d.png">    
   
  
  
- Pretrained model: "skt/kogpt2-base-v2"  
  
  
  
  
- 하이퍼 파라미터:  
  
        do_sample = True    
        top_p = 0.92  
        top_k = 3  
        temperature = 0.8  
  
  
  
  
- 예측 결과:  
  
    <img width="567" alt="스크린샷 2022-12-30 오후 3 25 03" src="https://user-images.githubusercontent.com/114709620/210040983-64e6e012-cc73-4699-846c-4987f322d55a.png">      
  
    <img width="567" alt="스크린샷 2022-12-30 오후 3 25 34" src="https://user-images.githubusercontent.com/114709620/210041012-17689509-ae1c-4360-b5ba-92a3f49d7c55.png">      
      
    
  
#### 5-3. BART  
  
- BERT와 GPT2를 합친 denoising auto-encoder이므로 문장생성에 용이    
  
    <img width="270" alt="스크린샷 2022-12-30 오후 3 28 18" src="https://user-images.githubusercontent.com/114709620/210041227-44ffb62f-10b9-4226-9837-fd9fa8642360.png">  

  
  
 
- Pretrained model: "cosmoquester/bart-ko-small"  
  
  
  
  
- 예측 결과    
  
    <img width="567" alt="스크린샷 2022-12-30 오후 3 28 59" src="https://user-images.githubusercontent.com/114709620/210041313-22587a7f-aa63-4467-9eda-a345410efcf0.png">    
  
  
  
## 6. 모델 선정   
 
- Text generation task이기 때문에 동일한 단어 및 순서 기준으로 판단하는 정량 지표는 적합하지 않음 
- 질문과 답변의 정합성, 구체성 등을 고려한 정성 평가 수행 결과, **KoGPT2 채택**   
  
    
  <img width="567" alt="스크린샷 2022-12-30 오후 3 57 03" src="https://user-images.githubusercontent.com/114709620/210043307-fefe79d0-b556-4a74-9774-2251a37e2f73.png">

  

## 7. Demo  
  
```python
!pip install gradio

import gradio as gr

iface = gr.Interface(
    fn = return_answer_by_chatbot,    # inference 함수명 입력
    inputs = gr.inputs.Textbox(lines = 1, placeholder = "힐링이에게 하고싶은 말을 적으세요."),
    outputs = "text")
iface.launch()  
```    
    
  
  
<img width="1173" alt="스크린샷 2022-12-30 오후 5 05 59" src="https://user-images.githubusercontent.com/114709620/210048461-a33cdaaf-8e43-494f-91b3-a07aac919951.png">



## 8. 멤버



  
---

- ## Model Load: Hugging Face에서 Pre-Trained Model 불러오기 ( pip install transformers )
   
   
    <table>
    <thead>
        <tr>
            <th>목록</th>
            <th>Model</th>
            <th>Pre-Trained Data</th>
            <th>링크(HuggingFace)</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>속성범주 (Category)</td>
            <td> ELECTRA</td>
            <td> 한국어로 된 블로그, 댓글, 리뷰 Data </td>
            <td>
                <a href="https://huggingface.co/kykim/electra-kor-base">kykim/electra-kor-base</a>
        </tr>
        <tr>
            <td> RoBERTa</td>
            <td> Wikipedia, BookCorpus, CommonCrawl data 등 100 languagues로 된 Data </td>
            <td>
                <a href="https://huggingface.co/xlm-roberta-base">xlm-roberta-base</a>
        </tr>
        <tr>
            <td>DeBERTa</td>
            <td>한국어로 된 모두의 말뭉치, 국민청원 등의 Data </td>
            <td>
                <a href="https://huggingface.co/lighthouse/mdeberta-v3-base-kor-further">mdeberta-v3-base-kor-further</a>
        </tr>
        <tr>
            <td>감성범주 (Polarity)</td>
            <td> ELECTRA</td>
            <td> 한국어로 된 블로그, 댓글, 리뷰 Data </td>
            <td>
                <a href="https://huggingface.co/kykim/electra-kor-base">kykim/electra-kor-base</a>
        </tr>
    </tbody>
    </table>
    
   > 속성 범주(Category)와 감성 범주(Polarity의 class 불균형을 해소하기 위해서 각 범주를 분리하여 전처리 및 학습을 진행하였다.     
    
   ```c
    # HuggingFace에서 불러오기
    from transformers import AutoTokenizer, AutoModel
    base_model = "HuggingFace주소"

    Model = AutoModel.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
   ```


- ## Data Load: jsonlload
    데이터가 line별로 저장된 json 파일( jsonl )이기 때문에 데이터 로드를 할 때 해당 코드로 구현함

    ```c
    import json
    import pandas as pd
    def jsonlload(fname, encoding="utf-8"):
        json_list = []
        with open(fname, encoding=encoding) as f:
            for line in f.readlines():
                json_list.append(json.loads(line))
        return json_list
    df = pd.DataFrame(jsonlload('/content/sample.jsonl'))
    ```


- ## Inference: predict_from_korean_form
   predict_from_korean_form 6가지의 방법 중 일부
   
   > [HappyBusDay/Korean_ABSA/code/test.ipynb 참조](https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/test.ipynb)

    - #### 방법 1: Force ( Force evaluation of a Argument )
    
         빈칸 " [ ] " 에 대해서 가장 높은 확률의 카테고리를 강제로 뽑아내는 방법 

         [Force에 대한 설명](https://rdrr.io/r/base/force.html)

         ```c

        def predict_from_korean_form_kelec_forcing(tokenizer_kelec, ce_model, pc_model, data):

            ...

            자세한 코드는 code/test.ipynb 참조

            return data
         ```

 
    - #### 방법 2: DeBERTa(RoBERTa)와 ELECTRA 
        
         모델 별 tokenizer를 이용한 inference 진행하는 방법

         ```c

        def predict_from_korean_form_deberta(tokenizer_deberta, tokenizer_kelec, ce_model, pc_model, data):

            ...

           자세한 코드는 code/test.ipynb 참조

            return data
         ```

     
    - #### 방법 3: Threshold
     
         확률 기반으로 annotation을 확실한 것만 가져오는 방법
         
         확실한 것만 잡고 확률값이 낮은 것은 그냥 " [ ] "으로 결과값 도출 

         ```c

        def predict_from_korean_form_kelec_threshold(tokenizer_kelec, ce_model, pc_model, data):

            ...

           자세한 코드는 code/test.ipynb 참조

            return data
         ```



- ## Pipeline 및 Ensemble

   > [HappyBusDay/Korean_ABSA/code/test.ipynb 참조](https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/test.ipynb)
   
    - #### Pipeline: 여러 모델을 불러 결과값 도출
        
        해당 코드는 **12종류[category{6종류} + polarity{6종류}]의 모델**을 불러옴

        " [ ] " 을 최소화 하기 위해 DeBERTa와 ELECTRA 등 여러 모델의 Weight파일을 불러 진행

        ```c
        def Win():

            print("Deberta!!")

            tokenizer_kelec = AutoTokenizer.from_pretrained(base_model_elec)
            tokenizer_deberta = AutoTokenizer.from_pretrained(base_model_deberta)
            tokenizer_roberta = AutoTokenizer.from_pretrained(base_model_roberta)

            num_added_toks_kelec = tokenizer_kelec.add_special_tokens(special_tokens_dict)
            num_added_toks_deberta = tokenizer_deberta.add_special_tokens(special_tokens_dict)
            num_added_toks_roberta = tokenizer_roberta.add_special_tokens(special_tokens_dict)

            ...    

            자세한 코드는 code/test.ipynb 참조

            return pd.DataFrame(jsonlload('/content/drive/MyDrive/Inference_samples.jsonl'))
        ```
    
    
    - #### Ensemble: 위의 Inference의 결과로 만들어진 jsonl파일을 불러와 Hard Voting을 진행
        > [Ensemble.ipynb 참조](https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/Ensemble.ipynb)

        > [Auto_Ensemble.ipynb 참조](https://github.com/HappyBusDay/Korean_ABSA/blob/main/code/Auto_Ensemble.ipynb)

       
        <img width="450" alt="KakaoTalk_20221113_222631386" src="https://user-images.githubusercontent.com/73925429/201582648-93ae75da-affe-4198-83a5-fb5280c54bdd.png">

        ( Hard Voting )

     


---

# 마. Reference

[1] [EDA: Easy Data Augmentation](https://arxiv.org/pdf/1901.11196.pdf): Wei, Jason, and Kai Zou. "Eda: Easy data augmentation techniques for boosting performance on text classification tasks." arXiv preprint arXiv:1901.11196 (2019).

[2] [Back-Trainslation](https://proceedings.neurips.cc/paper/2020/file/44feb0096faa8326192570788b38c1d1-Paper.pdf): Xie, Qizhe, et al. "Unsupervised data augmentation for consistency training." Advances in Neural Information Processing Systems 33 (2020): 6256-6268.

[3] [ELECTRA](https://arxiv.org/pdf/2003.10555.pdf): Clark, Kevin, et al. "Electra: Pre-training text encoders as discriminators rather than generators." arXiv preprint arXiv:2003.10555 (2020).

[4] [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf): Liu, Yinhan, et al. "Roberta: A robustly optimized bert pretraining approach." arXiv preprint arXiv:1907.11692 (2019).

[5] [DeBERTa](https://arxiv.org/pdf/2006.03654.pdf): He, Pengcheng, et al. "Deberta: Decoding-enhanced bert with disentangled attention." arXiv preprint arXiv:2006.03654 (2020).

[6] [teddysum/korean_ABSA_baseline](https://github.com/teddysum/korean_ABSA_baseline): GitHub

[7] [catSirup/KorEDA](https://github.com/catSirup/KorEDA): GitHub

---

# 바. Members
Yongjae Kim | dydwo322@naver.com<br>
Hyein Oh | gpdls741@naver.com<br>
Seungyong Guk | kuksy77@naver.com<br>
Jaehyeog Lee | tysl4545@naver.com<br>
Hyoje Jung | flash1253@naver.com<br>
Hyojin Kang | khj94111@gmail.com
