# BERT 구현 프로젝트

PyTorch로 BERT 인코더와 사전 학습용 헤드를 구현한 프로젝트입니다. 토큰·위치·문장 구분 임베딩, 다중 헤드 셀프 어텐션, MLM(Masked Language Modeling)과 NSP(Next Sentence Prediction) 출력을 포함합니다. 기본 설정은 BERT 원본 크기가 아닌 작은 실험용 모델입니다.

## 현재 구현 상태

- **사용 가능:** 모델 생성과 순전파, MLM 마스킹, WordPiece 토크나이저 학습, 단위 테스트
- **미완성:** `data/dataset.py`의 데이터셋, `data/prepare_corpus.py`의 전처리, `inference.py`의 추론 실행 코드
- `train.py`와 학습 루프는 작성되어 있지만, `BERTPretrainingDataset`이 아직 없어 **현재 전체 사전 학습을 실행할 수 없습니다**. 학습된 모델과 토크나이저도 별도로 준비해야 합니다.

## 실행 환경과 설치

- Python 3.11 환경을 기준으로 안내합니다. 모델 단위 테스트는 Python 3.11.5와 PyTorch 2.13.0 환경에서 확인했습니다.
- 필요한 외부 라이브러리는 **PyTorch (`torch`)**와 **Hugging Face Tokenizers (`tokenizers`)**입니다. `unittest`, `json` 등은 Python 표준 라이브러리입니다.
- CPU로 모델과 테스트를 실행할 수 있습니다. CUDA 학습을 하려면 호환되는 NVIDIA GPU, 드라이버와 CUDA 지원 PyTorch 빌드가 필요합니다.

프로젝트 루트에서 다음을 실행합니다. 아래 가상 환경 활성화 명령은 Windows PowerShell 기준입니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch tokenizers
```

CUDA를 사용한다면 [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)에서 운영체제와 CUDA 환경에 맞는 설치 명령을 선택하세요. Tokenizers 설치 방법은 [공식 문서](https://huggingface.co/docs/tokenizers/installation)를 참고하세요. 라이브러리 버전은 고정되어 있지 않으며, 현재 `requirements.txt`는 비어 있으므로 `pip install -r requirements.txt`만으로는 필요한 라이브러리가 설치되지 않습니다.

설치 및 CUDA 인식 여부 확인:

```powershell
python -c "import torch, tokenizers; print('torch:', torch.__version__); print('tokenizers:', tokenizers.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## 모델 실행 예제

프로젝트 루트에서 아래 코드를 Python으로 실행하면 임의의 토큰 ID를 사용하여 모델 출력 크기를 확인할 수 있습니다. 이 모델은 무작위로 초기화되므로 결과는 의미 있는 예측이 아닙니다.

```python
import torch

from config import BERTConfig
from models.bert import BERTForPreTraining

config = BERTConfig(
    vocab_size=128,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=128,
    max_position_embeddings=32,
)
model = BERTForPreTraining(config)
model.eval()

# [CLS] 토큰 1, 일반 토큰 10과 11, [SEP] 토큰 2, [PAD] 토큰 0
input_ids = torch.tensor([[1, 10, 11, 2, 0]], dtype=torch.long)
attention_mask = (input_ids != config.pad_token_id).long()
token_type_ids = torch.zeros_like(input_ids)

with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

print(outputs["sequence_output"].shape)         # [1, 5, 64]
print(outputs["pooled_output"].shape)           # [1, 64]
print(outputs["prediction_scores"].shape)       # [1, 5, 128]
print(outputs["seq_relationship_score"].shape)  # [1, 2]
```

입력 텐서는 `[배치 크기, 시퀀스 길이]` 형태입니다. `attention_mask`는 실제 토큰에 1, 패딩에 0을 사용합니다. 생략하면 모든 위치를 실제 토큰으로 취급하므로 패딩이 있다면 직접 전달해야 합니다. `token_type_ids`는 문장 구분 ID이며 생략하면 모두 0입니다. 출력의 `attention_probs`에는 인코더 층별 어텐션 값이 들어 있습니다.

## 토크나이저 학습

1. UTF-8 텍스트 학습 말뭉치를 `data/corpus.txt`에 직접 준비합니다. 이 파일은 현재 프로젝트에 포함되어 있지 않습니다.
2. 프로젝트 루트에서 다음 명령을 실행합니다.

   ```powershell
   python -m tokenizer.train_tokenizer
   ```

3. 학습된 WordPiece 토크나이저가 `tokenizer/tokenizer.json`에 저장됩니다. 기본 설정은 목표 어휘 크기 8,000과 최소 출현 빈도 2이며, 실제 어휘 크기는 말뭉치에 따라 달라질 수 있습니다.

생성된 토크나이저는 다음과 같이 사용할 수 있습니다.

```python
from tokenizer.tokenizer import BertTokenizer

tokenizer = BertTokenizer("tokenizer/tokenizer.json")
encoded = tokenizer.encode("안녕하세요.", "반갑습니다.")
print(encoded)  # input_ids, token_type_ids, attention_mask
print(tokenizer.decode(encoded["input_ids"]))
```

토크나이저의 어휘 크기와 특수 토큰 ID는 모델의 `BERTConfig`와 일치해야 합니다. `train.py`는 토크나이저에서 이 값을 읽어 설정하지만, 직접 모델을 만들 때는 해당 값을 직접 전달하세요.

## 사전 학습 준비

데이터셋 구현이 완성되면 `train.py`는 `tokenizer/tokenizer.json`과 `data/documents.jsonl`을 사용하도록 작성되어 있습니다. JSONL은 한 줄에 문서 하나를 담고, 각 문서는 `sentences` 배열을 포함해야 합니다.

```jsonl
{"sentences": ["첫 번째 문장입니다.", "다음 문장입니다."]}
{"sentences": ["다른 문서의 첫 문장입니다.", "다른 문서의 두 번째 문장입니다."]}
```

학습기는 배치에서 `input_ids`, `attention_mask`, `token_type_ids`, `mlm_labels`, `nsp_label`을 기대합니다. MLM 정답이 아닌 위치는 `-100`으로 표시합니다. 데이터셋과 입력 파일이 준비된 뒤 실행할 명령의 예는 다음과 같습니다. **현재는 데이터셋이 미구현이므로 이 명령이 실패합니다.**

```powershell
python train.py --epochs 10 --batch-size 16 --learning-rate 5e-5 --max-length 256 --device auto
```

기본 장치는 CUDA 사용 가능 시 CUDA, 아니면 CPU입니다. CUDA에서 `--use-amp`를 추가하면 혼합 정밀도를 사용합니다. 학습이 가능해진 후 체크포인트는 기본적으로 `checkpoints/last.pt`에 저장되며 `--output-dir`로 위치를 바꿀 수 있습니다.

## 테스트

프로젝트 루트에서 실행합니다.

```powershell
python -m unittest discover -s tests -v
```

모델 출력 형태, 패딩 어텐션, 가중치 공유와 마스킹 및 손실 계산을 확인합니다.

## 프로젝트 구조

```text
BERT/
├── config.py                  # 모델 및 특수 토큰 설정
├── models/                    # 임베딩, 어텐션, 인코더, MLM/NSP 헤드
├── data/
│   ├── masking.py             # MLM 입력 및 정답 생성
│   ├── dataset.py             # 미구현
│   └── prepare_corpus.py      # 미구현
├── tokenizer/                 # 토크나이저 학습 및 사용
├── training/                  # 옵티마이저, 스케줄러, 학습 루프
├── tests/                     # 단위 테스트
├── train.py                   # 사전 학습 진입점 (데이터셋 필요)
├── inference.py               # 미구현
└── requirements.txt           # 현재 비어 있음
```
