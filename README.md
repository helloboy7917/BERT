# BERT 구현 프로젝트

PyTorch로 BERT 인코더와 사전 학습용 헤드를 구현한 프로젝트입니다. 토큰·위치·문장 구분 임베딩, 다중 헤드 셀프 어텐션, MLM(Masked Language Modeling)과 NSP(Next Sentence Prediction) 출력을 포함합니다. 기본 설정은 BERT 원본 크기가 아닌 작은 실험용 모델입니다.

## 현재 구현 상태

- 모델, MLM·NSP 데이터 생성, 검증 및 체크포인트 재개를 포함한 사전학습 코드와 추론 코드를 제공합니다.
- 데이터 다운로드, 문장 분리 및 준비, 토크나이저 학습, 실제 사전학습과 결과 분석은 사용자가 수행합니다. 학습된 토크나이저와 모델은 별도로 준비해야 합니다.

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

CUDA를 사용한다면 [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)에서 운영체제와 CUDA 환경에 맞는 설치 명령을 선택하세요. Tokenizers 설치 방법은 [공식 문서](https://huggingface.co/docs/tokenizers/installation)를 참고하세요. `requirements.txt`에도 의존성이 명시되어 있으며 버전은 고정되어 있지 않습니다.

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

1. 아래 말뭉치 준비 단계로 `data/corpus.txt`를 생성하거나 UTF-8 텍스트를 직접 준비합니다. 이 파일은 프로젝트에 포함되어 있지 않습니다.
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

문장 분리를 마친 UTF-8 JSONL을 사용자가 준비합니다. 한 줄에 문서 하나를 담고 각 문서는 `sentences` 배열을 포함해야 합니다.

```jsonl
{"sentences": ["첫 번째 문장입니다.", "다음 문장입니다."]}
{"sentences": ["다른 문서의 첫 문장입니다.", "다른 문서의 두 번째 문장입니다."]}
{"sentences": ["세 번째 문서입니다.", "그다음 문장입니다."]}
{"sentences": ["네 번째 문서입니다.", "마지막 문장입니다."]}
```

아래 명령은 `data/documents.jsonl`과 토크나이저 학습용 `data/corpus.txt`를 생성합니다. 출력 파일이 이미 존재하면 덮어쓰므로 입력과 다른 경로를 사용하세요. `--documents`와 `--corpus`로 출력 경로를 바꿀 수 있습니다.

```powershell
python -m data.prepare_corpus path/to/raw.jsonl
```

토크나이저 학습 후 다음 명령을 사용자가 실행합니다. `--documents`와 `--tokenizer`로 입력을 변경할 수 있습니다. 기본 10%를 문서 단위로 검증용으로 분할하며 `--validation-ratio`로 변경할 수 있습니다. 각 분할은 비어 있지 않은 문서 두 개 이상과 연속 문장 두 개가 있는 문서 하나 이상을 필요로 합니다.

```powershell
python train.py --epochs 10 --batch-size 16 --learning-rate 5e-5 --max-length 256 --device auto
```

기본 장치는 CUDA 사용 가능 시 CUDA, 아니면 CPU입니다. CUDA에서 `--use-amp`를 추가하면 혼합 정밀도를 사용합니다. 체크포인트는 optimizer 업데이트마다 기본적으로 `checkpoints/last.pt`에 저장되며 `--output-dir`로 위치를 바꿀 수 있습니다. 저장 주기가 잦으므로 저장 장치 비용이 발생할 수 있습니다.

중단된 실행은 동일한 인자와 입력 파일을 사용하여 재개합니다. `--epochs`는 추가 에폭이 아니라 최초 실행의 전체 목표 에폭입니다. 마지막 저장 이후의 배치는 다시 처리됩니다. 예전 형식의 체크포인트는 추론에 사용할 수 있지만 학습 중간 재개에 필요한 상태가 없어 재개할 수 없습니다.

```powershell
python train.py --epochs 10 --batch-size 16 --learning-rate 5e-5 --max-length 256 --device auto --resume checkpoints/last.pt
```

## 추론

학습된 모델과 동일한 토크나이저를 사용합니다. 마스크 토큰 예측 또는 문장 관계 예측을 실행할 수 있습니다.

```powershell
python inference.py --checkpoint checkpoints/last.pt --tokenizer tokenizer/tokenizer.json --text "오늘은 [MASK] 날씨입니다."
python inference.py --checkpoint checkpoints/last.pt --tokenizer tokenizer/tokenizer.json --sentence-a "첫 문장입니다." --sentence-b "다음 문장입니다."
```

MLM의 `--top-k` 기본값은 5입니다. NSP 라벨은 `IsNext = 0`, `NotNext = 1`입니다. 모델 설정과 토크나이저의 어휘 크기 및 특수 토큰 ID가 일치해야 합니다.

## 테스트

프로젝트 루트에서 실행합니다.

```powershell
python -m unittest discover -s tests -v
```

모델 출력, 마스킹, 데이터 준비, 문서 분할, 추론 및 합성 모델을 이용한 중간 재개를 확인합니다. 테스트는 데이터나 모델을 다운로드하지 않고 토크나이저 학습 또는 실제 BERT 학습을 수행하지 않습니다.

## 프로젝트 구조

```text
BERT/
├── config.py                  # 모델 및 특수 토큰 설정
├── models/                    # 임베딩, 어텐션, 인코더, MLM/NSP 헤드
├── data/
│   ├── masking.py             # MLM 입력 및 정답 생성
│   ├── dataset.py             # MLM/NSP 데이터셋
│   └── prepare_corpus.py      # 로컬 JSONL 및 텍스트 준비
├── tokenizer/                 # 토크나이저 학습 및 사용
├── training/                  # 옵티마이저, 스케줄러, 학습 루프
├── tests/                     # 단위 테스트
├── train.py                   # 사전 학습 진입점
├── inference.py               # MLM/NSP 추론
└── requirements.txt           # 라이브러리 의존성
```
