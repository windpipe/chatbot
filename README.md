# 공무원 여비 문서 상담 RAG

LangChain + Streamlit 기반 PDF 상담 서비스입니다. 기본 자료는 인사혁신처
**「공무원 여비 100문 100답」(2022년 11월)**입니다. 자료가 현행 규정인지 자동 검증하지 않습니다.
PDF 페이지 출처, 조건·예외 설명, 근거 부족 시 답변 보류를 지원합니다.

## 빠른 실행

Python 3.12와 `uv`를 사용합니다. 클라우드에 준비된 환경:

```bash
cd /workspace/chatbot
source /workspace/rag-environment/.venv/bin/activate
python -m streamlit run app.py
```

새 환경에서 설치:

```bash
uv venv .venv --python 3.12
uv pip sync --python .venv/bin/python requirements.lock
source .venv/bin/activate
python -m streamlit run app.py
```

Windows PowerShell에서 소스 폴더로 이동한 뒤 설치·실행:

```powershell
uv venv .venv --python 3.12
uv pip sync --python .venv/Scripts/python.exe requirements.lock
.\.venv\Scripts\python.exe -m streamlit run app.py
```

`requirements.lock`은 Python 3.12 기준으로 모든 플랫폼에 대한 조건을 포함합니다.
Windows에서는 `uvloop`가 제외되고 Windows 전용 의존성이 설치됩니다.
의존성을 갱신할 때는 플랫폼 조건이 사라지지 않도록 다음 명령을 사용하세요:

```bash
uv pip compile requirements.in --python-version 3.12 --universal --generate-hashes --output-file requirements.lock
```

기본 PDF를 `data/documents/공무원여비100문100답.pdf`에 두거나 화면에서 PDF를 업로드하세요.
문서와 비밀정보는 Git에서 제외합니다. GitHub에서 소스를 복제해도 PDF는 포함되지 않습니다.
기본 PDF는 현재 클라우드 파일시스템에 준비되어 있습니다.

1. 문서를 선택하고 **문서 색인 만들기 / 갱신**을 누릅니다.
2. API 키 없이 키워드 검색으로 문서 근거를 확인할 수 있습니다.
3. 답변 LLM 제공자와 모델 ID, 필요한 API 키·주소를 설정합니다.
4. API 전송 동의 후 **검색만 사용**을 해제하고 질문합니다.
5. 출처 펼침에서 실제 문구와 PDF 페이지를 확인합니다.

## 모델 연결

| 제공자 | 설정 | 비고 |
| --- | --- | --- |
| OpenAI | 모델 ID + API 키 | 기본 OpenAI 주소 |
| Anthropic | 모델 ID + API 키 | Claude 계열 |
| Google Gemini | 모델 ID + API 키 | Gemini 계열 |
| Ollama | 모델 ID + 서버 주소 | 별도 Ollama 서버와 모델 준비 필요 |
| OpenAI 호환 API | 모델 ID + API 키 + `/v1`을 포함한 base URL | vLLM, 각종 호환 게이트웨이 등 |

예시 모델 ID는 권한과 서버 버전에 따라 달라지므로 실제 사용 가능한 ID로 바꾸세요.
“모든 LLM”은 단일 공통 API가 존재하지 않습니다. 위 어댑터와 호환 API로 넓게 연결하며,
추가 프로토콜은 `rag_chatbot/providers.py`에 LangChain `BaseChatModel` 어댑터를 추가할 수 있습니다.
모델이 JSON 답변·본문 출처 형식을 따르지 않으면 검증에서 거부합니다.
어댑터 설치 및 생성 검증과 실제 유료 모델 호출 검증은 다릅니다.

API 키는 사이드바에 세션 한정으로 입력하거나 환경 변수로 주입합니다.
`LLM_API_KEY`가 우선이며 제공자별 `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`도 지원합니다.
임베딩 키는 별도의 `EMBEDDING_API_KEY`를 사용합니다. 실제 키를 저장소·로그·스크립트에 넣지 마세요.
`.env`는 자동 로딩하지 않습니다. 클라우드에서는 환경 설정의 비밀정보 바인딩을 이용하세요.
전송 동의 후 질문과 검색 문맥이 선택한 모델 서버로 전송됩니다.

Ollama 예시 (이미 준비된 별도 서버에서 실행):

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
ollama serve
```

현재 클라우드에는 Ollama 서버나 모델을 설치하지 않았습니다. 로컬 주소는 **Streamlit 서버가
실행되는 머신**을 가리키며 방문자의 PC를 가리키지 않습니다. 외부 서버는 HTTPS로 연결하세요.

## 검색과 근거

- 기본: 한국어 문자 2/3-gram을 포함한 BM25. API 비용 없이 검색 가능.
- 선택: OpenAI / Ollama / OpenAI 호환 임베딩 + Chroma + BM25 결과의 RRF 결합.
- 임베딩 모델은 답변 모델과 별도로 선택합니다. 외부 임베딩 사용 시 전체 문서 본문이 전송됩니다.
- 1,100자/180자 중복 분할, PDF 페이지·자료 기준일·Q&A 번호 메타데이터를 유지합니다.
- 목차와 짧은 표지 페이지는 제외합니다. 스캔 문서는 별도 OCR이 필요합니다.
- 검색 색인과 대화는 세션 메모리에 유지됩니다. 문서/임베딩 변경 시 재색인이 필요합니다.
- 문서 내용은 프롬프트 명령으로 취급하지 않습니다. 모델 응답은 JSON 스키마와 출처 ID를 검증합니다.
  이 검증은 의미적 정확성을 보장하지 않습니다. 지급액·예외·최신 법령은 원문과 담당자 확인이 필요합니다.

## 검증

```bash
python -m pytest -q
curl --fail http://127.0.0.1:8501/_stcore/health
```

테스트는 실제 PDF 검색(기본 파일이 있을 때), 문서 처리, 잘못된 출처 차단,
모델 어댑터 생성, 가짜 LLM을 통한 전체 답변 흐름, Streamlit 화면 흐름을 검증합니다.
실제 외부 모델의 답변 품질·비용·인증 및 임베딩 호출은 해당 서비스 연결 후 별도로 검증해야 합니다.

## 서비스 배포

### Streamlit Community Cloud

GitHub 저장소 `windpipe/chatbot`, 브랜치 `main`, 시작 파일 `app.py`를 선택하고
배포 설정에서 Python **3.12**를 선택하세요. Cloud는 `requirements.txt`를 자동 감지해
그 파일이 참조하는 `requirements.lock`으로 의존성을 설치합니다.
잠금 파일은 Python 3.12를 기준으로 생성·검증했습니다.

Manage app의 Secrets에 `APP_PASSWORD`와 선택한 제공자의 `LLM_API_KEY`를
TOML 문자열 항목으로 안전하게 설정할 수 있습니다. 앱은 환경 변수를 먼저 읽고,
값이 없으면 Streamlit Secrets를 읽습니다. 실제 비밀정보를 GitHub에 올리지 마세요.
키를 이용하는 제공자와 모델 ID·API 주소는 사이드바에서 선택하고 데이터 전송에 동의해야 합니다.
기본 PDF는 GitHub에 없으므로 배포 후 업로드하고 색인을 만드세요.
문서 업로드와 색인은 세션별이며 서버 재시작 시 복구되지 않습니다.
이 형태는 데모 배포용이며 아래의 기관 운영 요건을 충족하는 정식 서비스는 아닙니다.

### 자체 서버

현재 설정은 localhost 개발용입니다. 외부 공개 전 `APP_PASSWORD`를 안전하게 주입하고,
인증 및 HTTPS를 제공하는 프록시 뒤에서 실행하세요:

```bash
python -m streamlit run app.py --server.address 0.0.0.0
```

내장 비밀번호는 단일 공유 비밀번호 방식입니다. 기관 운영에는 SSO·사용자별 권한·감사 로그·
속도 제한·문서 접근 제어와 운영 모니터링을 추가해야 합니다. 공개 사용자가 임의 API 주소나
문서를 설정할 수 있는 형태로 배포하지 말고, 해당 옵션을 관리자 설정으로 제한하세요.
테스트용 API 키나 파일을 소스에 포함하지 마세요.

Git 원격은 `windpipe/chatbot`입니다. 문서 PDF와 실제 비밀정보는 업로드 대상에서 제외합니다.
