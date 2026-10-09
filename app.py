from hashlib import sha256
import hmac
import json
import os
from pathlib import Path

import streamlit as st

from rag_chatbot.answering import answer_question
from rag_chatbot.documents import read_pdf, split_documents
from rag_chatbot.providers import PROVIDERS, ModelConfig, create_chat_model, create_embeddings
from rag_chatbot.retrieval import SearchIndex


ROOT = Path(__file__).parent
DEFAULT_PDF = ROOT / 'data/documents/공무원여비100문100답.pdf'
st.set_page_config(page_title='공무원 여비 문서 상담', page_icon='📚', layout='wide')


def authenticated():
    password = os.environ.get('APP_PASSWORD', '')
    if not password:
        return True
    if st.session_state.get('authenticated'):
        return True
    supplied = st.text_input('서비스 접속 비밀번호', type='password')
    if st.button('로그인'):
        if hmac.compare_digest(supplied.encode(), password.encode()):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error('비밀번호를 확인하세요.')
    return False


if not authenticated():
    st.stop()

st.title('공무원 여비 문서 상담')
st.caption('문서 근거 중심 상담 · 조건과 예외 확인 · PDF 페이지 출처 제공')
st.info('기본 자료: 인사혁신처 「공무원 여비 100문 100답」, 2022년 11월. '
        '답변은 해당 자료 기준이며 이후 개정 여부를 확인하지 않습니다. '
        '최종 지급 판단은 최신 규정과 소속기관 담당자 확인이 필요합니다.')

with st.sidebar:
    st.header('문서와 검색')
    use_default = st.checkbox('기본 여비 PDF 사용', value=DEFAULT_PDF.exists())
    uploads = st.file_uploader('추가 PDF (파일당 30MB / 500페이지 이하)', type=['pdf'],
                               accept_multiple_files=True)
    document_date = st.text_input('추가 PDF 기준일 / 발행일', placeholder='예: 2026-10-01 (모르면 비워두세요)')
    search_mode = st.selectbox('검색 방식', ['한국어 키워드 검색', '하이브리드 검색 (키워드 + 임베딩)'])
    k = st.slider('검색 근거 수', 3, 8, 5)
    embedding_config = None
    if search_mode.startswith('하이브리드'):
        st.caption('임베딩은 답변 LLM과 별도 설정입니다. 최초 색인 시 전체 문서 본문이 임베딩 서버로 전송됩니다.')
        embedding_provider = st.selectbox('임베딩 제공자', ['OpenAI', 'Ollama', 'OpenAI 호환 API'])
        embedding_model = st.text_input('임베딩 모델', value='nomic-embed-text' if embedding_provider == 'Ollama' else 'text-embedding-3-small')
        embedding_url = st.text_input('임베딩 API 주소', value='http://127.0.0.1:11434' if embedding_provider == 'Ollama' else '',
                                      placeholder='OpenAI 기본 API는 비워두세요. 호환 API는 /v1 포함')
        embedding_key_input = st.text_input('임베딩 API 키', type='password')
        embedding_key = embedding_key_input or os.environ.get('EMBEDDING_API_KEY', '')
        embedding_config = (embedding_provider, embedding_model, embedding_key, embedding_url)
    consent = st.checkbox('설정한 API 서버로 질문·문서 내용 전송에 동의', value=False)
    build = st.button('문서 색인 만들기 / 갱신', type='primary', use_container_width=True)

    st.divider()
    st.header('답변 모델')
    provider = st.selectbox('LLM 제공자', list(PROVIDERS))
    default_model, key_name = PROVIDERS[provider]
    model_name = st.text_input('모델 ID', value=default_model, key=f'model_{provider}',
                               help='사용 가능한 정확한 모델 ID를 입력하세요. 예시 모델은 계정·서버별로 다를 수 있습니다.')
    key_input = st.text_input('LLM API 키', type='password', key=f'key_{provider}') if provider != 'Ollama' else ''
    api_key = key_input or os.environ.get('LLM_API_KEY', '') or os.environ.get(key_name, '')
    base_url = ''
    if provider in ('Ollama', 'OpenAI 호환 API'):
        base_url = st.text_input('LLM API 주소', value='http://127.0.0.1:11434' if provider == 'Ollama' else '',
                                 key=f'url_{provider}', placeholder='예: https://your-server.example/v1')
    retrieval_only = st.checkbox('검색만 사용 (LLM 호출 없음)', value=not bool(api_key))
    st.caption('키는 환경 변수 또는 현재 세션에서만 사용하며 파일에 저장하지 않습니다. '
               '실제 LLM 호출에는 서버 접근 및 모델 사용 권한이 필요합니다.')
    if st.button('대화 지우기'):
        st.session_state.messages = []
        st.rerun()


files = []
if use_default and DEFAULT_PDF.exists():
    files.append((DEFAULT_PDF.name, DEFAULT_PDF.read_bytes(), '2022-11'))
for upload in uploads or []:
    files.append((Path(upload.name).name, upload.getvalue(), document_date.strip()))

signature = sha256(json.dumps({
    'files': [(name, sha256(content).hexdigest(), date) for name, content, date in files],
    'mode': search_mode,
    'embedding': embedding_config,
}, ensure_ascii=False).encode()).hexdigest()

if build:
    if not files:
        st.error('기본 문서를 선택하거나 PDF를 업로드하세요.')
    elif len(files) > 10 or sum(len(content) for _, content, _ in files) > 60 * 1024 * 1024:
        st.error('한 세션은 PDF 10개, 총 60MB까지 지원합니다.')
    elif embedding_config and not consent:
        st.error('임베딩 서버로 문서 전송에 동의해야 하이브리드 색인을 만들 수 있습니다.')
    else:
        new_index = None
        try:
            with st.spinner('문서를 읽고 검색 색인을 만드는 중입니다…'):
                pages = []
                seen = set()
                for name, content, date in files:
                    digest = sha256(content).hexdigest()
                    if digest not in seen:
                        pages.extend(read_pdf(content, name, date))
                        seen.add(digest)
                chunks = split_documents(pages)
                embeddings = create_embeddings(*embedding_config) if embedding_config else None
                new_index = SearchIndex(chunks, embeddings)
                old_index = st.session_state.get('index')
                st.session_state.index = new_index
                st.session_state.index_signature = signature
                st.session_state.index_stats = (len(seen), len(pages), len(chunks))
                st.session_state.messages = []
                if old_index is not None:
                    old_index.close()
            st.success('검색 색인을 준비했습니다.')
        except Exception as exc:
            # Do not display provider exception text: it can contain credentials or document text.
            st.error(f'색인 생성에 실패했습니다 ({type(exc).__name__}). '
                     'PDF 형식·크기, API 주소·인증, 임베딩 모델 및 네트워크 접근을 확인하세요.')

index = st.session_state.get('index')
current_index = index is not None and st.session_state.get('index_signature') == signature
if index is None:
    st.warning('왼쪽의 “문서 색인 만들기 / 갱신”을 눌러 시작하세요. 검색만 사용할 때는 API 키가 필요하지 않습니다.')
elif not current_index:
    st.warning('문서 또는 임베딩 설정이 바뀌었습니다. 색인을 갱신한 뒤 질문하세요.')
else:
    file_count, page_count, chunk_count = st.session_state.index_stats
    st.caption(f'색인 준비됨 · 문서 {file_count}개 · 텍스트 추출 {page_count}페이지 · 검색 단위 {chunk_count}개')

with st.expander('질문 예시와 사용 방법'):
    st.write('• 근무지 내 출장을 3시간 다녀오면 여비는 얼마인가요?')
    st.write('• 공용차량을 이용한 근무지 내 출장의 여비는 어떻게 감액하나요?')
    st.write('• 출장 중 식사를 제공받았을 때 식비를 전액 받을 수 있나요?')
    st.write('• 자가용 출장 연료비의 거리 계산 기준은 무엇인가요?')
    st.caption('출처 페이지는 PDF 뷰어의 1부터 시작하는 페이지 번호입니다. '
               '대화와 색인은 세션 메모리에 있으며 서버 재시작 시 사라집니다. '
               '키워드 검색은 형태소 분석 없이 문자 n-gram을 사용합니다. 의미 검색이 필요하면 하이브리드를 선택하세요.')


def render_sources(sources):
    for i, source in enumerate(sources, 1):
        metadata = source.metadata
        label = f"[S{i}] {metadata['source']} · PDF {metadata['pdf_page']}페이지"
        if metadata.get('qa_numbers'):
            label += f" · Q&A {metadata['qa_numbers']}"
        with st.expander(label):
            st.caption(f"자료 기준일: {metadata.get('source_date') or '확인되지 않음'}")
            st.text(source.page_content)


messages = st.session_state.setdefault('messages', [])
for message in messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('sources'):
            render_sources(message['sources'])

question = st.chat_input('출장 조건과 궁금한 여비 항목을 입력하세요', disabled=not current_index)
if question:
    if len(question) > 2000:
        st.error('질문은 2,000자 이내로 입력하세요.')
        st.stop()
    with st.chat_message('user'):
        st.write(question)
    history = [{'role': message['role'], 'content': message['content'][:3000]} for message in messages[-6:]]
    messages.append({'role': 'user', 'content': question})
    with st.chat_message('assistant'):
        try:
            query = question
            # Add the prior question only for short follow-ups, retaining the current query first.
            prior = next((item['content'] for item in reversed(history) if item['role'] == 'user'), '')
            if prior and len(question) < 35:
                query += '\n' + prior
            with st.spinner('문서 근거를 확인하는 중입니다…'):
                sources = index.search(query, k=k)
                if retrieval_only:
                    content = '검색된 문서 근거를 아래에서 확인하세요. 현재 검색 전용 모드로, LLM 답변을 생성하지 않았습니다.' if sources else '관련 문서 근거를 찾지 못했습니다.'
                elif not consent:
                    content = 'LLM 답변을 생성하려면 왼쪽에서 설정한 API 서버로 질문·문서 내용 전송에 동의해 주세요. 검색 근거는 아래에서 확인할 수 있습니다.'
                else:
                    model = create_chat_model(ModelConfig(provider, model_name, api_key, base_url))
                    result = answer_question(model, question, sources, history)
                    content = result.answer
                    if result.follow_up_questions:
                        content += '\n\n**추가 확인 사항**\n\n' + '\n'.join('- ' + q for q in result.follow_up_questions)
            st.markdown(content)
            render_sources(sources)
            messages.append({'role': 'assistant', 'content': content, 'sources': sources})
            # Bound per-session history; recent source excerpts remain available.
            del messages[:-40]
        except Exception as exc:
            content = (f'답변을 완료하지 못했습니다 ({type(exc).__name__}). '
                       'API 인증·모델 ID·서버 접근 및 모델의 JSON/출처 형식 지원을 확인하세요. '
                       '검증되지 않은 모델 답변은 표시하지 않습니다. 검색 전용 모드로 근거를 확인할 수 있습니다.')
            st.error(content)
            messages.append({'role': 'assistant', 'content': content})
