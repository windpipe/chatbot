from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def clear_credentials(monkeypatch):
    for key in ('APP_PASSWORD', 'LLM_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY'):
        monkeypatch.delenv(key, raising=False)


def test_streamlit_index_search_and_reset(monkeypatch):
    clear_credentials(monkeypatch)
    if not (ROOT / 'data/documents/공무원여비100문100답.pdf').exists():
        pytest.skip('기본 PDF가 필요합니다.')
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == '공무원 여비 문서 상담'
    assert app.chat_input[0].disabled
    next(button for button in app.button if button.label == '문서 색인 만들기 / 갱신').click().run()
    assert not app.exception
    assert not app.chat_input[0].disabled
    app.chat_input[0].set_value('자가용 출장 연료비의 거리 계산 기준은 무엇인가요?').run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert 'LLM 답변을 생성하지 않았습니다' in app.session_state['messages'][-1]['content']
    assert app.session_state['messages'][-1]['sources'][0].metadata['pdf_page'] == 21
    next(button for button in app.button if button.label == '대화 지우기').click().run()
    assert app.session_state['messages'] == []
    app.checkbox[0].uncheck().run()
    assert app.chat_input[0].disabled


def test_password_gate(monkeypatch):
    clear_credentials(monkeypatch)
    monkeypatch.setenv('APP_PASSWORD', 'fixture-password')
    app = AppTest.from_file(str(ROOT / 'app.py')).run()
    assert not app.exception
    assert not app.chat_input
    app.text_input[0].set_value('wrong')
    app.button[0].click().run()
    assert app.error
    app.text_input[0].set_value('fixture-password')
    app.button[0].click().run()
    assert not app.exception
    assert app.title[0].value == '공무원 여비 문서 상담'


def test_cloud_secrets_password_gate(monkeypatch):
    clear_credentials(monkeypatch)
    app = AppTest.from_file(str(ROOT / 'app.py'))
    app.secrets['APP_PASSWORD'] = 'fixture-cloud-password'
    app.run()
    assert not app.exception
    assert not app.chat_input
    app.text_input[0].set_value('fixture-cloud-password')
    app.button[0].click().run()
    assert not app.exception
    assert app.title[0].value == '공무원 여비 문서 상담'
