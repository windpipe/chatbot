from dataclasses import dataclass
from urllib.parse import urlparse


PROVIDERS = {
    'OpenAI': ('gpt-4.1-mini', 'OPENAI_API_KEY'),
    'Anthropic': ('claude-sonnet-4-5', 'ANTHROPIC_API_KEY'),
    'Google Gemini': ('gemini-2.5-flash', 'GOOGLE_API_KEY'),
    'Ollama': ('qwen3:8b', ''),
    'OpenAI 호환 API': ('', 'COMPATIBLE_API_KEY'),
}


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    api_key: str = ''
    base_url: str = ''
    temperature: float = 0.0
    max_tokens: int = 2000


def validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError('API 주소는 http:// 또는 https://로 시작해야 합니다.')
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('API 주소에 인증정보, 쿼리 또는 fragment를 포함하지 마세요.')
    if parsed.scheme == 'http' and parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('외부 API는 HTTPS를 사용하세요. HTTP는 로컬 서버만 허용됩니다.')
    return url.rstrip('/')


def create_chat_model(config: ModelConfig):
    if config.provider not in PROVIDERS:
        raise ValueError('지원하지 않는 제공자입니다. OpenAI 호환 API 어댑터를 사용하세요.')
    if not config.model.strip():
        raise ValueError('모델 이름을 입력하세요.')
    if config.provider != 'Ollama' and not config.api_key:
        raise ValueError('API 키를 설정하세요. 인증이 없는 로컬 서버는 Ollama를 사용하세요.')
    common = dict(model=config.model, temperature=config.temperature)
    if config.provider == 'Anthropic':
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(**common, api_key=config.api_key, max_tokens=config.max_tokens,
                             timeout=60, max_retries=1)
    if config.provider == 'Google Gemini':
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(**common, google_api_key=config.api_key,
                                     max_output_tokens=config.max_tokens, timeout=60, max_retries=1)
    if config.provider == 'Ollama':
        from langchain_ollama import ChatOllama
        return ChatOllama(**common, base_url=validate_url(config.base_url or 'http://127.0.0.1:11434'),
                          num_predict=config.max_tokens, client_kwargs={'timeout': 60})
    from langchain_openai import ChatOpenAI
    kwargs = dict(api_key=config.api_key, max_tokens=config.max_tokens, timeout=60, max_retries=1)
    if config.base_url:
        kwargs['base_url'] = validate_url(config.base_url)
    elif config.provider == 'OpenAI 호환 API':
        raise ValueError('OpenAI 호환 API의 base URL을 입력하세요. 일반적으로 /v1까지 포함합니다.')
    return ChatOpenAI(**common, **kwargs)


def create_embeddings(provider: str, model: str, api_key: str = '', base_url: str = ''):
    if not model.strip():
        raise ValueError('임베딩 모델 이름을 입력하세요.')
    if provider == 'Ollama':
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model, base_url=validate_url(base_url or 'http://127.0.0.1:11434'),
                                client_kwargs={'timeout': 60})
    if provider not in ('OpenAI', 'OpenAI 호환 API'):
        raise ValueError('지원하지 않는 임베딩 제공자입니다.')
    if not api_key:
        raise ValueError('임베딩 API 키를 설정하세요.')
    from langchain_openai import OpenAIEmbeddings
    kwargs = dict(model=model, api_key=api_key, request_timeout=60, max_retries=1)
    if base_url:
        kwargs['base_url'] = validate_url(base_url)
    elif provider == 'OpenAI 호환 API':
        raise ValueError('임베딩 API의 base URL을 입력하세요.')
    if provider == 'OpenAI 호환 API':
        kwargs['check_embedding_ctx_length'] = False
    return OpenAIEmbeddings(**kwargs)
