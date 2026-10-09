from hashlib import sha256
from io import BytesIO
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def read_pdf(content: bytes, filename: str, source_date: str = '') -> list[Document]:
    if len(content) > 30 * 1024 * 1024:
        raise ValueError('PDF 한 개의 크기는 30MB 이하여야 합니다.')
    reader = PdfReader(BytesIO(content))
    if reader.is_encrypted:
        raise ValueError('암호화된 PDF는 지원하지 않습니다. 암호를 해제한 문서를 사용하세요.')
    if len(reader.pages) > 500:
        raise ValueError('PDF는 500페이지 이하여야 합니다.')
    digest = sha256(content).hexdigest()
    pages = []
    for number, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or '').strip()
        if not text:
            continue
        pages.append(Document(page_content=text, metadata={
            'source': filename, 'pdf_page': number, 'source_date': source_date,
            'document_id': digest,
        }))
    if not pages:
        raise ValueError('PDF에서 텍스트를 추출할 수 없습니다. 스캔 문서는 OCR이 필요합니다.')
    return pages


def split_documents(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=180,
                                              separators=['\nQ&A ', '\n\n', '\n', '. ', ' ', ''])
    chunks = []
    for page in pages:
        # The supplied document has a repeated table of contents; don't retrieve it as evidence.
        if 'contents' in page.page_content.lower() or len(page.page_content) < 80:
            continue
        # Preserve each question with its answer instead of combining unrelated Q&As.
        sections = re.split(r'(?=Q&A\s+\d+\s)', page.page_content)
        for section in sections:
            if len(section.strip()) < 40:
                continue
            metadata = dict(page.metadata)
            metadata['qa_numbers'] = ', '.join(re.findall(r'Q&A\s+(\d+)', section))
            for part in splitter.split_documents([Document(page_content=section, metadata=metadata)]):
                part.metadata['chunk_id'] = sha256(
                    f"{part.metadata['document_id']}:{part.metadata['pdf_page']}:{part.page_content}".encode()
                ).hexdigest()[:20]
                chunks.append(part)
    if not chunks:
        raise ValueError('검색할 본문이 없습니다. PDF 내용을 확인하세요.')
    return chunks
