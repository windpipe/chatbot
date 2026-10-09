import re
from uuid import uuid4

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    words = re.findall(r'[가-힣a-z0-9]+', text.lower())
    tokens = list(words)
    # Korean particles vary; character n-grams allow matching without a morphology service.
    for word in words:
        if re.search(r'[가-힣]', word):
            tokens.extend(word[i:i + n] for n in (2, 3) for i in range(len(word) - n + 1))
    return tokens


class SearchIndex:
    def __init__(self, chunks: list[Document], embeddings=None):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(chunk.page_content) for chunk in chunks])
        self.vector = None
        if embeddings is not None:
            from langchain_chroma import Chroma
            self.vector = Chroma.from_documents(chunks, embeddings,
                                               collection_name='rag_' + uuid4().hex)

    def close(self):
        if self.vector is not None:
            self.vector.delete_collection()

    def search(self, query: str, k: int = 5) -> list[Document]:
        if not query.strip():
            return []
        scores = self.bm25.get_scores(tokenize(query))
        lexical = sorted((i for i, score in enumerate(scores) if score > 0),
                         key=lambda i: scores[i], reverse=True)[:max(k * 2, 10)]
        ranked = {self.chunks[i].metadata['chunk_id']: 1 / (60 + rank)
                  for rank, i in enumerate(lexical, 1)}
        documents = {chunk.metadata['chunk_id']: chunk for chunk in self.chunks}
        if self.vector is not None:
            for rank, chunk in enumerate(self.vector.similarity_search(query, k=max(k * 2, 10)), 1):
                cid = chunk.metadata['chunk_id']
                ranked[cid] = ranked.get(cid, 0) + 1 / (60 + rank)
        selected = sorted(ranked, key=ranked.get, reverse=True)[:k]
        return [documents[cid] for cid in selected]
