import os
import logging
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from typing import List, Literal
import requests

class JinaEmbeddings:
    def __init__(self, api_key, dimensions=1024):
        self.api_key = api_key
        self.dimensions = dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get_embeddings(texts, task='retrieval.passage')

    def embed_query(self, text: str) -> List[float]:
        return self._get_embeddings([text], task='retrieval.query')[0]

    def _get_embeddings(self, texts: List[str], task: Literal['retrieval.passage', 'retrieval.query']):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        data = {
            'input': texts,
            'model': 'jina-embeddings-v3',
            'dimensions': self.dimensions,
            'task': task
        }
        response = requests.post('https://api.jina.ai/v1/embeddings', headers=headers, json=data)
        return [e["embedding"] for e in response.json()["data"]]

class DocumentLoader:
    def __init__(self, pinecone_api_key, jina_api_key, index_name):
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.embeddings = JinaEmbeddings(api_key=jina_api_key, dimensions=1024)
        global logger  # 添加这一行以定义 logger
        logger = logging.getLogger(__name__)  # 初始化 logger

        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            logger.info(f'Available indexes: {self.pc.list_indexes()}')
            if self.index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating new index: gpt-engineer-index ")
                self.pc.create_index(name="gpt-engineer-index", dimension=1024, metric="cosine", 
                                     spec=ServerlessSpec(cloud="aws", region="us-east-1"))
            self.vectorStore = PineconeVectorStore(index_name=self.index_name, embedding=self.embeddings)
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise

    def file_already_embedded(self, file_name):
        query_result = self.vectorStore.similarity_search(
            query="",
            k=1,
            filter={"file_name": file_name}
        )
        return len(query_result) > 0

    def load_and_index_document(self, file_path):
        file_name = os.path.basename(file_path)
        if self.file_already_embedded(file_name):
            logger.info(f"File {file_name} has already been embedded. Skipping.")
            return

        try:
            _, file_extension = os.path.splitext(file_path)
            logger.info(f"Processing file: {file_path}")

            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.docx':
                loader = Docx2txtLoader(file_path)
            elif file_extension == '.md':
                loader = UnstructuredMarkdownLoader(file_path)
            elif file_extension in ['.html', '.htm']:
                loader = WebBaseLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from {file_path}")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)
            logger.info(f"Split into {len(texts)} text chunks")

            for text in texts:
                text.metadata.update({
                    "file_name": file_name,
                    "file_path": file_path
                })

            self.vectorStore.add_documents(texts)
            logger.info(f"Added {len(texts)} text chunks to Pinecone index for {file_name}")

            # 验证索引是否包含新添加的文档
            #self.verify_index_content()

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise

    #def verify_index_content(self):
    #    try:
    #        index = self.pc.Index(self.index_name)
    #        stats = index.describe_index_stats()
    #        logger.info(f"Index stats after loading: {stats}")
    #        # 可以在这里添加更详细的验证逻辑，如检查向量数量是否增加等
    #    except Exception as e:
    #        logger.error(f"Error verifying index content: {e}")

    def query_documents(self, query, k=2):
        try:
            results = self.vectorStore.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} relevant documents for query: {query}")
            logger.info(f"Relevant documents: {results}")
            return results
        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            raise