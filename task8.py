import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

# Document Loading Components
class DocumentLoader:
    """Load documents from various sources"""
    
    @staticmethod
    def load_from_text(file_path: str) -> str:
        """Load text from a file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def load_from_directory(directory: str, extensions: List[str] = ['.txt', '.md', '.pdf']) -> Dict[str, str]:
        """Load all documents from a directory"""
        documents = {}
        for filename in os.listdir(directory):
            if any(filename.endswith(ext) for ext in extensions):
                file_path = os.path.join(directory, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        documents[filename] = f.read()
                except:
                    print(f"Could not read {filename}")
        return documents

# Chunking Strategies
class TextChunker:
    """Split documents into chunks for embedding"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_text = ' '.join(words[start:end])
            
            chunk = {
                'text': chunk_text,
                'metadata': metadata.copy() if metadata else {},
                'start_word': start,
                'end_word': min(end, len(words))
            }
            chunks.append(chunk)
            
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def chunk_documents(self, documents: Dict[str, str]) -> List[Dict]:
        """Chunk multiple documents"""
        all_chunks = []
        for doc_name, text in documents.items():
            metadata = {'source': doc_name, 'document_id': doc_name}
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)
        return all_chunks

# Embedding with multiple backends
class EmbeddingModel:
    """Generate embeddings for text chunks"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', use_gpu: bool = False):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.model = None
    
    def initialize(self):
        """Initialize the embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            if self.use_gpu:
                self.model = self.model.to('cuda')
            print(f"Loaded embedding model: {self.model_name}")
        except ImportError:
            print("Please install sentence-transformers: pip install sentence-transformers")
            raise
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if self.model is None:
            self.initialize()
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings

# Vector Store for efficient retrieval
class VectorStore:
    """Simple vector store for similarity search"""
    
    def __init__(self):
        self.embeddings = None
        self.chunks = []
        self.metadata = []
    
    def add_embeddings(self, chunks: List[Dict], embeddings: np.ndarray):
        """Add chunks and their embeddings to the vector store"""
        self.chunks = chunks
        self.embeddings = embeddings
        self.metadata = [chunk.get('metadata', {}) for chunk in chunks]
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Find k most similar chunks to query"""
        if self.embeddings is None:
            raise ValueError("No embeddings in vector store")
        
        # Calculate cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        embeddings_norm = np.linalg.norm(self.embeddings, axis=1)
        
        similarities = np.dot(self.embeddings, query_embedding) / (embeddings_norm * query_norm)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return top chunks with similarity scores
        results = []
        for idx in top_indices:
            result = {
                'text': self.chunks[idx]['text'],
                'metadata': self.chunks[idx].get('metadata', {}),
                'similarity': float(similarities[idx])
            }
            results.append(result)
        
        return results
    
    def save(self, path: str):
        """Save vector store to disk"""
        data = {
            'chunks': self.chunks,
            'embeddings': self.embeddings.tolist() if self.embeddings is not None else None,
            'metadata': self.metadata
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str):
        """Load vector store from disk"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.chunks = data['chunks']
        self.embeddings = np.array(data['embeddings']) if data['embeddings'] else None
        self.metadata = data['metadata']

# Generation with LLM
class LLMGenerator:
    """Generate responses using LLM with retrieved context"""
    
    def __init__(self, model_type: str = "openai", api_key: Optional[str] = None):
        self.model_type = model_type
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    def generate(self, query: str, context: List[str], **kwargs) -> str:
        """Generate response using LLM"""
        
        # Prepare context
        context_text = "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(context)])
        
        # Prepare prompt
        prompt = f"""Based on the following context, answer the question.
        
Context:
{context_text}

Question: {query}

Answer the question based only on the context provided. If the context doesn't contain relevant information, say "I don't have enough information to answer this question based on the provided context."

Answer: """
        
        if self.model_type == "openai":
            return self._generate_openai(prompt, **kwargs)
        elif self.model_type == "mock":
            return self._generate_mock(query, context)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _generate_openai(self, prompt: str, **kwargs) -> str:
        """Generate using OpenAI API"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=kwargs.get('model', 'gpt-3.5-turbo'),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=kwargs.get('max_tokens', 500),
                temperature=kwargs.get('temperature', 0.7)
            )
            return response.choices[0].message.content
        except ImportError:
            print("Please install openai: pip install openai")
            return self._generate_mock(prompt, [])
    
    def _generate_mock(self, query: str, context: List[str]) -> str:
        """Mock generation for testing"""
        return f"Based on the context provided, here's an answer to: {query}\n\nContext used: {len(context)} chunks"

# Main RAG Pipeline
class RAGPipeline:
    """Complete RAG pipeline integrating all components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=config.get('chunk_size', 500),
            chunk_overlap=config.get('chunk_overlap', 50)
        )
        self.embedder = EmbeddingModel(
            model_name=config.get('embedding_model', 'all-MiniLM-L6-v2'),
            use_gpu=config.get('use_gpu', False)
        )
        self.vector_store = VectorStore()
        self.generator = LLMGenerator(
            model_type=config.get('llm_type', 'mock'),
            api_key=config.get('api_key')
        )
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the pipeline"""
        self.embedder.initialize()
        self.is_initialized = True
    
    def ingest_documents(self, documents: Dict[str, str]):
        """Ingest documents into the pipeline"""
        print(f"Ingesting {len(documents)} documents...")
        
        # Chunk documents
        chunks = self.chunker.chunk_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.embed(texts)
        print(f"Generated {len(embeddings)} embeddings")
        
        # Store in vector database
        self.vector_store.add_embeddings(chunks, embeddings)
        
        return len(chunks)
    
    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Query the RAG pipeline"""
        if not self.is_initialized:
            self.initialize()
        
        # Generate query embedding
        query_embedding = self.embedder.embed([question])[0]
        
        # Retrieve similar chunks
        retrieved = self.vector_store.similarity_search(query_embedding, k=k)
        
        # Prepare context for generation
        context_texts = [item['text'] for item in retrieved]
        
        # Generate response
        response = self.generator.generate(question, context_texts)
        
        return {
            'question': question,
            'response': response,
            'retrieved_chunks': retrieved,
            'context_count': len(context_texts)
        }
    
    def save_pipeline(self, path: str):
        """Save the entire pipeline state"""
        os.makedirs(path, exist_ok=True)
        
        # Save vector store
        self.vector_store.save(os.path.join(path, 'vector_store.json'))
        
        # Save config
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_pipeline(self, path: str):
        """Load pipeline state"""
        self.vector_store.load(os.path.join(path, 'vector_store.json'))
        
        with open(os.path.join(path, 'config.json'), 'r') as f:
            self.config = json.load(f)
        
        self.is_initialized = True

# Example usage and testing
def example_usage():
    """Example of how to use the RAG pipeline"""
    
    # Configuration
    config = {
        'chunk_size': 300,
        'chunk_overlap': 30,
        'embedding_model': 'all-MiniLM-L6-v2',
        'llm_type': 'mock',  # Change to 'openai' for real OpenAI usage
        'api_key': os.getenv('OPENAI_API_KEY')
    }
    
    # Initialize pipeline
    rag = RAGPipeline(config)
    rag.initialize()
    
    # Create sample documents
    sample_docs = {
        'doc1.txt': """Machine learning is a subset of artificial intelligence that enables 
        systems to learn and improve from experience without being explicitly programmed. 
        It focuses on developing computer programs that can access data and use it to learn for themselves.
        
        Deep learning is a specialized subset of machine learning that uses neural networks 
        with multiple layers (deep neural networks). These neural networks attempt to simulate 
        the behavior of the human brain to recognize patterns and make decisions.""",
        
        'doc2.txt': """Natural Language Processing (NLP) is a branch of artificial intelligence 
        that helps computers understand, interpret and manipulate human language. 
        NLP combines computational linguistics with statistical, machine learning, 
        and deep learning models.
        
        Transformers are a type of neural network architecture that has revolutionized 
        NLP. The attention mechanism allows transformers to process all words in a 
        sentence simultaneously rather than sequentially."""
    }
    
    # Ingest documents
    num_chunks = rag.ingest_documents(sample_docs)
    print(f"Successfully ingested {num_chunks} chunks")
    
    # Query the pipeline
    questions = [
        "What is machine learning?",
        "How does deep learning differ from machine learning?",
        "What are transformers in NLP?"
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        result = rag.query(question, k=2)
        print(f"Response: {result['response'][:200]}...")
        print(f"Retrieved {len(result['retrieved_chunks'])} chunks")
        for i, chunk in enumerate(result['retrieved_chunks']):
            print(f"  Chunk {i+1}: {chunk['text'][:100]}... (similarity: {chunk['similarity']:.3f})")
    
    # Save pipeline state
    rag.save_pipeline('./rag_pipeline_state')
    print("\nPipeline saved to './rag_pipeline_state'")

# Advanced features extension
class AdvancedRAGPipeline(RAGPipeline):
    """Extended RAG pipeline with advanced features"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.query_history = []
    
    def query_with_history(self, question: str, k: int = 5, use_history: bool = True) -> Dict[str, Any]:
        """Query with context from previous questions"""
        
        # Add conversation history to query if enabled
        if use_history and self.query_history:
            history_context = "\n".join([
                f"Previous Q: {q}\nPrevious A: {a}" 
                for q, a in self.query_history[-3:]  # Last 3 exchanges
            ])
            enhanced_query = f"{question}\n\nConversation history:\n{history_context}"
        else:
            enhanced_query = question
        
        result = super().query(enhanced_query, k=k)
        
        # Store in history
        self.query_history.append((question, result['response']))
        
        return result
    
    def rerank_results(self, query: str, retrieved: List[Dict], reranker_top_k: int = 3) -> List[Dict]:
        """Rerank retrieved results for better relevance"""
        # Simple length-based reranking (replace with cross-encoder for production)
        reranked = sorted(
            retrieved, 
            key=lambda x: len(x['text']),  # Simple heuristic
            reverse=True
        )
        return reranked[:reranker_top_k]

if __name__ == "__main__":
    print("Simple RAG Pipeline Implementation")
    print("=" * 60)
    
    # Run example
    example_usage()