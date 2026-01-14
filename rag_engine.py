import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate

load_dotenv()

class RAGEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.vector_store = None

    def process_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Professional Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        
        # Create Vector Store
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        return len(chunks)

    def get_response(self, query):
        if not self.vector_store:
            return "Please upload a document first.", 0, 0
        
        # 1. Retrieval
        docs = self.vector_store.similarity_search(query, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        # 2. Generation
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        response = self.llm.invoke(prompt).content
        
        # 3. Quality Evaluation (The "Advanced" part)
        eval_scores = self.evaluate_quality(query, context, response)
        
        return response, eval_scores['faithfulness'], eval_scores['confidence']

    def evaluate_quality(self, query, context, response):
        """Uses LLM-as-a-judge to score the output"""
        eval_prompt = f"""
        Evaluate the following RAG response based on the Context:
        Context: {context}
        Response: {response}
        
        Return a score from 0 to 1 for:
        1. Faithfulness: Is the answer derived ONLY from the context?
        2. Confidence: How well does the context actually answer the question?
        
        Format: Faithfulness: [score], Confidence: [score]
        """
        eval_res = self.llm.invoke(eval_prompt).content
        # Simple parsing logic (can be made more robust)
        try:
            f = float(eval_res.split("Faithfulness: ")[1].split(",")[0])
            c = float(eval_res.split("Confidence: ")[1])
        except:
            f, c = 0.5, 0.5
        return {"faithfulness": f, "confidence": c}
