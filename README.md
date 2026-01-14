# Quality-Aware-RAG
A next-generation Retrieval-Augmented Generation (RAG) system designed to eliminate hallucinations. Unlike standard RAG implementations, this system incorporates a Self-Evaluation Layer to verify the Faithfulness and Confidence of every response generated from your PDF documents.


**Key Features:**
> **Intelligent Document Parsing:** Uses recursive character splitting to maintain semantic context.

>**Vectorized Search:**Powered by FAISS for lightning-fast retrieval of relevant document segments.

>** Faithfulness Scoring:** Automatically evaluates if the AI's answer is actually supported by the uploaded PDF (Groundedness).

>** Confidence Indicators:** Provides a visual score to the user indicating how reliable the information is.

> **Conversational Memory:** Remembers the context of the chat for seamless follow-up questions.

🏗️ Architecture
This system follows an advanced "Closed-Loop" RAG architecture:
Ingestion: PDF $\rightarrow$ Text Chunks $\rightarrow$ Vector Embeddings.
Retrieval: User Query $\rightarrow$ Semantic Search $\rightarrow$ Top-k Context Chunks.
Generation: Context + Query $\rightarrow$ LLM $\rightarrow$ Response.
Evaluation: Response + Context $\rightarrow$ Quality Check (Faithfulness & Confidence).


📊 Comparison: Why this is better than "Normal" RAG
Standard RAG systems often suffer from "hallucinations" where the AI makes up facts not found in the source.
This project solves that by adding a verification step.

🛠️ Tech Stack
Framework: LangChain

LLM: OpenAI GPT-4o / GPT-3.5-Turbo

Vector Store: FAISS (Facebook AI Similarity Search)

Frontend: Streamlit

Environment: Python 3.9+
