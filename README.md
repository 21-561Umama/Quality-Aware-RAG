# Quality-Aware-RAG
A next-generation Retrieval-Augmented Generation (RAG) system designed to eliminate hallucinations. Unlike standard RAG implementations, this system incorporates a Self-Evaluation Layer to verify the Faithfulness and Confidence of every response generated from your PDF documents.


Key Features
Intelligent Document Parsing: Uses recursive character splitting to maintain semantic context.

Vectorized Search: Powered by FAISS for lightning-fast retrieval of relevant document segments.

Faithfulness Scoring: Automatically evaluates if the AI's answer is actually supported by the uploaded PDF (Groundedness).

Confidence Indicators: Provides a visual score to the user indicating how reliable the information is.

Conversational Memory: Remembers the context of the chat for seamless follow-up questions.
