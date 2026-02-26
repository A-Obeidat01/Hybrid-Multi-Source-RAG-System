#  Modular Hybrid RAG System

##  Overview

This project implements a **Modular Hybrid Retrieval-Augmented Generation (RAG) architecture** that integrates multiple Large Language Model (LLM) backends into a unified pipeline.

The system supports:

- Groq (high-speed cloud inference)
- Hugging Face hosted models
- Local LLM deployment

It is designed with a flexible and scalable architecture that enables dynamic model switching, backend abstraction, and efficient document retrieval.

---

##  Architecture Highlights

- Multi-LLM orchestration layer  
- Backend abstraction for seamless provider switching  
- Vector-based document retrieval pipeline  
- Modular and extensible system design  
- Production-oriented project structure  

This setup allows benchmarking, experimentation, and performance comparison across different inference providers while maintaining a consistent RAG workflow.

---

##  System Pipeline

1. Document ingestion & preprocessing  
2. Text chunking  
3. Embedding generation  
4. Vector storage & similarity search  
5. Context retrieval  
6. Multi-backend LLM response generation  

---

##  Objectives

- Compare performance across Groq, Hugging Face, and Local LLMs  
- Enable dynamic backend switching  
- Maintain clean modular architecture  
- Provide a scalable foundation for AI-powered applications  

---

##  Tech Stack

- Python  
- Vector Database  
- Groq API  
- Hugging Face Inference API  
- Local LLM Runtime  
