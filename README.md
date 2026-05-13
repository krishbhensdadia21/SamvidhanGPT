# SamvidhanGPT – AI-Powered Indian Constitution Assistant

**SamvidhanGPT** is an intelligent and modern AI-powered web application designed to help users explore and understand the **Constitution of India** through conversational interaction. Built using **Python, Flask, HTML, CSS, JavaScript, FAISS Vector Search, Sentence Transformers and Groq LLM API**, the platform delivers fast, context-aware, and accurate constitutional answers using Retrieval-Augmented Generation (RAG).

## Features

- 🤖 AI-powered chatbot focused exclusively on the Indian Constitution  
- 📜 Covers Articles, Schedules, Amendments, Preamble, Rights & Duties  
- 🧠 Smart topic guard to block non-constitutional queries  
- 🔍 Semantic search using **FAISS vector database**  
- 📚 Context-aware Retrieval-Augmented Generation (RAG) pipeline  
- ⚡ Fast responses powered by **Groq LLM API**  
- 🏛 Direct lookup for Articles, Schedules, and Preamble queries  
- 🧾 Amendment notes integration for precise legal context  
- 💬 Interactive modern chatbot UI with responsive design  
- 🔄 Multi-stage fallback retrieval for better accuracy  

## Tech Stack

- *Python*  
- *Flask*  
- *HTML5*  
- *CSS3*  
- *JavaScript*  
- *Groq API*  
- *FAISS Vector Database*  
- *Sentence Transformers*  
- *NumPy*  
- *JSON Dataset*  

## Core Functionalities

- **Topic Guard System** – Detects whether the question is related to the Indian Constitution  
- **Article Lookup Engine** – Instantly fetches specific constitutional Articles  
- **Schedule Search Module** – Retrieves information from all 12 Schedules  
- **Preamble Query Support** – Directly answers questions related to the Preamble  
- **Semantic Search (FAISS)** – Finds relevant constitutional content using embeddings  
- **Amendment Tracking** – Includes amendment notes wherever applicable  
- **Conversational AI** – Converts retrieved legal context into simple understandable answers  
- **Fallback Retrieval Pipeline** – Combines direct search + semantic search for best results  

## API Endpoints

- **/** – Serves the frontend web interface  
- **/get-response** – Handles chatbot user queries and returns AI responses  
- **/static/** – Serves CSS, JavaScript, and assets  

## Project Highlights

- 🚀 Built a complete Constitution-focused AI assistant from scratch  
- ⚖️ Combined LLM intelligence with legal document retrieval  
- 🔍 Implemented FAISS semantic vector search for precision results  
- 🧩 Added constitutional topic filtering for domain-specific answers  
- 📡 Optimized Groq API responses with contextual prompting  
- 🛠 Developed scalable Flask backend with structured routing logic  
- 🎯 Created educational AI tool for students, citizens, and exam aspirants

## Project Structure

SamvidhanGPT/
│── app.py
│── final.json
│── documents.json
│── documents_meta.json
│── embed.py
│── constitution.index
│── templates/
│   └── index.html
│── static/
│   └── Constitution.jpg
│   └──  styles.css
│   └── script.js
│── .env
│── requirements.txt
