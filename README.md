# 🔎 Multimodal Agentic Research Assistant

An AI-powered research assistant that can understand and retrieve information from both **textual and visual content inside PDF documents**.

This project combines:

- Retrieval-Augmented Generation (RAG)
- Multimodal document processing
- Semantic vector search
- Agentic question routing
- Gemini-based reasoning
- Evidence and citation resolution
- Visual evidence display
- PDF source-page previews

The result is a document research assistant that can answer questions from **text, figures, charts, tables, and other visual content** while showing the supporting source evidence.

---

## 🚀 Overview

Traditional document question-answering systems mainly focus on text.

Research papers, reports, manuals, and technical documents often contain important information inside:

- Charts
- Tables
- Figures
- Graphs
- Images
- Visual comparisons

This project addresses that limitation by creating **two searchable knowledge layers** from every document:

```text
                         PDF
                          │
              ┌───────────┴───────────┐
              │                       │
             TEXT                   IMAGES
              │                       │
              ▼                       ▼
       Text Extraction         Image Extraction
              │                       │
              ▼                       ▼
           Chunking             Gemini Vision
              │                       │
              ▼                       ▼
        Text Embeddings         Visual Analysis
              │                       │
              ▼                       ▼
          Text FAISS          Visual Embeddings
              │                       │
              ▼                       ▼
       Text Vector Store       Visual Vector Store
              │                       │
              └───────────┬───────────┘
                          │
                          ▼
                   Research Agent
                          │
                          ▼
                  Grounded Answer
                          │
                          ▼
                 Citation Resolver
                          │
                          ▼
                  Answer + Evidence