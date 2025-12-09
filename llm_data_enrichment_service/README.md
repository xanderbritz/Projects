# AI-Powered Data Enrichment API Service

A lightweight FastAPI service that fetches data from an external REST API, 
processes and normalises it, and enriches the result using an LLM 
(OpenAI / Anthropic / AWS Bedrock). The service exposes clean, enriched JSON 
through simple API endpoints.

This project demonstrates:
- Python API integration and automation scripts  
- REST API consumption + authentication  
- FastAPI backend development  
- LLM prompt design and structured output generation  
- Clean data handling using Pydantic models  
- Error handling, caching, and service organisation

---

## 🚀 Purpose

Many external APIs return **raw, unstructured, or overly detailed data**.  
This service acts as an *intelligent middle layer*, taking that raw data and 
using an LLM to turn it into **summaries, tags, explanations, or normalised fields**.

Examples of enrichment:
- Summarise book info from OpenLibrary  
- Generate metadata/tags for a product  
- Classify datasets into categories  
- Convert messy text into structured JSON  

The goal is to produce clean, consistent, and useful enriched data with **one simple API call**.

---

## ✨ Features

- **FastAPI backend** with typed request/response models  
- **External API integration** (configurable; defaults to OpenLibrary)  
- **LLM enrichment** using OpenAI/Anthropic/Bedrock via a single adapter  
- **API key authentication** for your endpoints  
- **Caching layer** (optional SQLite or in-memory)  
- **Prompt templates** for consistent LLM output  
- **Automation script** for pre-fetching and normalising data  
- Clean exception handling and logging

---

## 🛠️ Tech Stack

- Python 3.10+  
- FastAPI  
- Pydantic  
- httpx (async HTTP client)  
- OpenAI / Anthropic / AWS SDK  
- Uvicorn (development server)  
- SQLite (simple caching layer)

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/llm-data-enrichment-service
cd llm-data-enrichment-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt