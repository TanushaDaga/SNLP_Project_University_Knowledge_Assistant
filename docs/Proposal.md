# Project Proposal: University Knowledge Assistant

## 1. Problem & Institutional Context
University students and faculty frequently struggle to quickly locate specific institutional information across a vast repository of scattered PDFs, handbooks, policy documents, and guidelines. The manual search process is time-consuming and prone to errors, often leading to missed deadlines or non-compliance with university rules. There is a strong need for an automated, centralized, and intelligent retrieval system that can synthesize this information efficiently.

## 2. Target Users
- **Students (Primary):** Seeking immediate answers regarding attendance policies, examination rules, capstone project guidelines, placements, and campus facilities.
- **Faculty & Administration (Secondary):** Requiring quick reference to institutional policies to advise students accurately and ensure rule enforcement.

## 3. System Interface
- **Input:** Natural language query from the user (e.g., "What is the minimum attendance required to be eligible for exams?").
- **Output:** The most relevant information chunk extracted from the documents, alongside the exact source document name and a confidence/relevance score.

## 4. NLP Task Definition
This project addresses the **Document Retrieval (Semantic Search)** task, a core component of Retrieval-Augmented Generation (RAG) systems. The system must parse unstructured text from PDFs, chunk it into digestible segments, and match a user's query against this corpus to retrieve the most contextually relevant passages.

## 5. Dataset Overview
The dataset consists of a localized repository of official university documents provided in PDF format (`data/raw/`). Key documents include:
- Student Resource Book
- Attendance, Leave, and Examination Rules
- Placement and Internship Policies
- Community Service and Capstone Project Guidelines
- CC Studio and Facility Guidelines

## 6. Proposed Baseline Approach
To establish a rigorous baseline for retrieval performance before introducing complex neural architectures, we implemented an advanced traditional NLP pipeline:
- **Data Processing:** Extraction and cleaning of text from PDFs using `pdfplumber`.
- **Semantic Chunking & Lemmatization:** Instead of arbitrary word counts, we used `nltk` sentence tokenization to chunk documents logically by overlapping sentences, preserving semantic boundaries. We also implemented `WordNetLemmatizer` to ensure query terms match variations in the text.
- **Retrieval Algorithm:** We utilized **BM25Okapi** (via `rank_bm25`), the gold-standard algorithm for sparse retrieval. Unlike raw TF-IDF, BM25 handles term saturation and document length normalization, significantly improving recall on complex queries.

## 7. Expected Institutional Value
Deploying this knowledge assistant will drastically reduce the time spent searching for administrative and academic policies. It will empower students with self-service capabilities, reduce the administrative burden on faculty, and ensure consistent, accurate dissemination of university rules across the campus.
