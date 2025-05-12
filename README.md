# resume-booster-rag 🚀

## Project description
Resume Booster AI helps job-seekers refine their CVs.
Paste a target role or short JD and the app surfaces anonymised résumé bullets, rewrites them in second-person imperative, and shows the source résumé IDs so you can adapt them.

### Name & live URL
| Item | URL |
|------|-----|
| Hugging Face Space | _tbd_ |
| Embedding model | [BAAI / bge-m3](https://huggingface.co/BAAI/bge-m3) |
| Generation model | [Gemini 2.0 Flash-Lite](https://ai.google.dev/gemini-api/docs/models?hl=de#gemini-2.0-flash-lite) |
| Code repo | https://github.com/Jasminh/resume-booster-rag |

## Data sources
| Source | Notes |
|--------|-------|
| [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) | 54 PDFs / txt résumés across multiple professions |

---

## RAG Improvements

| Improvement          | Description                                                                                          |
| -------------------- | ---------------------------------------------------------------------------------------------------- |
| **Query Expansion**  | Generate synonym / lay-term variants of the clinician’s question to widen recall.                    |
| **Query Rewriting**  | Reformulate queries into “active substance + indication” patterns that match the registry style.     |
| **Result Reranking** | Re-rank the top 10 registry snippets with a lightweight cross-encoder for higher clinical relevance. |

---

## Chunking

### Data Chunking Method

| Type of Chunking                 |                Configuration |
| -------------------------------- | ---------------------------: |
| `RecursiveCharacterTextSplitter` | 1000 characters, 100 overlap |
| Alternative tried                |                       *x, y* |
| Alternative tried                |                       *x, y* |

---

## Choice of LLM

| Name                  | Link                                                                                                                                                               |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Gemini 2.0 Flash Lite | [https://ai.google.dev/gemini-api/docs/models/gemini?hl=de#gemini-2.0-flash-lite](https://ai.google.dev/gemini-api/docs/models/gemini?hl=de#gemini-2.0-flash-lite) |

*(Add rows if additional models are compared or combined.)*

---

## Test Method

Describe how test questions (e.g., “What is the maximum daily dose of ibuprofen in adults?”) were compiled, how ground-truth answers were sourced, and which metrics (exact-match, F1, human clinical validity scoring) were applied to evaluate retrieval and generated answers.

---

## Results

| Model / Method                   | Accuracy | Precision | Recall |
| -------------------------------- | -------- | --------- | ------ |
| Retrieved chunks with config xyz | –        | –         | –      |
| Generated answer with config xyz | –        | –         | –      |

---

## References
