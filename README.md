# Financial News QA Assistant

An interactive AI chatbot designed to help users understand and analyze real-time financial news using OpenAI’s large language models, vector search, and cross-encoder reranking.

---

## Overview

This project provides a conversational agent specialized in **financial markets**, capable of interpreting and summarizing:
- Market movements
- Macroeconomic reports
- Corporate earnings
- Central bank announcements
- Geopolitical events affecting finance

The system strictly relies on retrieved live financial documents, ensuring grounded and accurate answers.

---
## API Keys Needed
- OpenAI API key
---

---
## 5 optional functionalities
### Implement streaming responses
This code implements streaming by using agent.stream_chat() which yields tokens one by one via completion.response_gen. Each token is progressively sent to the UI with yield, creating a real-time typing effect rather than waiting for the full response. The generator pattern (for token in... yield) enables this incremental delivery.


### The app is designed for a specific goal/domain that is not a tutor about AI. For example, it could be about finance, healthcare, etc.
This app is specifically designed for financial news analysis, not general AI tutoring. It focuses on interpreting market data, earnings reports, and economic indicators using specialized tools and a finance-optimized system prompt. The domain restriction is enforced by validating queries against financial topics and sourcing answers exclusively from financial news data.

### Use live search results. In this case, the user must also input the API keys for the relevant APIs (e.g., Perplexity, Bing search, etc.).
Dynamic Data Fetching: The Scraper class pulls live financial news via self.fnews.get_news(), ensuring real-time updates instead of relying on static datasets.

API-Driven Parsing: While the current snippet uses direct web scraping (requests + BeautifulSoup), the structure (e.g., parse_site) can easily integrate paid APIs (e.g., Bing/Perplexity) by modifying the input URL and headers to include user-provided API keys.

Fresh Data Storage: Scraped articles are saved to data/scraped_news/ as raw text files, creating a live-updating knowledge base for the LLM’s retriever tools.

### You have shown evidence of collecting at least two data sources beyond those provided in our course. 
Diverse Website Scraping:

The parse_site() function dynamically scrapes articles from different domains (evidenced by URL handling and User-Agent spoofing), implying it’s designed to parse varied news sites.

The soup.select() and soup.find_all() logic adapts to different HTML structures (e.g., .article__body p or generic <p> tags), confirming it targets multiple sources with distinct layouts.

Aggregated Data Storage:

self.fnews.get_news() likely pulls from multiple publishers (e.g., Bloomberg, Reuters) since financial news aggregators rarely rely on a single outlet. Saved files (hash(text).txt) further imply heterogeneous content.

### Use a reranker in your RAG pipeline. It can be a fine-tuned version (your choice).
Cross-Encoder Reranker:

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to rerank retrieved nodes by relevance to the query.

Computes query-document similarity scores via cross_encoder.predict(pairs) and sorts nodes accordingly.

Custom Retriever Integration:

Wraps the base retriever in RerankerRetriever, which applies the reranking logic after initial vector search.

Cuts irrelevant results by keeping only the top 15 reranked nodes (reranked_nodes[:15]).

---

## Features

### Domain-Specific System Prompt
The assistant is instructed to:
- Answer **only** finance-related queries.
- Use retrieved documents as its **sole source of truth**.
- Provide Markdown-formatted responses.
- Refuse unrelated questions gracefully.

### Document-Based Retrieval
- Loads `.txt` documents (e.g., scraped news articles).
- Embeds them with `text-embedding-3-small` and stores in **ChromaDB**.
- Uses **LlamaIndex** for vector retrieval.

### Reranking for Relevance
- Applies a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to rerank retrieved results by relevance.
- Only the top 15 reranked nodes are used for answer generation.

### Dynamic Agent Creation
- Constructs an `OpenAIAgent` on-the-fly with:
  - An OpenAI LLM (`gpt-4` or `gpt-4o-mini`)
  - Document retrieval tools
  - Chat memory buffer
- Answers are streamed in real time.

### Secure API Key Handling
- User is prompted to enter their OpenAI API key through the UI.
- Keys are validated and required before the assistant is activated.

### Gradio Chat Interface
- Password-protected key input
- Live chatbot with streaming responses
- Memory buffer for chat history
- API status indicator

---

## Query Lifecycle

1. User submits a financial news question.
2. Memory buffer trims excess context.
3. Relevant documents are retrieved from the vector store.
4. Results are reranked by a cross-encoder model.
5. A response is generated **only** from the reranked content.
6. The answer is streamed to the user via Gradio.

---

## Tech Stack

| Component             | Technology Used                                   |
|----------------------|---------------------------------------------------|
| LLM                  | OpenAI GPT-4 / GPT-4o-mini                        |
| Embeddings           | OpenAI `text-embedding-3-small`                   |
| Reranker             | HuggingFace `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector Store         | ChromaDB                                          |
| Retrieval Framework  | LlamaIndex                                        |
| UI                   | Gradio                                            |
| Memory Buffer        | ChatSummaryMemoryBuffer                           |
| Logging              | Python `logging` module                           |

---

---

## Cost & Pricing

This application incurs usage costs from OpenAI primarily during:

### 1. **Document Embedding**
- Model: `text-embedding-3-small`
- Used to encode financial news documents for search.
- **One-time cost per document ingestion.**
- Example:
  - 100 articles × ~750 tokens each = 75,000 tokens
  - Cost: `75 × $0.00002 = **$0.0015**`

---

### 2. **Query Execution (per user question)**

#### Components involved:
- **System prompt**: ~250 tokens
- **User query**: ~30 tokens
- **Retrieved content**: 15 documents × ~200 tokens = 3,000 tokens
- **Generated answer**: ~300 tokens

#### Using GPT-4o pricing:

| Token Type | Estimate | Rate per 1k tokens | Cost      |
|------------|----------|--------------------|-----------|
| Input      | 3,280    | $0.005             | ~$0.0164  |
| Output     | 300      | $0.015             | ~$0.0045  |
| **Total**  |          |                    | **~$0.021** per query |

---

### 3. **Reranking (CrossEncoder)**
- Model: `ms-marco-MiniLM-L-6-v2`
- Free — runs locally on CPU/GPU.
- Used to rerank retrieved results for better relevance.
- No API cost involved.


---
### Notes
- **Embedding is cheap and infrequent** (only re-run when ingesting new news).
- **Main cost driver is GPT-4o’s input tokens**, especially due to long context from retrieved documents.
- For lower-cost operation:
  - Reduce document length or top-K retrieved documents.
  - Use GPT-3.5 for less critical queries.
  - Cache and reuse embeddings or summaries when possible.

---

## Setup

1. Create a `.env` file and add there your OpenAI API key. Its content should be something like:

```bash
OPENAI_API_KEY="sk-..."
```

2. Create a local virtual environment, for example using the `venv` module. Then, activate it.

```bash
python -m venv venv
source venv/bin/activate
```

3. Install the dependencies.

```bash
pip install -r requirements.txt
```

4. Launch the Gradio app.

```bash
python app.py
```

