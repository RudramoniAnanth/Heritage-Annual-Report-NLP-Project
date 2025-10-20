## Heritage Foods NLP Project Report (FY 2024–25)

### Abstract
This project builds a lightweight Retrieval-Augmented Generation (RAG) assistant to answer investor- and company-related questions using the Heritage Foods Annual Report FY 2024–25 and the Heritage Foods FAQ document. The system combines classical information retrieval (TF–IDF with cosine similarity) for fast, transparent context selection and a generative model (via OpenRouter) for fluent, grounded answers. The app is delivered as a Streamlit interface with configurable parameters and explainability through displayed source snippets.

### Objectives
- Provide accurate, grounded answers strictly from Heritage Foods authoritative documents.
- Offer explainability by surfacing the exact passages used to answer.
- Keep the solution lightweight and easily deployable on modest hardware.
- Support multiple corpora (Annual Report + FAQ PDF) and flexible retrieval settings.

### Documents and Data Sources
- Annual Report: `Heritage_Annual_Report_2024-25.pdf` (preprocessed into `chatbot_data.json` with pages, vocabulary, TF–IDF matrix)
- FAQs: `Heritage_Foods_2024_25_FAQ.pdf` (parsed at runtime, vectorized against the same vocabulary)

### Methodology
1. Preprocessing (Annual Report)
   - Segment PDF into pages and clean text.
   - Build a vocabulary and TF–IDF matrix offline.
   - Persist artifacts into `chatbot_data.json` for quick loading.

2. Runtime Retrieval (Annual Report + FAQ)
   - Build a `CountVectorizer` with the fixed vocabulary to encode queries.
   - Encode the FAQ pages into a TF–IDF matrix aligned with the same vocabulary at app startup.
   - For a user query, compute cosine similarity against both matrices and select top‑k passages.

3. Answer Generation
   - Construct a prompt with the top passages (source and page metadata included).
   - Call OpenRouter (configurable model; default `openai/gpt-3.5-turbo`).
   - Enforce groundedness: the system prompt instructs the model to only use provided context and avoid fabrication.

### System Architecture
- UI Layer: Streamlit app (`faq_app.py`) with search input, Ask button, Quick FAQs, and a context viewer.
- Retrieval Layer: TF–IDF + cosine similarity over Annual Report and FAQ corpora.
- Generation Layer: OpenRouter API integrates a hosted LLM for answer synthesis.
- Caching: Streamlit resource caching for loaded artifacts and FAQ matrix.

```
[User] → Streamlit UI → Query Encoding → Similarity Search (AR + FAQ) → Top‑k Snippets
                                                       ↓
                                         Prompt Builder (grounded context)
                                                       ↓
                                         OpenRouter LLM → Final Answer
```

### Implementation Overview
- File: `faq_app.py`
  - Loads `chatbot_data.json` (pages, vocabulary, TF–IDF matrix for Annual Report).
  - Builds a fixed‑vocabulary `CountVectorizer` for query encoding.
  - Parses `Heritage_Foods_2024_25_FAQ.pdf` using `pypdf` (fallback to `PyPDF2`) and creates a TF–IDF matrix for the FAQ corpus.
  - Retrieves top‑k snippets from both sources, merges and sorts by score, and limits to k.
  - Displays the answer and the exact supporting passages with source labels (Annual Report or FAQ PDF).
  - Includes a horizontally centered Quick FAQs section to prefill common queries.
  - Adds a simple centered footer.

#### Key Dependencies
- Python 3.10+
- `streamlit`, `numpy`, `scikit-learn`, `requests`, `pypdf` or `PyPDF2`

#### Running the App
```bash
py -m pip install -r requirements.txt  # if available
py -m pip install streamlit numpy scikit-learn requests pypdf PyPDF2
streamlit run faq_app.py
```

#### Configuration
- API: OpenRouter key can be set in code or overridden via the sidebar.
- Retrieval: `k` (number of snippets) and `MAX_SNIPPET_CHARS` are adjustable from the sidebar.

Indicative Test Queries
- “Summarize FY 2024–25 performance highlights.”
- “Who should I contact for investor relations?”
- “What is the shareholding pattern?”
- “What are Heritage Foods' key business segments?”
- “What is Heritage Foods’ vision for 2030?”
- “How does Heritage Foods ensure corporate governance?”

### Results (Illustrative)
- The merged retrieval from Annual Report and FAQ improved coverage for investor‑centric questions.
- Displaying source metadata (Report vs FAQ) increases trust and traceability.
- Quick FAQs reduced time‑to‑answer for common queries.

### Conclusion
This project delivers a practical, explainable RAG assistant for Heritage Foods by combining fast TF–IDF retrieval with an LLM to generate grounded answers. Merging the Annual Report with the FAQ corpus improves coverage for investor‑focused questions while keeping the system lightweight and easy to deploy. The app’s “Context used” view increases trust through transparent citations, and the Quick FAQs streamline common queries. With the proposed enhancements (dense retrieval, reranking, and richer citations), the system can further improve recall and user confidence while maintaining simplicity.

### References
- Heritage Foods Annual Report 2024–25
- Heritage Foods FAQ document 2024–25
- OpenRouter API documentation (`https://openrouter.ai/docs`)
- Scikit‑learn documentation (`https://scikit-learn.org/`)
- Streamlit documentation (`https://docs.streamlit.io/`)

### Appendix A: Environment and Variables
```ini
# OpenRouter
OPENROUTER_API_KEY=<your_key>
```

### Appendix B: High‑Level Pseudocode
```text
load pages, vocabulary, tfidf_matrix from chatbot_data.json
vectorizer = CountVectorizer(vocabulary)
faq_pages = read_pdf(FAQ_PDF_PATH)
faq_matrix = TfidfTransformer().fit_transform(vectorizer.transform(text_of(faq_pages)))

def retrieve(query):
    q = normalize(vectorizer.transform([query]))
    s1 = cosine_similarity(q, matrix).argsort(desc)[:k]
    s2 = cosine_similarity(q, faq_matrix).argsort(desc)[:k]
    return top_k_by_score(merge(s1_pages, s2_pages), k)

snippets = retrieve(user_query)
prompt = build_prompt(snippets, user_query)
answer = openrouter(prompt)
display(answer, snippets)
```


