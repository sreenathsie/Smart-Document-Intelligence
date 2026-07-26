# 🧠 Smart Document Intelligence Platform (Colab T4)

A single-notebook, GPU-accelerated document intelligence app that runs entirely on a free/paid Google Colab T4 instance. Upload PDFs, DOCX, or XLSX files and chat with them, compare them, summarize them, or turn their tables into charts — no backend server, no cloud API keys, no cost beyond Colab compute.

Built as the Python/Streamlit-style prototype stage before porting the same logic to an on-device Android app.

---

## What it does

| Feature | Tab | What happens under the hood |
|---|---|---|
| 💬 Multi-Doc Chat | `Multi-Doc Chat` | Ask a question across one or more ingested documents; relevant chunks are retrieved and answered by the local LLM |
| 📊 Visual Chart Generation | `Visual Chart Generation` | Describe the chart you want; the LLM extracts numeric data from the documents as JSON, which is rendered as a Plotly chart |
| ⚖️ Compare Documents | `Compare Documents` | Pick Document A and Document B; the model retrieves context from each separately and produces a structured comparison |
| 📝 Auto Summary | `Auto Summary` | Extracts the top key points from the selected document(s) as a bulleted list |

Everything runs inside one Colab cell as a [Gradio](https://gradio.app) app with a public `share=True` link.

---

## Architecture

```
┌─────────────────────────────┐
│   Uploaded PDF / DOCX / XLSX │
└──────────────┬───────────────┘
               │
               ▼
      Docling DocumentConverter        (layout-aware parsing → Markdown)
               │
               ▼
   RecursiveCharacterTextSplitter      (1000 chars, 200 overlap,
               │                        splits on headers first)
               ▼
   BAAI/bge-small-en-v1.5 embeddings   (HuggingFaceEmbeddings, on CUDA)
               │
               ▼
        Chroma vector store           (persisted to ./chroma_db)
               │
     ┌─────────┴──────────┐
     │  retriever.invoke() │  filtered by {"source": {"$in": selected_docs}}
     └─────────┬──────────┘
               ▼
     Qwen2.5-7B-Instruct (4-bit NF4)   (HuggingFacePipeline, transformers + bitsandbytes)
               │
               ▼
        Gradio Blocks UI               (Chat / Chart / Compare / Summary tabs)
```

### Core components

- **Parsing — [Docling](https://github.com/DS4SD/docling):** converts PDF/DOCX/XLSX into structured Markdown, preserving headings and table layout far better than raw text extraction (e.g. PyMuPDF alone).
- **Chunking — LangChain `RecursiveCharacterTextSplitter`:** splits on `##`/`###` headers first, then paragraphs, so chunks stay semantically coherent (1000 chars, 200 overlap).
- **Embeddings — `BAAI/bge-small-en-v1.5`:** small, fast sentence-transformer run on the T4 GPU via `HuggingFaceEmbeddings`.
- **Vector store — Chroma:** local, file-persisted (`./chroma_db`), no external DB needed. Metadata filtering (`source`) scopes retrieval to only the documents the user selected — this is what makes multi-doc chat and A/B comparison possible without cross-document bleed.
- **LLM — Qwen2.5-7B-Instruct, 4-bit NF4 quantized:** loaded via `transformers` + `bitsandbytes` (`BitsAndBytesConfig`) so a 7B model fits comfortably in a T4's 15 GB VRAM.
- **Charting — Plotly Express:** the LLM is prompted to return *only* a JSON object (`title`, `chart_type`, `x_label`, `y_label`, `data`), which is parsed with a defensive `extract_valid_json()` helper (handles code fences, trailing text, malformed JSON) and rendered as bar/line/pie/scatter.
- **UI — Gradio Blocks:** left column handles upload + ingestion + a shared document selector; right column holds the four feature tabs, wired to backend functions via `.click()` events.

---

## Setup — running in Google Colab

1. **Runtime → Change runtime type → T4 GPU.**
2. Run the install cell:
   ```python
   !pip install -q -U docling langchain langchain-community langchain-huggingface chromadb
   !pip install -q -U transformers accelerate bitsandbytes sentence-transformers gradio plotly pandas
   ```
3. Run the main cell. It will, in order:
   - Load the embedding model onto CUDA
   - Download and 4-bit-quantize Qwen2.5-7B-Instruct
   - Initialize Docling's `DocumentConverter` and a persistent Chroma store
   - Launch the Gradio app with `share=True` and print a public URL
4. Open the printed `https://xxxxx.gradio.live` link, upload files, click **Ingest Documents**, then use any tab.

### Resource usage (T4, 15 GB VRAM)
From testing: ~6.9/15.0 GB GPU RAM and ~9/12.7 GB system RAM at idle with the model loaded — leaves headroom for a document or two in context, but keep an eye on GPU RAM if ingesting many large files at once.

---

## Known limitations / things to watch

- **Chart JSON reliability:** since Qwen2.5-7B isn't using constrained decoding (no `strict: true` schema enforcement like GPT-4o), the chart-generation step occasionally returns malformed or incomplete JSON (extra trailing text, truncated arrays). `extract_valid_json()` mitigates this but won't catch every case — if you see a "Failed to extract valid chart JSON" error, try rephrasing the request to be more specific (e.g. name the exact metric and years).
- **Small-model comparison quality:** with short/sparse source documents (e.g. a table of contents only), "Compare Documents" may report no differences simply because too little content was retrieved — increase `k` in `compare_documents()` or ingest fuller documents for better results.
- **No persistence across sessions:** `./chroma_db` lives in the Colab VM's ephemeral disk — it's wiped when the runtime disconnects. Mount Google Drive and point `persist_directory` there if you want documents to survive across sessions.
- **`share=True` link is public:** anyone with the URL can access your documents and the model for as long as the Colab cell is running. Don't leave it open with sensitive files.
- **Single global vector store:** all users of a shared `share=True` link write into the same Chroma instance — fine for solo/demo use, not for multi-user deployment.

---

## Future Development

This Colab/Gradio notebook is deliberately the **rapid-prototyping stage** — a place to validate the RAG pipeline and prompts cheaply before committing to a mobile build. The long-term target is a fully offline Android app with **no backend API server at all**.

### Stage 1 — Harden the Python prototype
- [ ] Extract `process_uploaded_files`, `chat_with_documents`, `compare_documents`, `extract_key_points`, and `generate_chart_from_docs` out of the notebook into standalone modules (`extract.py`, `index.py`, `chat.py`) — some of this is already scaffolded in a parallel Streamlit version
- [ ] Move from `share=True` public Gradio links to a proper Streamlit demo for local/offline testing
- [ ] Replace ad-hoc `extract_valid_json()` recovery with stricter prompt constraints or a smaller structured-output schema, to cut down malformed chart JSON
- [ ] Add persistent storage (mount Google Drive for `chroma_db`) so ingested documents survive across Colab sessions
- [ ] Tune retrieval `k` and chunk size per feature (chat vs. compare vs. summarize currently share similar defaults but have different context needs)

### Stage 2 — Swap in the on-device stack
- [ ] Replace Chroma → **FAISS** (lighter-weight, easier to bundle on-device)
- [ ] Replace `bge-small-en-v1.5` + Qwen2.5-7B with a **Gemma 3/4** model converted to `.litertlm` format
- [ ] Move inference from `transformers`/`bitsandbytes` (GPU-bound, Colab-only) to **LiteRT-LM**, targeting on-device CPU/NPU inference on Android
- [ ] Re-validate chat, chart, compare, and summary prompts against the smaller on-device model — accuracy/latency will differ meaningfully from a 7B cloud-scale model

### Stage 3 — Android app feature build-out
Beyond the four features already working in this prototype, the planned Android app adds:
- [ ] PDF editing, format conversion, and compression
- [ ] Document translation
- [ ] Form auto-filling from extracted document data
- [ ] Web search integration (to supplement retrieval with live results)
- [ ] AI agent scheduling (background tasks that re-run queries or checks on a schedule)
- [ ] Export chat sessions as PDF reports
- [ ] Sending results via WhatsApp/email using Android Intents (no email/SMS API keys needed)

### Stage 4 — Polish
- [ ] Multi-user isolation if the app ever supports shared/synced use (currently single-user by design, in line with the no-backend constraint)
- [ ] Proper error surfacing in the UI for failed ingestion or malformed model output, instead of raw exception text
- [ ] Benchmark on-device latency/memory across a range of Android devices, not just flagship hardware

---

## Tech stack summary

`Docling` · `LangChain` · `Chroma` · `bge-small-en-v1.5` · `Qwen2.5-7B-Instruct (4-bit)` · `transformers` / `bitsandbytes` · `Gradio` · `Plotly` · `pandas`
