# Archon

**Production-grade multi-agent AI research platform.**

Built following [arXiv 2512.08769](https://arxiv.org/abs/2512.08769) best practices for designing, developing, and deploying agentic AI workflows.

---

## What It Does

Upload documents or ask research questions → Multi-agent system researches, analyzes, synthesizes → Get structured reports with citations.

**Use Cases:**
- Due diligence on companies
- Competitive intelligence
- Market research
- Document analysis
- Investment research

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ARCHON                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  AGENTS (Single-Responsibility, LangGraph)                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Planner  │ │Researcher│ │ Analyst  │ │  Writer  │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│                                                                  │
│  RAG PIPELINE                                                    │
│  Ingest → Chunk → Embed → Store → Retrieve → Rerank             │
│                                                                  │
│  TOOLS (Deterministic + LLM-Invoked)                            │
│  Pure functions for infra │ LLM tools for reasoning             │
│                                                                  │
│  API (FastAPI)                                                   │
│  REST + WebSocket │ Auth │ Rate Limiting │ Pydantic             │
│                                                                  │
│  MONITORING                                                      │
│  Structured Logs │ OpenTelemetry │ Prometheus │ Cost Tracking   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.11+ with type hints |
| **Agents** | LangGraph, LangChain |
| **LLMs** | OpenAI, Anthropic (multi-model consortium) |
| **Vector DB** | Chroma (dev), Pinecone (prod) |
| **API** | FastAPI, Pydantic, uvicorn |
| **Database** | PostgreSQL, SQLAlchemy |
| **Caching** | Redis |
| **Monitoring** | OpenTelemetry, Prometheus |
| **Deployment** | Docker, GitHub Actions |
| **Frontend** | React, TypeScript, TailwindCSS |

---

## Best Practices Implemented

From [arXiv 2512.08769](https://arxiv.org/abs/2512.08769):

1. **Tool-first design over MCP** — MCP as thin adapter only
2. **Pure-function invocation** — Deterministic ops without LLM
3. **Single-tool agents** — One tool per agent maximum
4. **Single-responsibility** — One task per agent
5. **Externalized prompts** — Version-controlled in `/prompts`
6. **Model consortium** — Multi-LLM with reasoning consolidation
7. **Workflow/MCP separation** — Business logic outside MCP
8. **Containerized deployment** — Docker + cloud configs
9. **KISS principle** — Flat, function-driven architecture

---

## Project Structure

```
archon/
├── src/
│   ├── agents/          # LangGraph multi-agent system
│   ├── rag/             # RAG pipeline
│   ├── tools/           # Deterministic + LLM tools
│   ├── prompts/         # Externalized, versioned prompts
│   ├── memory/          # Working, episodic, semantic memory
│   ├── models/          # LLM providers, routing, consortium
│   ├── api/             # FastAPI routes, middleware
│   ├── mcp/             # MCP thin adapter
│   ├── security/        # OWASP Agentic implementation
│   ├── monitoring/      # Logging, tracing, metrics, costs
│   ├── evaluation/      # Test sets, LLM-judge, RAGAS
│   └── core/            # Config, errors, types
├── frontend/            # React UI
├── tests/               # Unit, integration, adversarial
├── docker/              # Containerization
├── deploy/              # GCP, Azure, AWS configs
├── docs/                # Architecture, API docs
└── .github/workflows/   # CI/CD
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/Faranmo/Archon.git
cd Archon

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run
uvicorn src.api.main:app --reload
```

---

## Skills Demonstrated

This project demonstrates competencies from 31 AI Engineer job descriptions:

### Tier 1 (Critical)
- Python, LLM APIs, RAG, Vector DBs, Prompt Engineering
- Agentic AI, LangGraph, FastAPI, Cloud Deployment
- Monitoring, Evaluation

### Tier 2 (Important)
- MCP, PyTorch/HuggingFace, FinOps, Guardrails
- SQL, A/B Testing, CI/CD, MLOps concepts

---

## License

MIT

---

## Author

Built by [Faran](https://github.com/Faranmo)
