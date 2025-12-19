# ATNF-Chat

A RAG-powered conversational interface for querying the ATNF Pulsar Catalogue.

## Overview

ATNF-Chat uses **Retrieval-Augmented Generation (RAG)** to provide accurate, grounded answers about pulsars. Unlike a simple LLM wrapper, every response is backed by real data retrieved from the ATNF Pulsar Catalogue.

**How it works:**
1. You ask a question in natural language (e.g., "What are the fastest spinning pulsars?")
2. The LLM translates your question into a structured query
3. The query executes against the live ATNF catalogue (~3800+ pulsars)
4. Results are returned to the LLM, which synthesizes a grounded response

This means you get the convenience of natural language with the accuracy of direct database queries - no hallucinated pulsar data.

## Features

- **Grounded Responses**: Every answer is backed by real catalogue data, not LLM training data
- **Natural Language Queries**: Ask questions in plain English
- **Live Data Access**: Queries the current ATNF catalogue via psrqpy
- **Interactive Visualizations**: P-Pdot diagrams, sky maps, and more with Plotly
- **Scientific Safety**: Explicit null handling and data quality warnings
- **Reproducible**: All queries exportable as validated Python code

## Quick Start

### Local Development

1. **Clone and install dependencies:**
   ```bash
   git clone https://github.com/tomkimpson/atnf-chat.git
   cd atnf-chat
   pip install -e ".[dev]"
   ```

2. **Start the backend:**
   ```bash
   uvicorn atnf_chat.api.app:app --reload
   ```

3. **Start the frontend** (in a new terminal):
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. **Open** http://localhost:3000 and enter your Anthropic API key to start chatting.

### Using Docker

```bash
docker compose up --build
```

This starts both the backend (port 8000) and frontend (port 3000).

## Deployment

The app is deployed using **Vercel** (frontend) and **Railway** (backend).

**Frontend (Vercel):**

1. Import the repository in [Vercel](https://vercel.com)
2. Set the root directory to `frontend`
3. Add environment variable:
   - `NEXT_PUBLIC_API_URL`: Your Railway backend URL
4. Deploy

**Backend (Railway):**

1. Create a new project in [Railway](https://railway.app)
2. Connect your GitHub repository
3. Railway auto-detects the Dockerfile
4. Add environment variables:
   - `ENVIRONMENT`: `production`
   - `API_HOST`: `0.0.0.0`
   - `API_PORT`: `8000`
5. Deploy

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Not needed server-side (users provide their own) | - |
| `ENVIRONMENT` | `development` / `production` | `development` |
| `API_HOST` | Backend host | `127.0.0.1` |
| `API_PORT` | Backend port | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `NEXT_PUBLIC_API_URL` | Backend URL for frontend | `http://localhost:8000` |

### API Key

Users provide their own Anthropic API key in the web interface. This means:
- No API costs for you as the host
- Users control their own usage
- Keys are stored only in browser localStorage (never sent to your server for storage)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User's Browser                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Next.js Frontend (React)                │    │
│  │  - Chat interface                                    │    │
│  │  - Plotly visualizations                             │    │
│  │  - API key management                                │    │
│  └─────────────────────┬───────────────────────────────┘    │
└─────────────────────────┼───────────────────────────────────┘
                          │ HTTP/SSE
┌─────────────────────────┼───────────────────────────────────┐
│  ┌─────────────────────┴───────────────────────────────┐    │
│  │              FastAPI Backend (Python)                │    │
│  │  - Chat endpoint with streaming                      │    │
│  │  - Query DSL validation                              │    │
│  │  - psrqpy catalogue interface                        │    │
│  │  - Tool execution (query, analyze, plot)             │    │
│  └─────────────────────┬───────────────────────────────┘    │
│                        │                                     │
│  ┌─────────────────────┴───────────────────────────────┐    │
│  │              ATNF Pulsar Catalogue                   │    │
│  │  - Cached via psrqpy                                 │    │
│  │  - ~3800+ pulsars                                    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Development

### Running Tests

```bash
pytest
```

### Code Quality

```bash
ruff check .
ruff format .
mypy atnf_chat
```

### Project Structure

```
atnf-chat/
├── atnf_chat/           # Python backend
│   ├── api/             # FastAPI endpoints
│   ├── core/            # DSL, schema, validation
│   ├── tools/           # LLM tool implementations
│   └── visualization/   # Plotly chart generators
├── frontend/            # Next.js frontend
│   └── src/
│       ├── app/         # Pages
│       ├── components/  # React components
│       └── lib/         # API client, hooks
├── tests/               # Python tests
├── benchmarks/          # LLM accuracy benchmarks
├── Dockerfile           # Backend container
├── docker-compose.yml   # Full stack orchestration
└── project_brief.md     # Technical specification
```

## API Reference

### Chat Endpoint

```bash
POST /chat/
Content-Type: application/json
X-API-Key: sk-ant-...

{
  "messages": [
    {"role": "user", "content": "Show me millisecond pulsars"}
  ],
  "stream": true
}
```

### Query Endpoint

```bash
POST /query
{
  "query_dsl": {
    "select_fields": ["JNAME", "P0", "DM"],
    "filters": {
      "op": "and",
      "clauses": [
        {"field": "P0", "cmp": "lt", "value": 0.03}
      ]
    }
  }
}
```

### Health Check

```bash
GET /health
```

## License

MIT

## Acknowledgments

- [ATNF Pulsar Catalogue](https://www.atnf.csiro.au/research/pulsar/psrcat/) - CSIRO
- [psrqpy](https://github.com/mattpitkin/psrqpy) - Python interface to the catalogue
- [Anthropic Claude](https://www.anthropic.com) - LLM powering the chat interface
