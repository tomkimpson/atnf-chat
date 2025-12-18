# ATNF-Chat

LLM-Powered Conversational Interface for ATNF Pulsar Catalogue Queries.

## Overview

ATNF-Chat provides a natural language interface to the ATNF Pulsar Catalogue,
enabling researchers to query pulsar data, generate visualizations, and perform
analyses through conversational interactions rather than SQL queries or Python scripts.

## Features

- **Natural Language Queries**: Ask questions in plain English
- **Validated Query DSL**: Type-safe query construction with pre-execution validation
- **Scientific Safety**: Explicit null handling, provenance tracking, and data quality warnings
- **ATNF-Native Preference**: Uses official derived parameters when available
- **Reproducible**: All queries exportable as validated Python code
- **Interactive Visualizations**: P-Pdot diagrams, sky maps, and more with Plotly

## Quick Start

### Local Development

1. **Clone and install dependencies:**
   ```bash
   git clone https://github.com/tkimpson/atnf-chat.git
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

### Option 1: Vercel + Railway (Recommended)

This is the simplest deployment for public access.

**Frontend (Vercel):**

1. Push your code to GitHub
2. Import the repository in [Vercel](https://vercel.com)
3. Set the root directory to `frontend`
4. Add environment variable:
   - `NEXT_PUBLIC_API_URL`: Your Railway backend URL (e.g., `https://atnf-chat-backend.up.railway.app`)
5. Deploy

**Backend (Railway):**

1. Create a new project in [Railway](https://railway.app)
2. Connect your GitHub repository
3. Set the root directory to `/` (project root)
4. Railway auto-detects the Dockerfile
5. Add environment variables:
   - `ENVIRONMENT`: `production`
   - `API_HOST`: `0.0.0.0`
   - `API_PORT`: `8000`
6. Deploy

### Option 2: Docker on VPS

For self-hosted deployment on any VPS (DigitalOcean, Linode, etc.):

```bash
# Clone repository
git clone https://github.com/tkimpson/atnf-chat.git
cd atnf-chat

# Build and run
docker compose up -d --build

# View logs
docker compose logs -f
```

Add a reverse proxy (nginx/Caddy) for HTTPS.

### Option 3: Single Platform (Railway/Render)

Both frontend and backend can be deployed on Railway or Render as separate services within one project. This simplifies management but may have slightly higher latency for the frontend.

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
