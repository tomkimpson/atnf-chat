# ATNF-Chat Frontend

Web interface for the ATNF Pulsar Catalogue conversational query system.

## Features

- Natural language chat interface with streaming responses
- Interactive welcome screen with example queries
- Code block syntax highlighting with copy button
- Tool call visibility (expandable)
- Responsive design with Tailwind CSS
- Plotly visualization support

## Development

### Prerequisites

- Node.js 18+
- npm or yarn
- Backend API running at http://localhost:8000

### Setup

```bash
# Install dependencies
npm install

# Copy environment file
cp .env.local.example .env.local

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

```bash
npm run build
npm start
```

## Configuration

Environment variables (`.env.local`):

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `http://localhost:8000` |

## Project Structure

```
src/
├── app/
│   ├── globals.css      # Global styles
│   ├── layout.tsx       # Root layout
│   └── page.tsx         # Main page
├── components/
│   ├── Chat.tsx         # Main chat container
│   ├── ChatInput.tsx    # Message input
│   ├── ChatMessage.tsx  # Message rendering
│   ├── Header.tsx       # App header
│   ├── Plot.tsx         # Plotly wrapper
│   └── WelcomeScreen.tsx # Initial screen
├── lib/
│   └── api.ts           # API client
└── types/
    └── chat.ts          # TypeScript types
```

## Running with Backend

1. Start the backend API:
   ```bash
   cd ..
   python -m atnf_chat.cli --serve
   ```

2. Start the frontend:
   ```bash
   npm run dev
   ```

3. Open http://localhost:3000

## Tech Stack

- [Next.js 16](https://nextjs.org/) - React framework
- [Tailwind CSS](https://tailwindcss.com/) - Styling
- [Plotly.js](https://plotly.com/javascript/) - Visualizations
- [Lucide React](https://lucide.dev/) - Icons
- [React Markdown](https://github.com/remarkjs/react-markdown) - Markdown rendering
