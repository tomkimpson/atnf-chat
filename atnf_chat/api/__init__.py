"""FastAPI backend for ATNF-Chat.

This module contains:
- REST API endpoints for chat and queries
- Streaming response handlers
- Code export functionality
"""

from atnf_chat.api.app import app
from atnf_chat.api.chat import router as chat_router

__all__ = [
    "app",
    "chat_router",
]
