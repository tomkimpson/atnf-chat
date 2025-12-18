"""Command-line interface for ATNF-Chat."""

import argparse
import sys


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="atnf-chat",
        description="LLM-Powered Conversational Interface for ATNF Pulsar Catalogue",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the API server",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="API server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (default: 8000)",
    )

    args = parser.parse_args()

    if args.serve:
        try:
            import uvicorn

            from atnf_chat.api.app import app

            uvicorn.run(app, host=args.host, port=args.port)
        except ImportError as e:
            print(f"Error: {e}. Make sure uvicorn is installed.", file=sys.stderr)
            return 1
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
