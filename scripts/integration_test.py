#!/usr/bin/env python3
"""Integration test: LLM responses vs ground truth.

Sends benchmark questions to real LLM providers (OpenRouter and/or Anthropic),
then compares the natural language answers against known ground truth values
computed from the ATNF Pulsar Catalogue.

Usage:
    python scripts/integration_test.py --provider openrouter
    python scripts/integration_test.py --provider anthropic --questions 1,5
    python scripts/integration_test.py --provider both --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Imports from the atnf_chat codebase
# ---------------------------------------------------------------------------
from atnf_chat.api.chat import _build_system_prompt, _execute_tool
from atnf_chat.config import get_settings
from atnf_chat.core.catalogue import get_catalogue
from atnf_chat.llm.providers import AnthropicProvider, OpenRouterProvider
from atnf_chat.tools import get_tools_for_claude

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ground truth definitions
# ---------------------------------------------------------------------------

TOLERANCE_RELATIVE = "relative"
TOLERANCE_ABSOLUTE = "absolute"


@dataclass
class GroundTruth:
    """A benchmark question with its expected answer."""

    qid: str
    question: str
    expected: list[float]  # one or more expected values
    tol_type: str  # "relative" or "absolute"
    tol_value: float  # e.g. 0.05 for 5%


BENCHMARKS: list[GroundTruth] = [
    GroundTruth(
        qid="Q1",
        question="How many millisecond pulsars have orbital periods less than 1 day?",
        expected=[174],
        tol_type=TOLERANCE_RELATIVE,
        tol_value=0.02,
    ),
    GroundTruth(
        qid="Q2",
        question="How many pulsars have a period less than 10 milliseconds?",
        expected=[694],
        tol_type=TOLERANCE_RELATIVE,
        tol_value=0.02,
    ),
    GroundTruth(
        qid="Q3",
        question="How many pulsars are in globular clusters?",
        expected=[239],
        tol_type=TOLERANCE_RELATIVE,
        tol_value=0.02,
    ),
    GroundTruth(
        qid="Q4",
        question="How many binary pulsars are known?",
        expected=[571],
        tol_type=TOLERANCE_RELATIVE,
        tol_value=0.02,
    ),
    GroundTruth(
        qid="Q5",
        question="What is the period and period derivative of the Crab pulsar?",
        expected=[0.0334, 4.21e-13],
        tol_type=TOLERANCE_RELATIVE,
        tol_value=0.02,
    ),
    GroundTruth(
        qid="Q6",
        question="What is the dispersion measure (DM) of PSR J0437-4715?",
        expected=[2.6454],
        tol_type=TOLERANCE_ABSOLUTE,
        tol_value=0.05,
    ),
    GroundTruth(
        qid="Q7",
        question="How many pulsars have both a measured period derivative (P1) and a DM greater than 100?",
        expected=[1641],
        tol_type=TOLERANCE_RELATIVE,
        tol_value=0.02,
    ),
    GroundTruth(
        qid="Q8",
        question="What is the median spin period of all pulsars in milliseconds?",
        expected=[486],
        tol_type=TOLERANCE_RELATIVE,
        tol_value=0.05,
    ),
    GroundTruth(
        qid="Q9",
        question=(
            "What is the mean DM of pulsars in the Galactic plane "
            "(absolute Galactic latitude less than 5 degrees)?"
        ),
        expected=[301.43],
        tol_type=TOLERANCE_RELATIVE,
        tol_value=0.05,
    ),
    GroundTruth(
        qid="Q10",
        question="What is the characteristic age of the Crab pulsar in years?",
        expected=[1257],
        tol_type=TOLERANCE_RELATIVE,
        tol_value=0.05,
    ),
    GroundTruth(
        qid="Q11",
        question=(
            "How many pulsars have a surface magnetic field strength "
            "above 10^14 Gauss (i.e. magnetars)?"
        ),
        expected=[19],
        tol_type=TOLERANCE_RELATIVE,
        tol_value=0.02,
    ),
]


# ---------------------------------------------------------------------------
# Number extraction from natural language
# ---------------------------------------------------------------------------

_NUMBER_PATTERNS = [
    # Scientific notation: 4.21e-13, 4.21E-13, 4.21 × 10^-13, 4.21 x 10^-13
    r"(\d+\.?\d*)\s*[×xX]\s*10\^?\s*[{(]?\s*(-?\d+)\s*[})]?",
    r"(-?\d+\.?\d*)\s*[eE]\s*(-?\d+)",
    # Comma-separated integers: 1,641
    r"(?<!\d)(\d{1,3}(?:,\d{3})+)(?!\d)",
    # Decimals and integers: 174, 0.0334, 486.123
    r"(?<!\d)(-?\d+\.?\d*)(?!\d*[eExX×])",
]


def extract_numbers(text: str) -> list[float]:
    """Extract numerical values from natural language text.

    Handles integers, decimals, comma-separated numbers,
    scientific notation (4.21e-13), and 'x 10^' notation.
    """
    numbers: list[float] = []
    # Work on a copy we can consume to avoid double-matching
    remaining = text

    # Pass 1: scientific notation with × / x 10^
    for m in re.finditer(
        r"(\d+\.?\d*)\s*[×xX]\s*10\s*\^?\s*[{(]?\s*(-?\d+)\s*[})]?", remaining
    ):
        mantissa, exponent = float(m.group(1)), int(m.group(2))
        numbers.append(mantissa * 10 ** exponent)
    # Remove matched spans so they don't re-match below
    remaining = re.sub(
        r"\d+\.?\d*\s*[×xX]\s*10\s*\^?\s*[{(]?\s*-?\d+\s*[})]?", " ", remaining
    )

    # Pass 2: e-notation  (4.21e-13)
    for m in re.finditer(r"(-?\d+\.?\d*)[eE](-?\d+)", remaining):
        numbers.append(float(m.group(0)))
    remaining = re.sub(r"-?\d+\.?\d*[eE]-?\d+", " ", remaining)

    # Pass 3: comma-separated integers (1,641)
    for m in re.finditer(r"(?<!\d)(\d{1,3}(?:,\d{3})+)(?!\d)", remaining):
        numbers.append(float(m.group(1).replace(",", "")))
    remaining = re.sub(r"(?<!\d)\d{1,3}(?:,\d{3})+(?!\d)", " ", remaining)

    # Pass 4: plain decimals / integers
    for m in re.finditer(r"(?<![.\d])(-?\d+\.?\d*)(?![.\d])", remaining):
        numbers.append(float(m.group(1)))

    return numbers


# ---------------------------------------------------------------------------
# Value checking
# ---------------------------------------------------------------------------


_CLARIFY_PATTERNS = [
    r"what (?:criterion|criteria|definition|do you mean|would you like)",
    r"(?:could|can|would) you (?:clarify|specify|define|tell me)",
    r"how (?:do you|should I|would you) (?:define|classify)",
    r"what (?:threshold|cutoff|limit) (?:do you|should|would)",
    r"do you (?:mean|want|prefer)",
    r"which (?:definition|method|approach)",
    r"please (?:clarify|specify|define)",
    r"before I (?:proceed|answer|query)",
]
_CLARIFY_RE = re.compile("|".join(_CLARIFY_PATTERNS), re.IGNORECASE)


def is_clarifying_question(text: str) -> bool:
    """Detect whether the LLM response is asking for clarification
    rather than providing an answer."""
    return bool(_CLARIFY_RE.search(text))


def check_value(gt: GroundTruth, text: str) -> tuple[bool, list[float]]:
    """Check whether the LLM response contains the expected value(s).

    Returns (passed, extracted_numbers).
    """
    extracted = extract_numbers(text)
    if not extracted:
        return False, extracted

    matched_all = True
    for exp in gt.expected:
        found = False
        for num in extracted:
            if gt.tol_type == TOLERANCE_ABSOLUTE:
                if abs(num - exp) <= gt.tol_value:
                    found = True
                    break
            else:
                # Relative tolerance
                if exp == 0:
                    if num == 0:
                        found = True
                        break
                elif abs(num - exp) / abs(exp) <= gt.tol_value:
                    found = True
                    break
        if not found:
            matched_all = False

    return matched_all, extracted


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------

RATE_LIMIT_DELAYS = {
    "openrouter": 3.0,
    "anthropic": 1.0,
}


def make_provider(
    provider_name: str, settings: Any, tool_executor: Any
) -> AnthropicProvider | OpenRouterProvider:
    """Create a provider instance by name."""
    if provider_name == "anthropic":
        key = settings.anthropic_api_key
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        return AnthropicProvider(key, settings.anthropic_model, tool_executor)

    if provider_name == "openrouter":
        key = settings.openrouter_api_key
        if not key:
            raise ValueError("OPENROUTER_API_KEY not set in .env")
        return OpenRouterProvider(key, settings.openrouter_model, tool_executor)

    raise ValueError(f"Unknown provider: {provider_name}")


# ---------------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------------


STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_CLARIFY = "CLARIFY"
STATUS_ERROR = "ERROR"


@dataclass
class QuestionResult:
    qid: str
    question: str
    expected: list[float]
    extracted: list[float]
    status: str  # PASS, FAIL, CLARIFY, ERROR
    elapsed: float
    error: str | None = None
    response_text: str = ""


async def run_question(
    provider: AnthropicProvider | OpenRouterProvider,
    gt: GroundTruth,
    tools: list[dict[str, Any]],
    system_prompt: str,
    timeout: float,
    verbose: bool,
) -> QuestionResult:
    """Send one question to the provider and check the answer."""
    messages = [{"role": "user", "content": gt.question}]
    start = time.monotonic()

    try:
        text, _tool_calls = await asyncio.wait_for(
            provider.chat(messages, tools, system_prompt),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        return QuestionResult(
            qid=gt.qid,
            question=gt.question,
            expected=gt.expected,
            extracted=[],
            status=STATUS_ERROR,
            elapsed=elapsed,
            error="TIMEOUT",
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        return QuestionResult(
            qid=gt.qid,
            question=gt.question,
            expected=gt.expected,
            extracted=[],
            status=STATUS_ERROR,
            elapsed=elapsed,
            error=str(exc),
        )

    elapsed = time.monotonic() - start

    if verbose:
        print(f"\n    --- {gt.qid} full response ---")
        print(f"    {text[:500]}{'...' if len(text) > 500 else ''}")

    # Check for clarifying question before numeric comparison
    if is_clarifying_question(text):
        # Still extract numbers in case the LLM answered AND asked
        _, extracted = check_value(gt, text)
        return QuestionResult(
            qid=gt.qid,
            question=gt.question,
            expected=gt.expected,
            extracted=extracted,
            status=STATUS_CLARIFY,
            elapsed=elapsed,
            response_text=text,
        )

    passed, extracted = check_value(gt, text)

    return QuestionResult(
        qid=gt.qid,
        question=gt.question,
        expected=gt.expected,
        extracted=extracted,
        status=STATUS_PASS if passed else STATUS_FAIL,
        elapsed=elapsed,
        response_text=text,
    )


async def run_provider(
    provider_name: str,
    provider: AnthropicProvider | OpenRouterProvider,
    benchmarks: list[GroundTruth],
    tools: list[dict[str, Any]],
    system_prompt: str,
    timeout: float,
    verbose: bool,
) -> list[QuestionResult]:
    """Run all questions against one provider."""
    results: list[QuestionResult] = []
    delay = RATE_LIMIT_DELAYS.get(provider_name, 2.0)

    for i, gt in enumerate(benchmarks):
        if i > 0:
            time.sleep(delay)

        result = await run_question(provider, gt, tools, system_prompt, timeout, verbose)

        # Format extracted value for display
        if result.status == STATUS_ERROR:
            val_str = f"ERROR: {result.error}"
        elif result.status == STATUS_CLARIFY:
            # Show a snippet of the clarifying question
            snippet = result.response_text.replace("\n", " ")[:80]
            val_str = f'"{snippet}"'
        elif result.extracted:
            val_str = ", ".join(f"{v:g}" for v in result.extracted[:5])
        else:
            val_str = "(no numbers found)"

        q_short = gt.question[:50] + ("..." if len(gt.question) > 50 else "")
        print(
            f"  [{provider_name}] {gt.qid}: {q_short:<54} "
            f"{result.status:<7} ({result.elapsed:.1f}s) -> {val_str}"
        )

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_expected(expected: list[float]) -> str:
    """Format expected values for display."""
    return " / ".join(f"{v:g}" for v in expected)


def best_match(result: QuestionResult) -> str:
    """Pick the extracted number closest to expected for display."""
    if result.error:
        return f"ERR"
    if not result.extracted:
        return "—"
    # For multi-value questions, show all relevant
    if len(result.expected) > 1:
        return ", ".join(f"{v:g}" for v in result.extracted[:5])
    # For single-value, show closest
    exp = result.expected[0]
    closest = min(result.extracted, key=lambda x: abs(x - exp))
    return f"{closest:g}"


def print_comparison_table(
    all_results: dict[str, list[QuestionResult]],
    benchmarks: list[GroundTruth],
) -> None:
    """Print a side-by-side comparison table."""
    providers = list(all_results.keys())

    # Column widths
    id_w = 4
    q_w = 30
    exp_w = 14
    val_w = 18

    # Header
    header_parts = [
        f"{'ID':<{id_w}}",
        f"{'Question':<{q_w}}",
        f"{'Expected':<{exp_w}}",
    ]
    for p in providers:
        header_parts.append(f"{p.upper():<{val_w}}")

    sep = "+" + "+".join("-" * (w + 2) for w in [id_w, q_w, exp_w] + [val_w] * len(providers)) + "+"
    header = "| " + " | ".join(header_parts) + " |"

    print(f"\n{sep}")
    print(header)
    print(sep)

    # Build a lookup: provider -> qid -> result
    lookup: dict[str, dict[str, QuestionResult]] = {}
    for pname, results in all_results.items():
        lookup[pname] = {r.qid: r for r in results}

    for gt in benchmarks:
        # Only show questions that were actually run
        if not any(gt.qid in lookup[p] for p in providers):
            continue

        q_short = gt.question[:q_w - 3] + "..." if len(gt.question) > q_w - 3 else gt.question
        row_parts = [
            f"{gt.qid:<{id_w}}",
            f"{q_short:<{q_w}}",
            f"{format_expected(gt.expected):<{exp_w}}",
        ]
        for p in providers:
            r = lookup[p].get(gt.qid)
            if r is None:
                cell = "—"
            elif r.status == STATUS_CLARIFY:
                cell = "— CLARIFY"
            elif r.status == STATUS_ERROR:
                cell = "— ERROR"
            else:
                val = best_match(r)
                cell = f"{val} {r.status}"
            row_parts.append(f"{cell:<{val_w}}")

        print("| " + " | ".join(row_parts) + " |")

    print(sep)


def print_summary(all_results: dict[str, list[QuestionResult]]) -> None:
    """Print pass/fail/clarify summary per provider."""
    print("\nSUMMARY")
    print("=" * 50)
    for pname, results in all_results.items():
        total = len(results)
        n_pass = sum(1 for r in results if r.status == STATUS_PASS)
        n_fail = sum(1 for r in results if r.status == STATUS_FAIL)
        n_clarify = sum(1 for r in results if r.status == STATUS_CLARIFY)
        n_error = sum(1 for r in results if r.status == STATUS_ERROR)
        pct = (n_pass / total * 100) if total > 0 else 0
        parts = [f"Passed {n_pass}/{total} ({pct:.0f}%)"]
        if n_clarify:
            parts.append(f"{n_clarify} clarify")
        if n_fail:
            parts.append(f"{n_fail} fail")
        if n_error:
            parts.append(f"{n_error} error")
        print(f"  {pname.upper():<14} {', '.join(parts)}")

    # Print clarifying question details
    has_clarify = any(
        r.status == STATUS_CLARIFY
        for results in all_results.values()
        for r in results
    )
    if has_clarify:
        print("\nCLARIFYING QUESTIONS")
        print("-" * 50)
        for pname, results in all_results.items():
            for r in results:
                if r.status == STATUS_CLARIFY:
                    snippet = r.response_text.replace("\n", " ").strip()
                    if len(snippet) > 120:
                        snippet = snippet[:120] + "..."
                    print(f"  [{pname}] {r.qid}: {snippet}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integration test: LLM responses vs ground truth",
    )
    parser.add_argument(
        "--provider",
        choices=["openrouter", "anthropic", "both"],
        default="openrouter",
        help="Which provider(s) to test (default: openrouter)",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Comma-separated question IDs to run, e.g. '1,2,5' (default: all)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Per-question timeout in seconds (default: 90)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full LLM responses",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    # Select which questions to run
    if args.questions:
        selected_ids = {f"Q{q.strip()}" for q in args.questions.split(",")}
        benchmarks = [b for b in BENCHMARKS if b.qid in selected_ids]
        if not benchmarks:
            print(f"No matching questions for: {args.questions}")
            sys.exit(1)
    else:
        benchmarks = BENCHMARKS

    # Determine providers
    if args.provider == "both":
        provider_names = ["openrouter", "anthropic"]
    else:
        provider_names = [args.provider]

    # Load settings and pre-warm catalogue
    settings = get_settings()
    print("Loading ATNF catalogue (first call may be slow)...")
    get_catalogue()
    print("Catalogue loaded.\n")

    # Build system prompt and tools (same as production)
    system_prompt = _build_system_prompt()
    tools = get_tools_for_claude()

    # Run tests
    all_results: dict[str, list[QuestionResult]] = {}

    for pname in provider_names:
        print(f"Testing {pname.upper()}...")
        try:
            provider = make_provider(pname, settings, _execute_tool)
        except ValueError as exc:
            print(f"  Skipping {pname}: {exc}")
            continue

        results = await run_provider(
            pname, provider, benchmarks, tools, system_prompt, args.timeout, args.verbose
        )
        all_results[pname] = results
        print()

    if not all_results:
        print("No providers were tested. Check your .env configuration.")
        sys.exit(1)

    # Print comparison table and summary
    print_comparison_table(all_results, benchmarks)
    print_summary(all_results)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(main())
