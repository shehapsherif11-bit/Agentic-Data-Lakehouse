"""
router.py — Master Supervisor CLI.

Thin presentation layer over router_graph.py: Rich terminal UI (with a
plain-ANSI fallback), a persistent rotating log, and the interactive chat
loop. All routing/graph logic lives in router_graph.py so it can be tested
without a terminal attached.

Fixes applied vs. the original single-file version:
  * `arabic_reshaper` / `python-bidi` are now optional, like Rich already
    was — the original hard-imported them, so a chat that only needs
    English would still crash on startup if they weren't installed.
  * Reshaping + bidi are applied per-line and only when the answer's
    detected language is Arabic or mixed. Applying `get_display()` to an
    entire multi-paragraph Markdown answer in one shot (the original
    behavior) can visually scramble numbered lists, tables, and code
    fences, since the bidi algorithm doesn't understand Markdown structure
    across line breaks. Line-by-line reshaping avoids that.
  * A reshape failure degrades to printing the plain text instead of
    crashing the response.

Requirements:
    pip install -r requirements.txt
    (rich and arabic-reshaper/python-bidi are optional — see requirements.txt)
"""
import sys
import time
from datetime import datetime

from langchain_core.messages import HumanMessage

import router_config as cfg
from router_graph import build_graph, logger

# ==========================================
# Optional: Rich terminal UI
# ==========================================
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# ==========================================
# Optional: Arabic shaping/bidi (only needed for terminals that can't
# render Arabic ligatures/RTL correctly on their own)
# ==========================================
try:
    import arabic_reshaper
    from bidi.algorithm import get_display

    ARABIC_SHAPING_AVAILABLE = True
except ImportError:
    ARABIC_SHAPING_AVAILABLE = False


class C:
    BLUE, GREEN, YELLOW, RED, CYAN, BOLD, END = (
        "\033[94m", "\033[92m", "\033[93m", "\033[91m", "\033[96m", "\033[1m", "\033[0m",
    )


def say(msg: str, style: str = "cyan", plain_color: str = C.CYAN):
    """Unified print: Rich if available, otherwise plain ANSI color."""
    if RICH_AVAILABLE:
        console.print(msg, style=style)
    else:
        print(f"{plain_color}{msg}{C.END}")


def _reshape_for_terminal(text: str, language: str) -> str:
    """Reshape + bidi-reorder Arabic text for terminals that need it.
    Applied per-line (not to the whole block) so Markdown structure
    (lists, tables, code fences) isn't scrambled by bidi reordering, and
    skipped entirely for pure-English answers or when the optional
    dependency isn't installed."""
    if not ARABIC_SHAPING_AVAILABLE or language not in ("ar", "mixed"):
        return text
    try:
        return "\n".join(get_display(arabic_reshaper.reshape(line)) for line in text.split("\n"))
    except Exception as e:
        logger.warning("Arabic reshaping failed, showing plain text: %s", e)
        return text


def render_answer(answer: str, elapsed: float, language: str = "en"):
    display_text = _reshape_for_terminal(answer, language)
    if RICH_AVAILABLE:
        console.print(
            Panel(Markdown(display_text), title=f"🎯 Answer  •  ⏱ {elapsed:.1f}s", border_style="green", padding=(1, 2))
        )
    else:
        print(f"\n{C.GREEN}{C.BOLD}🎯 Answer (⏱ {elapsed:.1f}s):{C.END}\n{display_text}")


def print_banner():
    if RICH_AVAILABLE:
        console.print(
            Panel.fit(
                "[bold]🚀 Master Agent System[/bold]\n"
                "SQL Analyst  •  ETL Analyst  •  General Assistant\n"
                "[dim]Type in Arabic, English, or a mix — I'll match your style.[/dim]",
                border_style="cyan",
            )
        )
    else:
        print("\n" + "*" * 60)
        print(f"{C.BOLD}🚀 Master Agent System Initialized successfully!{C.END}")
        print("Type your question in Arabic, English, or a mix of both.")
        print("*" * 60)


def _run_with_status(label: str, fn):
    if RICH_AVAILABLE:
        with console.status(f"[bold cyan]{label}...", spinner="dots"):
            return fn()
    say(f"{label}...", "cyan", C.CYAN)
    return fn()


def main():
    if not cfg.GROQ_API_KEY:
        say("❌ GROQ_API_KEY is missing from your .env — add it and re-run.", "bold red", C.RED)
        sys.exit(1)

    print_banner()
    logger.info("%s New session started %s %s", "=" * 40, datetime.now().isoformat(), "=" * 40)

    app = build_graph()
    thread_id = f"session-{int(time.time())}"
    graph_config = {"configurable": {"thread_id": thread_id}}

    while True:
        print()
        try:
            if RICH_AVAILABLE:
                q = console.input("[bold]🗣️  You:[/bold] ").strip()
            else:
                q = input(f"{C.BOLD}🗣️  You: {C.END}").strip()
        except (EOFError, KeyboardInterrupt):
            say("\nGoodbye! 👋", "bold", C.BOLD)
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit", "خروج"):
            say("Goodbye! 👋", "bold", C.BOLD)
            break

        start = time.time()
        language = "en"
        try:
            result = _run_with_status(
                "🧭 Thinking",
                lambda: app.invoke({"messages": [HumanMessage(content=q)]}, config=graph_config),
            )
            answer = result.get("final_answer", "No answer generated.")
            language = result.get("language", "en")
            route = result.get("route")
            if route:
                emoji, name, style = cfg.ROUTE_LABELS.get(route, ("🧭", route, "bold"))
                say(f"\n{emoji} Routed to {name}  (confidence {result.get('confidence', 0):.2f})", style, C.GREEN)
        except Exception as e:
            logger.exception("Unhandled error in graph execution")
            answer = f"Unexpected error: {e}"

        elapsed = time.time() - start
        render_answer(answer, elapsed, language)
        logger.info("Answered in %.2fs | answer_len=%d", elapsed, len(answer))


if __name__ == "__main__":
    main()
