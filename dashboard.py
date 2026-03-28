"""AETHER COMMAND DECK — Live Terminal Dashboard.

Full-screen Rich UI with real-time training metrics, CSSC horizon
visualisation, GGR gradient flow, and baseline delta comparison.

Layout
------
┌─────────────────────────── AETHER COMMAND DECK ───────────────────────────┐
│ [Progress bar ─────────────────────────────── step / max_steps]            │
├────────────────────────────┬───────────────────────────────────────────────┤
│  CSSC HORIZON              │  GGR GRADIENT ENGINE                          │
│  (temporal arc heatmap)    │  (expert flow diagram)                        │
├────────────────────────────┴───────────────────────────────────────────────┤
│  METRICS                                                                    │
│  CE  PPL  GSI  CE_eff  Throughput  Δ vs Baseline   VRAM                    │
├────────────────────────────────────────────────────────────────────────────┤
│  STATUS BAR   step  lr  epoch  ms/step  elapsed                             │
└────────────────────────────────────────────────────────────────────────────┘

Imports no torch at module level — zero GPU impact.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.style import Style
from rich.table import Table
from rich.text import Text

from aether2_config import Aether2Config
from aether2_train import TrainingMetrics


# ─────────────────────────────────────────────────────────────────────────────
# Unicode helpers
# ─────────────────────────────────────────────────────────────────────────────

_BLOCKS = " ▁▂▃▄▅▆▇█"

def _sparkline(values: list[float], width: int = 20) -> str:
    """Render a list of floats as a Unicode block sparkline."""
    if not values:
        return " " * width
    mn, mx = min(values), max(values)
    span = mx - mn if mx != mn else 1.0
    chars = [_BLOCKS[int((v - mn) / span * (len(_BLOCKS) - 1))] for v in values]
    return "".join(chars[-width:]).ljust(width)


def _pct_bar(pct: float, width: int = 20, style_name: str = "green") -> Text:
    """Render a percentage as a coloured Unicode bar."""
    pct = max(0.0, min(1.0, pct))
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    colour = "green" if pct < 0.7 else ("yellow" if pct < 0.9 else "red")
    return Text(f"[{bar}] {pct*100:.1f}%", style=colour)


def _delta_text(delta_pct: float) -> Text:
    """Render Aether2 vs Baseline delta as coloured text."""
    if math.isnan(delta_pct):
        return Text("─", style="dim")
    if delta_pct > 0:
        return Text(f"▲ {delta_pct:+.1f}% better", style="bold green")
    elif delta_pct < -1:
        return Text(f"▼ {abs(delta_pct):.1f}% worse", style="bold red")
    else:
        return Text(f"≈ {delta_pct:+.1f}% parity", style="yellow")


def _gsi_colour(gsi: float) -> str:
    if gsi >= 0.85: return "bold green"
    if gsi >= 0.6:  return "yellow"
    return "bold red"


# ─────────────────────────────────────────────────────────────────────────────
# CSSC Horizon Visualisation
# ─────────────────────────────────────────────────────────────────────────────

_ARC_CHARS = ["·", "╌", "─", "═", "▬"]   # increasing correlation strength


def _render_cssc_horizon(scale_weights: list[float], width: int = 44) -> Panel:
    """Draw a horizon map of temporal correlation arcs.

    Three rows correspond to token / sentence / block scale.
    Arc brightness (char density) encodes real-time correlation weight.
    """
    token_w, sent_w, block_w = scale_weights if len(scale_weights) == 3 else [0.5, 0.3, 0.2]

    def _arc_row(label: str, weight: float, n_arcs: int, colour: str) -> Text:
        """Draw one horizon row: arcs fade in towards the right (temporal)."""
        t = Text()
        t.append(f" {label:<7}", style="bold white")
        t.append("│", style="dim white")

        idx = min(int(weight * (len(_ARC_CHARS) - 1) + 0.5), len(_ARC_CHARS) - 1)
        char = _ARC_CHARS[idx]

        # Build a "horizon" — arcs get denser near the right (current time)
        positions = width - 9
        for i in range(positions):
            progress = (i + 1) / positions
            # Hyperbolic density: more arcs close to current position
            density = 1.0 / (1.0 + 0.5 * (1.0 - progress) * 10)
            if progress > (1.0 - weight) and (i % max(1, int(1.0 / (weight + 0.01))) == 0):
                brightness = min(int(density * 4), 4)
                c = _ARC_CHARS[min(idx, brightness)]
                t.append(c, style=colour)
            else:
                t.append("·", style="dim " + colour)
        t.append(f" {weight:.2f}", style="bold " + colour)
        return t

    lines = [
        Text("  CSSC HORIZON — Temporal Correlation Arcs", style="bold cyan"),
        Text(""),
        Text("  Scale    │" + " " * (width - 9) + " Weight", style="dim white"),
        Text("  ─────────┼" + "─" * (width - 9) + "────────", style="dim white"),
        _arc_row("TOKEN  ", token_w, 8,  "bright_cyan"),
        _arc_row("SENT   ", sent_w,  5,  "bright_blue"),
        _arc_row("BLOCK  ", block_w, 3,  "bright_magenta"),
        Text(""),
        Text.from_markup(
            f"  α-decay={0.5:.2f} │ scale blend: "
            f"[cyan]T={token_w:.2f}[/] [blue]S={sent_w:.2f}[/] [magenta]B={block_w:.2f}[/]"
        ),
    ]
    content = Text("\n").join(lines)
    return Panel(content, title="[bold cyan]▸ CSSC HORIZON[/]",
                 border_style="cyan", padding=(0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# GGR Engine Visualisation
# ─────────────────────────────────────────────────────────────────────────────

_EXPERT_COLOURS = {
    "Math":    "bold yellow",
    "Code":    "bold green",
    "Logic":   "bold blue",
    "General": "bold magenta",
}
_FLOW_CHARS = ["▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]


def _flow_bar(weight: float, width: int = 12) -> str:
    """Render a horizontal flow bar with sub-character precision."""
    filled = weight * width
    full   = int(filled)
    frac   = filled - full
    bar    = "█" * full
    if frac > 0 and full < width:
        bar += _FLOW_CHARS[int(frac * len(_FLOW_CHARS))]
    return bar.ljust(width)


def _render_ggr_engine(
    expert_names: list[str],
    expert_loads: list[float],
    gsi: float,
    entropy: float,
) -> Panel:
    """Draw GGR expert flow diagram with box-drawing pipes."""
    N = len(expert_names)
    loads = expert_loads if len(expert_loads) == N else [1.0/N] * N
    total = sum(loads) + 1e-8
    normed = [l / total for l in loads]

    lines: list[Text] = [
        Text("  GGR GRADIENT ENGINE — Expert Routing", style="bold green"),
        Text(""),
        Text(f"  Input Entropy: {entropy:.3f}  │  GSI: {gsi:.3f}",
             style=_gsi_colour(gsi)),
        Text(""),
        Text("  ┌──────────┐", style="dim white"),
        Text("  │  Input x │", style="white"),
        Text("  └────┬─────┘", style="dim white"),
        Text("       │  Entropy-conditioned router", style="dim"),
        Text("  ─────┴──────────────────────────────", style="dim white"),
    ]

    # Pipe splits
    for i, (name, load) in enumerate(zip(expert_names, normed)):
        colour = _EXPERT_COLOURS.get(name, "white")
        bar    = _flow_bar(load, 10)
        pct    = load * 100

        connector = "├" if i < N - 1 else "└"
        lines.append(
            Text(f"  {connector}──▶ [{name:<7}] {bar} {pct:4.1f}%", style=colour)
        )

    lines += [
        Text(""),
        Text("  ─────────────────────────────────────", style="dim white"),
        Text("  └──▶  ┌──────────────┐  Soft Merge", style="dim white"),
        Text("        │   Output out │", style="white"),
        Text("        └──────────────┘", style="dim white"),
    ]

    content = Text("\n").join(lines)
    return Panel(content, title="[bold green]▸ GGR ENGINE[/]",
                 border_style="green", padding=(0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Panel
# ─────────────────────────────────────────────────────────────────────────────

def _render_metrics(m: TrainingMetrics, history: dict[str, deque]) -> Panel:
    tbl = Table.grid(expand=True, padding=(0, 2))
    tbl.add_column("Metric",  style="bold dim", min_width=14)
    tbl.add_column("Value",   min_width=12)
    tbl.add_column("Trend",   min_width=22)

    def _row(label: str, val_text: str | Text, key: str) -> None:
        spark = _sparkline(list(history[key]))
        tbl.add_row(label, val_text, Text(spark, style="cyan"))

    ce_text = f"{m.ce_loss:.4f}" if not math.isnan(m.ce_loss) else "─"
    _row("CE Loss",   ce_text,  "ce")

    ppl_text = f"{m.ppl:.2f}" if not math.isnan(m.ppl) else "─"
    _row("Perplexity", ppl_text, "ppl")

    gsi_text = Text(f"{m.ggr_gsi:.4f}", style=_gsi_colour(m.ggr_gsi))
    _row("GSI",    gsi_text, "gsi")

    ce_eff = Text(f"{m.cssc_ce:.4f}", style="cyan")
    _row("CE (context eff.)", ce_eff, "cssc_ce")

    tok_s = f"{m.tokens_per_sec:,.0f} tok/s"
    _row("Throughput", tok_s, "tok_s")

    vram_bar = _pct_bar(
        m.vram_alloc_gib / (m.vram_total_gib + 1e-6),
        width=16,
    )
    _row("VRAM",  f"{m.vram_alloc_gib:.2f}/{m.vram_total_gib:.1f} GiB", "vram")

    # Baseline delta row (prominent)
    tbl.add_row(
        Text("Δ vs Baseline", style="bold white"),
        _delta_text(m.delta_pct),
        Text(_sparkline(list(history["delta"]), 22), style="yellow"),
    )

    return Panel(tbl, title="[bold white]▸ METRICS[/]",
                 border_style="white", padding=(0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Header & Status
# ─────────────────────────────────────────────────────────────────────────────

_HEADER = Text.assemble(
    ("  ╔═══════════════════════════════════════╗\n", "dim cyan"),
    ("  ║   ", "dim cyan"),
    ("A E T H E R  2", "bold cyan"),
    ("   COMMAND DECK   ║\n", "dim cyan"),
    ("  ╚═══════════════════════════════════════╝", "dim cyan"),
)


def _render_status(m: TrainingMetrics, start_time: float, max_steps: int) -> Text:
    elapsed = time.time() - start_time
    h = int(elapsed // 3600)
    mi = int((elapsed % 3600) // 60)
    s  = int(elapsed % 60)
    elapsed_str = f"{h:02d}:{mi:02d}:{s:02d}"

    pct = m.step / max(max_steps, 1) * 100
    return Text(
        f"  Step {m.step:,}/{max_steps:,} ({pct:.1f}%)  │  "
        f"LR={m.lr:.2e}  │  "
        f"Epoch {m.epoch}  │  "
        f"{m.ms_per_step:.0f}ms/step  │  "
        f"GradNorm={m.grad_norm:.3f}  │  "
        f"Elapsed {elapsed_str}",
        style="dim white",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard class
# ─────────────────────────────────────────────────────────────────────────────

class Aether2Dashboard:
    """Aether Command Deck — live Rich terminal dashboard.

    Usage
    -----
    with Aether2Dashboard(cfg, max_steps=50_000) as dash:
        train(cfg, metrics_callback=dash.update)
    """

    HISTORY_LEN = 80   # sparkline window

    def __init__(self, cfg: Aether2Config, max_steps: int = 50_000) -> None:
        self.cfg       = cfg
        self.max_steps = max_steps
        self.start_time = time.time()
        self._last_m   = TrainingMetrics()

        # History buffers for sparklines
        self._history: dict[str, deque] = {
            k: deque(maxlen=self.HISTORY_LEN)
            for k in ("ce", "ppl", "gsi", "cssc_ce", "tok_s", "vram", "delta")
        }

        # CSSC scale weights (from last metrics update)
        self._cssc_weights: list[float] = [0.5, 0.3, 0.2]
        self._ggr_names   = list(cfg.ggr_expert_names)
        self._ggr_loads   = [1.0 / cfg.ggr_n_experts] * cfg.ggr_n_experts
        self._gsi         = 1.0
        self._entropy     = 0.5

        self._console = Console(highlight=False)
        self._live: Optional[Live] = None

    # ── Build the full layout ────────────────────────────────────────────────

    def _build_layout(self) -> Layout:
        m  = self._last_m
        layout = Layout()

        layout.split_column(
            Layout(name="header",  size=5),
            Layout(name="top",     size=18),
            Layout(name="metrics", size=12),
            Layout(name="status",  size=1),
        )

        layout["top"].split_row(
            Layout(name="cssc",   ratio=1),
            Layout(name="ggr",    ratio=1),
        )

        # Header
        layout["header"].update(Panel(_HEADER, border_style="cyan", padding=(0, 0)))

        # CSSC panel
        layout["cssc"].update(
            _render_cssc_horizon(self._cssc_weights)
        )

        # GGR panel
        layout["ggr"].update(
            _render_ggr_engine(
                self._ggr_names, self._ggr_loads, self._gsi, self._entropy
            )
        )

        # Metrics panel
        layout["metrics"].update(
            _render_metrics(m, self._history)
        )

        # Status bar
        layout["status"].update(
            _render_status(m, self.start_time, self.max_steps)
        )

        return layout

    # ── Rich renderable protocol ─────────────────────────────────────────────

    def __rich_console__(self, console, options):
        """Called by Rich on every refresh tick — always yields a fresh layout."""
        yield self._build_layout()

    # ── Public API ───────────────────────────────────────────────────────────

    def update(self, m: TrainingMetrics) -> None:
        """Receive new metrics from the training loop and store them.

        The Live display refreshes automatically at 4 Hz via __rich_console__;
        no explicit live.update() call is needed here.
        """
        self._last_m = m

        # Update history
        if not math.isnan(m.ce_loss):
            self._history["ce"].append(m.ce_loss)
        if not math.isnan(m.ppl):
            self._history["ppl"].append(min(m.ppl, 500.0))
        self._history["gsi"].append(m.ggr_gsi)
        self._history["cssc_ce"].append(m.cssc_ce)
        self._history["tok_s"].append(m.tokens_per_sec)
        self._history["vram"].append(m.vram_alloc_gib)
        if not math.isnan(m.delta_pct):
            self._history["delta"].append(m.delta_pct)

        # Update CSSC / GGR state
        self._ggr_loads = m.ggr_expert_load
        self._gsi       = m.ggr_gsi
        self._entropy   = getattr(m, "entropy", 0.5)

    def __enter__(self) -> "Aether2Dashboard":
        # Pass `self` as the renderable — Rich calls __rich_console__ on every
        # refresh tick, so the elapsed timer and VRAM update continuously even
        # during long training steps (not just when metrics arrive).
        self._live = Live(
            self,
            console=self._console,
            refresh_per_second=2,   # 2 Hz reduces CPU overhead vs 4 Hz
            screen=False,           # inline mode: no terminal capture → no freeze/clip
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._live is not None:
            self._live.__exit__(*args)
            self._live = None
        # Print final summary after leaving full-screen
        self._print_final_summary()

    def _print_final_summary(self) -> None:
        m = self._last_m
        self._console.print("\n")
        self._console.rule("[bold cyan]AETHER 2 — Training Complete[/]")
        tbl = Table(title="Final Metrics", border_style="cyan", show_lines=True)
        tbl.add_column("Metric", style="bold")
        tbl.add_column("Value")
        rows = [
            ("Total Steps",    f"{m.step:,}"),
            ("Final CE Loss",  f"{m.ce_loss:.4f}"),
            ("Final PPL",      f"{m.ppl:.2f}"),
            ("GSI",            f"{m.ggr_gsi:.4f}"),
            ("Context Eff.",   f"{m.cssc_ce:.4f}"),
            ("Throughput",     f"{m.tokens_per_sec:,.0f} tok/s"),
            ("Δ vs Baseline",  f"{m.delta_pct:+.1f}%"),
        ]
        for k, v in rows:
            tbl.add_row(k, v)
        self._console.print(tbl)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone preview mode
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """Run the dashboard with synthetic data for presentation preview."""
    import random

    cfg = Aether2Config()
    max_steps = 1000

    print("Launching Aether Command Deck preview (Ctrl+C to exit)…")
    time.sleep(0.5)

    with Aether2Dashboard(cfg, max_steps=max_steps) as dash:
        for step in range(max_steps + 1):
            # Synthetic metrics that look plausible during a real training run
            ce  = 8.0 * math.exp(-step / 300) + 2.0 + random.gauss(0, 0.05)
            gsi = min(1.0, 0.3 + step / 500 + random.gauss(0, 0.02))
            loads = [random.random() for _ in range(4)]
            s = sum(loads)
            loads = [l / s for l in loads]

            m = TrainingMetrics(
                step=step,
                epoch=step // 250,
                lr=3e-4 * max(0.01, 1 - step / max_steps),
                ce_loss=ce,
                ppl=math.exp(min(ce, 20)),
                ggr_aux_loss=0.002 + random.gauss(0, 0.0005),
                total_loss=ce + 0.002,
                vram_alloc_gib=7.2 + random.gauss(0, 0.05),
                vram_total_gib=16.0,
                ms_per_step=280 + random.gauss(0, 10),
                tokens_per_sec=4096 * 1000 / 280,
                grad_norm=0.8 + random.gauss(0, 0.1),
                curvature_mean=0.9 + step / 20000,
                ggr_expert_load=loads,
                ggr_gsi=gsi,
                cssc_ce=0.4 + step / 2500,
                baseline_loss=ce + 0.5 + random.gauss(0, 0.05),
                delta_pct=(0.5 / (ce + 0.5)) * 100,
            )

            dash.update(m)
            time.sleep(0.08)   # ~12 fps synthetic


if __name__ == "__main__":
    _demo()
