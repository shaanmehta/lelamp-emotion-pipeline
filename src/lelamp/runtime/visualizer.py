"""A throwaway terminal lamp, so you can see the state change instead of reading
JSON.

The point of the structured output is that a body consumes it. The fastest way
to show the contract is sufficient is to drive something that looks like a body
using nothing but those fields. This renderer reads only `behavior`, plus affect
for the numeric readouts. If the schema were missing something a robot needs,
this file wouldn't be able to draw.
"""
from __future__ import annotations

import colorsys

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Posture -> lamp art. Column 0 is the base; the head leans.
POSTURES = {
    "upright":     ["   ___   ", "  /###\\  ", "  |   |  ", "   \\|/   ", "  __|__  "],
    "lean_in":     ["  ___    ", " /###\\   ", " |   |   ", "   \\\\    ", "  __|__  "],
    "lean_back":   ["    ___  ", "   /###\\ ", "   |   | ", "    //   ", "  __|__  "],
    "droop":       ["         ", "  ___    ", " /###\\   ", "  \\_|    ", "  __|__  "],
    "recoil_pose": ["     ___ ", "    /###\\", "    |   |", "    ///  ", "  __|__  "],
}


def hsv_hex(hue_deg: float, sat: float, val: float) -> str:
    r, g, b = colorsys.hsv_to_rgb((hue_deg % 360) / 360.0,
                                  max(0.0, min(1.0, sat)),
                                  max(0.08, min(1.0, val)))
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _meter(v: float, width: int = 18, lo: float = -1.0, hi: float = 1.0) -> Text:
    """Centre-anchored bar for signed quantities, so zero is visually zero."""
    t = Text()
    mid = width // 2
    pos = int(round((v - lo) / (hi - lo) * (width - 1)))
    for i in range(width):
        if i == mid and pos != mid:
            t.append("│", style="grey42")
        elif (pos >= mid and mid <= i <= pos) or (pos < mid and pos <= i <= mid):
            t.append("█", style="cyan" if v >= 0 else "magenta")
        else:
            t.append("░", style="grey30")
    return t


def _bar(v: float, width: int = 18, style: str = "green") -> Text:
    n = int(round(max(0.0, min(1.0, v)) * width))
    return Text("█" * n, style=style) + Text("░" * (width - n), style="grey30")


def render(state: dict | None, log: list[str], title: str = "LeLamp") -> Panel:
    if state is None:
        return Panel(Text("waiting for stream...", style="grey50"), title=title)

    beh, aff = state["behavior"], state["affect"]
    colour = hsv_hex(beh["light"]["hue_deg"], beh["light"]["sat"],
                     beh["light"]["value"])
    art = POSTURES.get(beh["motion"]["posture"], POSTURES["upright"])

    lamp = Text()
    for i, row in enumerate(art):
        lamp.append(row, style=f"bold {colour}" if i < 3 else "grey58")
        lamp.append("\n")

    t = Table.grid(padding=(0, 1))
    t.add_column(justify="left", style="grey62", width=10)
    t.add_column(justify="left", width=20)
    t.add_column(justify="left")

    emo = aff["emotion"] or "—"
    tier = state["tier"]
    t.add_row("tier", Text(tier, style="yellow" if tier == "reflexive" else "bold green"),
              Text(f"t+{state['t_emit_ms']:.0f} ms", style="grey62"))
    t.add_row("emotion", Text(emo.upper(), style=f"bold {colour}"),
              Text("UNCERTAIN", style="bold red") if aff["uncertain"]
              else Text(""))
    t.add_row("confidence", _bar(aff["confidence"]),
              Text(f"{aff['confidence']:.2f}", style="grey70"))
    t.add_row("valence", _meter(aff["valence"]), Text(f"{aff['valence']:+.2f}"))
    t.add_row("arousal", _meter(aff["arousal"]), Text(f"{aff['arousal']:+.2f}"))
    t.add_row("", Text(""), Text(""))
    t.add_row("intent", Text(beh["intent"], style=f"bold {colour}"),
              Text(f"priority {beh['priority']}", style="grey62"))
    t.add_row("motion", Text(f"{beh['motion']['posture']} / {beh['motion']['gaze']}",
                             style="grey70"),
              Text(f"amp {beh['motion']['amplitude']:.2f}  "
                   f"spd {beh['motion']['speed']:.2f}", style="grey62"))
    t.add_row("light", Text(f"■■■  hue {beh['light']['hue_deg']:.0f}°", style=colour),
              Text(f"sat {beh['light']['sat']:.2f}  val {beh['light']['value']:.2f}"
                   + (f"  pulse {beh['light']['pulse_hz']:.1f}Hz"
                      if beh["light"]["pulse_hz"] else ""), style="grey62"))
    t.add_row("timing", Text(f"hold {beh['hold_ms']} ms", style="grey70"),
              Text(f"decay {beh['decay_ms']} ms  expr {beh['expressivity']:.2f}",
                   style="grey62"))

    head = Table.grid(padding=(0, 3))
    head.add_column()
    head.add_column()
    head.add_row(lamp, t)

    body = [head]
    ev = state.get("evidence")
    if ev:
        body.append(Text(
            f"evidence  dominant={ev['dominant_modality']}  "
            f"text_margin={ev['text_margin']:.2f}  "
            f"vision_margin={ev['vision_margin']:.2f}  "
            f"agreement={ev['modality_agreement']:.2f}", style="grey50"))
    bl = state.get("belief")
    if bl and bl.get("switched") is not None:
        body.append(Text(
            f"belief    raw={bl['raw_emotion']}  committed={aff['emotion']}  "
            f"switched={bl['switched']}  suppressed={bl['suppressed_switches']}",
            style="grey50"))
    sp = state.get("speech")
    if sp and sp.get("text"):
        body.append(Text(f'\n  💬 "{sp["text"]}"', style=f"italic bold {colour}"))
    elif sp and not sp.get("allowed"):
        body.append(Text("\n  (staying quiet — not confident enough)",
                         style="italic grey50"))

    if log:
        body.append(Text("\n" + "\n".join(log[-9:]), style="grey42"))

    return Panel(Group(*body), title=title, border_style=colour)


class LampView:
    """Thin wrapper so the demo can push states without knowing about rich."""

    def __init__(self, title: str = "LeLamp — live perception"):
        self.console = Console()
        self.title = title
        self.log: list[str] = []
        self._live: Live | None = None
        self.state: dict | None = None

    def __enter__(self):
        self._live = Live(render(None, [], self.title), console=self.console,
                          refresh_per_second=12, transient=False)
        self._live.__enter__()
        return self

    def __exit__(self, *a):
        if self._live:
            self._live.__exit__(*a)

    def push(self, state: dict, note: str | None = None):
        self.state = state
        if note:
            self.log.append(note)
        if self._live:
            self._live.update(render(state, self.log, self.title))
