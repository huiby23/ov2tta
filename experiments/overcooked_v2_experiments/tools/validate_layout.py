#!/usr/bin/env python3
"""Validate Overcooked V2 layout connectivity/reachability and draw a small SVG."""
from __future__ import annotations

import argparse
import csv
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from jaxmarl.environments.overcooked_v2 import layouts as layout_module
from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts

Position = Tuple[int, int]
WALKABLE = {" ", "A"}
DEFAULT_REQUIRED_OBJECTS = ("0", "1", "R", "P", "B", "X")
CELL_COLORS = {
    "W": ("#354052", "#ffffff"),
    "A": ("#2F80ED", "#ffffff"),
    "P": ("#111827", "#ffffff"),
    "B": ("#F2F4F7", "#111827"),
    "X": ("#27AE60", "#ffffff"),
    "R": ("#F2994A", "#ffffff"),
    "L": ("#9B51E0", "#ffffff"),
    "0": ("#F2C94C", "#111827"),
    "1": ("#EB5757", "#ffffff"),
    "2": ("#56CCF2", "#111827"),
    "3": ("#BB6BD9", "#ffffff"),
    " ": ("#ffffff", "#111827"),
}


def _raw_grid(layout_name: str) -> List[str]:
    if not hasattr(layout_module, layout_name):
        raise KeyError(f"No raw layout string named {layout_name!r} in layouts.py")
    raw = getattr(layout_module, layout_name)
    return [row for row in raw.split("\n") if row]


def _neighbors(pos: Position) -> Iterable[Position]:
    x, y = pos
    yield x + 1, y
    yield x - 1, y
    yield x, y + 1
    yield x, y - 1


def _walkable_positions(rows: Sequence[str]) -> Tuple[List[Position], Dict[Position, str]]:
    agents: List[Position] = []
    walkable: Dict[Position, str] = {}
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in WALKABLE:
                walkable[(x, y)] = ch
            if ch == "A":
                agents.append((x, y))
    return agents, walkable


def _components(walkable: Dict[Position, str]) -> Tuple[List[List[Position]], Dict[Position, int]]:
    seen = set()
    comps: List[List[Position]] = []
    comp_id: Dict[Position, int] = {}
    for start in walkable:
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp: List[Position] = []
        while q:
            pos = q.popleft()
            comp_id[pos] = len(comps)
            comp.append(pos)
            for nxt in _neighbors(pos):
                if nxt in walkable and nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        comps.append(comp)
    return comps, comp_id


def _object_positions(rows: Sequence[str], objects: Sequence[str]) -> List[Tuple[str, Position]]:
    wanted = set(objects)
    found: List[Tuple[str, Position]] = []
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in wanted:
                found.append((ch, (x, y)))
    return found


def _csv_list(items: Iterable[object]) -> str:
    return ";".join(str(x) for x in items)


def _write_svg(rows: Sequence[str], out_path: Path, title: str) -> None:
    cell = 54
    margin = 34
    title_h = 58
    width = max(len(r) for r in rows) * cell + margin * 2
    height = len(rows) * cell + margin * 2 + title_h
    font = "Avenir, Helvetica, Arial, sans-serif"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="42" font-family="{font}" font-size="28" font-weight="700" fill="#111827">{title}</text>',
    ]
    y0 = margin + title_h
    for y, row in enumerate(rows):
        for x in range(max(len(r) for r in rows)):
            ch = row[x] if x < len(row) else " "
            bg, fg = CELL_COLORS.get(ch, ("#ffffff", "#111827"))
            rx = margin + x * cell
            ry = y0 + y * cell
            parts.append(f'<rect x="{rx}" y="{ry}" width="{cell}" height="{cell}" rx="6" fill="{bg}" stroke="#D0D5DD" stroke-width="1.5"/>')
            if ch == "A":
                parts.append(f'<circle cx="{rx + cell / 2}" cy="{ry + cell / 2}" r="{cell * 0.33}" fill="{bg}" stroke="#1D4ED8" stroke-width="3"/>')
            if ch != " ":
                parts.append(f'<text x="{rx + cell / 2}" y="{ry + cell / 2 + 8}" font-family="{font}" font-size="22" font-weight="800" fill="{fg}" text-anchor="middle">{ch}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def validate(layout_name: str, output_dir: Path, required_objects: Sequence[str]) -> int:
    if layout_name not in overcooked_v2_layouts:
        raise KeyError(f"Layout {layout_name!r} is not registered")
    layout = overcooked_v2_layouts[layout_name]
    rows = _raw_grid(layout_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    agents, walkable = _walkable_positions(rows)
    comps, comp_id = _components(walkable)
    agent_comp_ids = [comp_id.get(pos, -1) for pos in agents]
    same_component = len(set(agent_comp_ids)) == 1 and len(agents) == 2

    connected_csv = output_dir / "connected_component_check.csv"
    with connected_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layout",
                "width",
                "height",
                "num_agents",
                "agent_positions",
                "agent_component_ids",
                "same_walkable_component",
                "num_components",
                "component_sizes",
                "num_ingredients",
                "possible_recipes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "layout": layout_name,
                "width": layout.width,
                "height": layout.height,
                "num_agents": len(agents),
                "agent_positions": _csv_list(agents),
                "agent_component_ids": _csv_list(agent_comp_ids),
                "same_walkable_component": same_component,
                "num_components": len(comps),
                "component_sizes": _csv_list(len(c) for c in comps),
                "num_ingredients": layout.num_ingredients,
                "possible_recipes": _csv_list(layout.possible_recipes),
            }
        )

    reachable_csv = output_dir / "reachable_objects_check.csv"
    object_rows = []
    for ch, pos in _object_positions(rows, required_objects):
        neighbors = [p for p in _neighbors(pos) if p in walkable]
        neighbor_comps = sorted({comp_id[p] for p in neighbors})
        reachable_by_agents = [i for i, c in enumerate(agent_comp_ids) if c in neighbor_comps]
        object_rows.append(
            {
                "layout": layout_name,
                "object": ch,
                "position": pos,
                "walkable_neighbors": _csv_list(neighbors),
                "neighbor_component_ids": _csv_list(neighbor_comps),
                "reachable_agent_indices": _csv_list(reachable_by_agents),
                "reachable_by_both_agents": len(reachable_by_agents) == len(agents),
            }
        )
    with reachable_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layout",
                "object",
                "position",
                "walkable_neighbors",
                "neighbor_component_ids",
                "reachable_agent_indices",
                "reachable_by_both_agents",
            ],
        )
        writer.writeheader()
        writer.writerows(object_rows)

    svg_path = output_dir / f"{layout_name}.svg"
    _write_svg(rows, svg_path, layout_name)

    all_required_found = set(required_objects).issubset({r["object"] for r in object_rows})
    all_objects_reachable = all(r["reachable_by_both_agents"] for r in object_rows)
    ok = same_component and all_required_found and all_objects_reachable

    summary_path = output_dir / "layout_validation_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                f"# Layout validation: {layout_name}",
                "",
                f"- generated_at: `{datetime.now().isoformat()}`",
                f"- parser_registered: `{layout_name in overcooked_v2_layouts}`",
                f"- same_walkable_component: `{same_component}`",
                f"- all_required_objects_found: `{all_required_found}`",
                f"- all_required_objects_reachable_by_both_agents: `{all_objects_reachable}`",
                f"- possible_recipes: `{layout.possible_recipes}`",
                f"- connected_component_check: `{connected_csv}`",
                f"- reachable_objects_check: `{reachable_csv}`",
                f"- svg: `{svg_path}`",
                "",
                "## Raw layout",
                "",
                "```text",
                *rows,
                "```",
            ]
        )
    )
    print(summary_path)
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="open_cramped_room_v2")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/open_cramped_room_v2_validation"))
    parser.add_argument("--required-objects", nargs="*", default=list(DEFAULT_REQUIRED_OBJECTS))
    args = parser.parse_args()
    return validate(args.layout, args.output_dir, tuple(args.required_objects))


if __name__ == "__main__":
    raise SystemExit(main())
