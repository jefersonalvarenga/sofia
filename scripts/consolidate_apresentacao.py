"""
Consolidate APRESENTAÇÃO eval reports into a single per-version markdown file.

Reads existing per-round reports under:
    ~/Documents/easyscale/kb/07-MVP/Tech/Tests/Agente Greeting/

Outputs:
    apresentacao v23 consolidado (data).md  (3 temps × 3 rounds = 9 runs)
    apresentacao v25 consolidado (data).md  (1 temp × 3 rounds = 3 runs)

The new format groups all runs of the same test case in a single table,
keeps user prompt and few-shots shown once per case, and moves the
SYSTEM_PROMPT and full configuration to the end of the document.
"""

from __future__ import annotations

import os
import re
import glob
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any


FOLDER = os.path.expanduser(
    "~/Documents/easyscale/kb/07-MVP/Tech/Tests/Agente Greeting"
)


# ---------------------------------------------------------------------------
# Parser for individual round reports.
# ---------------------------------------------------------------------------

CASE_HEADER_RE = re.compile(r"^### (Q\d+\.\d+) — (.+?)$", re.MULTILINE)


def parse_round_file(path: str) -> Dict[str, Any]:
    """Extract per-case data from a single round report."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    # Latency from header
    lat_match = re.search(
        r"Latência:\*\*\s*min=(\d+)ms\s+p50=(\d+)ms\s+max=(\d+)ms", text
    )
    header_latency = (
        {"min": int(lat_match.group(1)), "p50": int(lat_match.group(2)), "max": int(lat_match.group(3))}
        if lat_match
        else None
    )

    # Auto-score totals
    yes_total = len(re.findall(r"heurístico:\*\*\s*`YES`", text))
    no_total = len(re.findall(r"heurístico:\*\*\s*`NO`", text))

    # Split into case blocks
    cases: Dict[str, Dict[str, Any]] = {}
    matches = list(CASE_HEADER_RE.finditer(text))
    for i, m in enumerate(matches):
        case_id = m.group(1)
        case_label = m.group(2)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        # Latency for this case
        lat = re.search(r"_Latência:\s*(\d+)ms_", block)
        case_latency = int(lat.group(1)) if lat else None

        # Quadrant from "## ..." preceding the case (walk back)
        quadrant_match = None
        for q_m in re.finditer(r"^## (.+?)$", text[: m.start()], re.MULTILINE):
            if q_m.group(1).startswith("Configuração") or q_m.group(1).startswith(
                "Critério"
            ) or q_m.group(1).startswith("Score") or q_m.group(1).startswith("Anexo"):
                continue
            quadrant_match = q_m.group(1)
        quadrant = quadrant_match

        expected = re.search(r"\*\*Expectativa:\*\*\s*`(\w+)`", block)
        expected_val = expected.group(1) if expected else None

        # Input (user prompt) — between '```' after **Input** and next '```'
        user_prompt_match = re.search(
            r"\*\*Input\s*\(user prompt enviado ao LLM\):\*\*\s*\n+```\s*\n(.*?)\n```",
            block,
            re.DOTALL,
        )
        user_prompt = user_prompt_match.group(1) if user_prompt_match else ""

        # Output
        out_match = re.search(
            r"\*\*Output:\*\*\s*\n+> (.+?)(?:\n|$)", block
        )
        output = out_match.group(1).strip() if out_match else ""
        if "_(vazio" in output:
            output = "(vazio)"

        # Reasoning
        reasoning_match = re.search(
            r"\*\*Reasoning do modelo \(depuração\):\*\*\s*\n+```\s*\n(.*?)\n```",
            block,
            re.DOTALL,
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        # Auto verdict
        verdict_match = re.search(r"heurístico:\*\*\s*`(YES|NO)`", block)
        verdict = verdict_match.group(1) if verdict_match else "?"

        cases[case_id] = {
            "label": case_label,
            "quadrant": quadrant or "",
            "expected": expected_val or "",
            "user_prompt": user_prompt,
            "output": output,
            "reasoning": reasoning,
            "verdict": verdict,
            "latency_ms": case_latency,
        }

    # Extract SYSTEM_PROMPT (between '### System prompt' and next '### ')
    sysprompt_match = re.search(
        r"### System prompt\s*\n+```\s*\n(.*?)\n```",
        text,
        re.DOTALL,
    )
    system_prompt = sysprompt_match.group(1) if sysprompt_match else ""

    # Model config
    model = re.search(r"\*\*Modelo:\*\*\s*`(.+?)`", text)
    temp = re.search(r"\*\*Temperature:\*\*\s*`(.+?)`", text)
    max_tok = re.search(r"\*\*max_tokens:\*\*\s*`(.+?)`", text)

    return {
        "cases": cases,
        "system_prompt": system_prompt,
        "header_latency": header_latency,
        "auto_yes": yes_total,
        "auto_no": no_total,
        "model": model.group(1) if model else "",
        "temperature": temp.group(1) if temp else "",
        "max_tokens": max_tok.group(1) if max_tok else "",
    }


# ---------------------------------------------------------------------------
# Few-shots fixture (same across all rounds — kept here for the report)
# ---------------------------------------------------------------------------

FEW_SHOTS_PRES_ACOLHEDOR = [
    "Olá! Aqui é da Lumina Estética. Como posso te ajudar hoje?",
    "Oi! 😊 Aqui é da Lumina. Em que posso ajudar?",
    "Olá, tudo bem? Aqui é da Lumina, como posso te ajudar?",
]
FEW_SHOTS_PRES_SEMIFORMAL = [
    "Olá, seja bem-vindo à Clínica Vita Premium. Como podemos te ajudar?",
    "Bom dia, aqui é da Vita Premium. Em que posso ser útil?",
    "Boa tarde, sou Helena da Vita Premium. Como posso atender você?",
]
FEW_SHOTS_SEM_APRESENTACAO_NEUTRO = [
    "Olá! Como posso te ajudar?",
    "Oi! Tudo certo?",
    "Olá, em que posso ajudar?",
]
FEW_SHOTS_SEM_APRESENTACAO_PERIODO = [
    "Bom dia! Como posso te ajudar?",
    "Boa tarde, em que posso ser útil?",
    "Olá! No que posso ajudar?",
]


# ---------------------------------------------------------------------------
# Render consolidated report.
# ---------------------------------------------------------------------------

def render_consolidated(version: str, runs: List[Tuple[str, int, Dict[str, Any]]]) -> str:
    """Render the consolidated markdown for a given version.

    runs: list of (temp_str, round_num, parsed_dict)
    """
    if not runs:
        return ""

    # Aggregate: case_id -> { (temp, round) -> data }
    cases_map: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = defaultdict(dict)
    case_meta: Dict[str, Dict[str, Any]] = {}
    for temp, rnd, data in runs:
        for cid, cdata in data["cases"].items():
            cases_map[cid][(temp, rnd)] = cdata
            # Capture metadata from first occurrence
            if cid not in case_meta:
                case_meta[cid] = {
                    "label": cdata["label"],
                    "quadrant": cdata["quadrant"],
                    "expected": cdata["expected"],
                    "user_prompt": cdata["user_prompt"],
                }

    # KEEP ONLY round 1 of each temperature for the consolidated report.
    # Reasoning: temperature is the variable under test; running the same
    # temperature multiple times only adds variance noise, which is not the
    # comparison this report is meant to support.
    cases_map_r1: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for cid, by_key in cases_map.items():
        for (temp, rnd), cdata in by_key.items():
            if rnd == 1:
                cases_map_r1[cid][temp] = cdata
    cases_map = {cid: data for cid, data in cases_map_r1.items() if data}

    # Sort case ids by quadrant then numeric id
    ordered_cases = sorted(cases_map.keys(), key=lambda x: (case_meta[x]["quadrant"], x))

    # Group cases by quadrant
    by_quadrant: Dict[str, List[str]] = defaultdict(list)
    for cid in ordered_cases:
        by_quadrant[case_meta[cid]["quadrant"]].append(cid)

    # Pick a representative run for system_prompt / model config (first one)
    rep = runs[0][2]

    # Temperatures actually present (after filtering to round 1)
    temps_used = sorted(
        {t for case_data in cases_map.values() for t in case_data.keys()},
        key=lambda x: float(x),
    )

    # Aggregate auto-score per temperature (sum across cases)
    score_by_temp: Dict[str, Dict[str, int]] = {}
    for temp in temps_used:
        y = sum(1 for c in cases_map.values() if c.get(temp, {}).get("verdict") == "YES")
        n = sum(1 for c in cases_map.values() if c.get(temp, {}).get("verdict") == "NO")
        score_by_temp[temp] = {"yes": y, "no": n}

    # Build markdown
    lines: List[str] = []
    lines.append(f"# Avaliação APRESENTAÇÃO — {version} (consolidado)")
    lines.append("")
    lines.append(f"- **Versão do prompt:** {version}")
    lines.append(f"- **Modelo:** `{rep['model']}`")
    lines.append(f"- **max_tokens:** `{rep['max_tokens']}`")
    lines.append(f"- **Temperaturas testadas:** {', '.join(temps_used)}")
    lines.append("- **Rounds por temperatura:** 1")
    lines.append(
        f"- **Total de runs:** {len(cases_map)} casos × {len(temps_used)} temperatura(s) = {len(cases_map) * len(temps_used)}"
    )
    lines.append("")

    # Scoreboard table — auto-YES por temp
    lines.append("## Scoreboard (auto-heurístico)")
    lines.append("")
    lines.append("| temp | auto-score |")
    lines.append("|---|---|")
    for temp in temps_used:
        s = score_by_temp[temp]
        lines.append(f"| **{temp}** | {s['yes']}/{s['yes'] + s['no']} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-case sections, grouped by quadrant
    for quadrant in by_quadrant:
        if quadrant:
            lines.append(f"## {quadrant}")
            lines.append("")
        for cid in by_quadrant[quadrant]:
            meta = case_meta[cid]
            lines.append(f"### {cid} — {meta['label']}")
            lines.append("")
            lines.append(f"**Expectativa:** `{meta['expected']}`")
            lines.append("")
            lines.append("**Input (user prompt enviado ao LLM):**")
            lines.append("")
            lines.append("```")
            lines.append(meta["user_prompt"])
            lines.append("```")
            lines.append("")

            # Outputs table across temps (one row per temperature)
            lines.append("**Outputs por temperatura:**")
            lines.append("")
            lines.append("| temp | latência | output | auto |")
            lines.append("|---|---|---|---|")
            for temp in temps_used:
                cdata = cases_map[cid].get(temp)
                if not cdata:
                    continue
                lat = f"{cdata['latency_ms']}ms" if cdata["latency_ms"] is not None else "—"
                out = cdata["output"].replace("|", "\\|")
                lines.append(f"| {temp} | {lat} | {out} | {cdata['verdict']} |")
            lines.append("")

            # Reasonings — one per temperature
            lines.append("**Reasonings:**")
            lines.append("")
            for temp in temps_used:
                cdata = cases_map[cid].get(temp)
                if not cdata or not cdata["reasoning"]:
                    continue
                lines.append(f"- **temp {temp}:** {cdata['reasoning']}")
            lines.append("")

            lines.append("**Veredito humano:**")
            lines.append("")
            for temp in temps_used:
                cdata = cases_map[cid].get(temp)
                if not cdata:
                    continue
                lines.append(f"- [ ] temp {temp} — YES  /  NO: ___")
            lines.append("")
            lines.append("---")
            lines.append("")

    # Configuração no final (uma única vez)
    lines.append("## Configuração do agente")
    lines.append("")
    lines.append(f"- **Modelo:** `{rep['model']}`")
    lines.append(f"- **max_tokens:** `{rep['max_tokens']}`")
    lines.append("- **Fallback técnico:** `'Olá! Tudo bem?'`")
    lines.append("- **Cache:** desabilitado (eval)")
    lines.append("")
    lines.append("### Few-shots utilizados")
    lines.append("")
    for name, fs in [
        ("FEW_SHOTS_PRES_ACOLHEDOR (Lumina, com apresentação)", FEW_SHOTS_PRES_ACOLHEDOR),
        ("FEW_SHOTS_PRES_SEMIFORMAL (Vita Premium, com apresentação)", FEW_SHOTS_PRES_SEMIFORMAL),
        ("FEW_SHOTS_SEM_APRESENTACAO_NEUTRO", FEW_SHOTS_SEM_APRESENTACAO_NEUTRO),
        ("FEW_SHOTS_SEM_APRESENTACAO_PERIODO", FEW_SHOTS_SEM_APRESENTACAO_PERIODO),
    ]:
        lines.append(f"**{name}**:")
        lines.append("```")
        for ex in fs:
            lines.append(f"- {ex}")
        lines.append("```")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Anexo A — SYSTEM_PROMPT")
    lines.append("")
    lines.append("```")
    lines.append(rep["system_prompt"])
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def consolidate_version(version: str) -> Optional[str]:
    """Read all per-round reports for a version and write a consolidated file."""
    # Pattern: apresentacao v23 temp0.X - round N (YYYY-MM-DD).md
    #          apresentacao v25 - round N (YYYY-MM-DD).md   (no temp suffix)
    pattern_temp = os.path.join(FOLDER, f"apresentacao {version} temp* - round * (*).md")
    pattern_notemp = os.path.join(FOLDER, f"apresentacao {version} - round * (*).md")

    files: List[Tuple[str, int, str]] = []  # (temp, round, path)
    has_temp_variants = bool(glob.glob(pattern_temp))

    for path in glob.glob(pattern_temp):
        m = re.search(rf"{re.escape(version)} temp(\d+\.\d+) - round (\d+)", path)
        if m:
            files.append((m.group(1), int(m.group(2)), path))

    # Only include no-temp-suffix files when there are NO temp variants for
    # this version. Otherwise the no-suffix file is a stray from an older
    # run and would duplicate a temp 0.3 entry.
    if not has_temp_variants:
        for path in glob.glob(pattern_notemp):
            m = re.search(rf"{re.escape(version)} - round (\d+)", path)
            if m:
                files.append(("0.3", int(m.group(1)), path))

    if not files:
        print(f"[warn] no files found for version {version}")
        return None

    # Sort by (temp, round)
    files.sort(key=lambda x: (float(x[0]), x[1]))

    runs = []
    for temp, rnd, path in files:
        print(f"  parsing {os.path.basename(path)}")
        data = parse_round_file(path)
        runs.append((temp, rnd, data))

    md = render_consolidated(version, runs)
    date_tag = "2026-05-18"
    out_path = os.path.join(FOLDER, f"apresentacao {version} consolidado ({date_tag}).md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"[ok] wrote {out_path}")
    return out_path


if __name__ == "__main__":
    for version in ("v23", "v25"):
        print(f"\n=== Consolidating {version} ===")
        consolidate_version(version)
