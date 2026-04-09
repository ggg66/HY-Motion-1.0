"""
Sample evaluation prompts for FlowSteer-Motion Phase 1 evaluation.

Two modes:
  1. HumanML3D mode (--humanml3d_path): stratified sample from the official test split.
     Requires the HumanML3D annotation directory (texts/*.txt) and test split list.
  2. Curated mode (default): uses the built-in set of 100 prompts covering
     walk / run / dance / sport / interaction categories.

Output: JSON file consumable by eval/run_eval.py

Usage:
    # Curated set (no data needed):
    python scripts/sample_prompts.py --output eval/prompts/full_eval.json

    # From HumanML3D test split (on server):
    python scripts/sample_prompts.py \
        --humanml3d_path /data/HumanML3D \
        --n_per_category 15 \
        --output eval/prompts/humanml3d_eval.json

    # Terminal subset only:
    python scripts/sample_prompts.py \
        --output eval/prompts/terminal_eval.json \
        --filter terminal
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Keyword-based constraint auto-assignment
# ---------------------------------------------------------------------------

_LOCOMOTION_KW = [
    "walk", "run", "jog", "march", "stride", "limp", "stroll", "trot",
    "sprint", "shuffle", "crawl", "sneak", "creep",
]
_DANCE_KW = ["dance", "bachata", "jazz", "salsa", "ballet", "twist", "groove"]
_STATIC_KW = ["sit", "lie", "kneel", "stand still", "squat"]
_TERMINAL_KW = [
    "to a destination", "to the other side", "towards", "goes to",
    "walks to", "reaches", "arrives",
]


def _assign_constraints(text: str) -> List[str]:
    t = text.lower()
    has_terminal = any(kw in t for kw in _TERMINAL_KW)
    if has_terminal:
        return ["foot_contact", "terminal"]
    return ["foot_contact"]


def _frames_to_duration(frames: int, fps: float = 30.0) -> float:
    dur = round(frames / fps, 1)
    return max(2.0, min(dur, 10.0))


# ---------------------------------------------------------------------------
# Built-in curated prompt set  (100 prompts, 3 categories)
# ---------------------------------------------------------------------------
#
# Format: (text, duration_seconds, constraints, terminal_xz or None)
# Constraints: "fc" = foot_contact, "fc+t" = foot_contact + terminal

_CURATED: List[tuple] = [
    # ── Locomotion: foot_contact only (60) ───────────────────────────────
    ("a person walks forward and stops", 3.0, "fc", None),
    ("a person walks forward.", 4.0, "fc", None),
    ("a person walks forward, moving arms and legs while looking left and right.", 6.0, "fc", None),
    ("a person walks with a limp, their left leg is injured.", 6.4, "fc", None),
    ("a person walks on a tightrope.", 6.0, "fc", None),
    ("a person walks like a zombie, dragging their feet forward.", 4.0, "fc", None),
    ("a person walks forward, holding a tray at shoulder height with one hand.", 3.1, "fc", None),
    ("a person walks in a catwalk style, swinging their left arm while placing their right hand on their hip.", 6.0, "fc", None),
    ("a person walks unsteadily, then slowly sits down.", 5.0, "fc", None),
    ("a person turns backward 180 degrees, then walks forward.", 4.0, "fc", None),
    ("a person runs forward.", 2.0, "fc", None),
    ("a person runs and then slows down", 4.0, "fc", None),
    ("a person runs forward, then kicks a soccer ball.", 2.0, "fc", None),
    ("a person sprints forward.", 3.0, "fc", None),
    ("a person jogs in a circle.", 5.0, "fc", None),
    ("a person jogs forward at a moderate pace.", 4.0, "fc", None),
    ("a person runs in a zigzag pattern.", 4.0, "fc", None),
    ("a person marches in place, swinging their arms forward and backward.", 7.0, "fc", None),
    ("a person walks backward slowly.", 4.0, "fc", None),
    ("a person walks sideways to the right.", 3.0, "fc", None),
    ("a person shuffles forward with small steps.", 4.0, "fc", None),
    ("a person walks quickly, then abruptly stops.", 3.0, "fc", None),
    ("a person struts confidently with wide steps.", 4.0, "fc", None),
    ("a person walks while looking around cautiously.", 5.0, "fc", None),
    ("a person jogs on the spot.", 4.0, "fc", None),
    ("the man walked forward, spun right on one foot and walked back to his original position.", 3.1, "fc", None),
    ("this person stumbles left and right while moving forward.", 4.4, "fc", None),
    ("a person lifts a long gun, then walks forward slowly.", 3.0, "fc", None),
    ("a person walks forward then turns left.", 4.0, "fc", None),
    ("a person walks forward then turns right.", 4.0, "fc", None),
    # ── Dance and rhythmic (15) ──────────────────────────────────────────
    ("a person dances.", 4.0, "fc", None),
    ("a person dances bachata, executing rhythmic hip movements and footwork.", 8.0, "fc", None),
    ("a person dances jazz, jumping rhythmically.", 8.0, "fc", None),
    ("a person performs a waltz step.", 5.0, "fc", None),
    ("a person does a hip-hop dance.", 5.0, "fc", None),
    ("a person shuffles in a dance move.", 4.0, "fc", None),
    ("a person performs a line dance with footwork.", 6.0, "fc", None),
    ("a person does a salsa step.", 5.0, "fc", None),
    ("a person performs a contemporary dance routine.", 6.0, "fc", None),
    ("a person dances in place with rhythmic foot tapping.", 5.0, "fc", None),
    ("a person performs a breakdance toprock.", 4.0, "fc", None),
    ("a person does a two-step dance.", 4.0, "fc", None),
    ("a person dances flamenco, stamping their feet rhythmically.", 6.0, "fc", None),
    ("a person performs a Scottish reel step.", 5.0, "fc", None),
    ("a person does a moonwalk step.", 4.0, "fc", None),
    # ── Sports and dynamic actions (15) ──────────────────────────────────
    ("a person jumps upward with both legs twice.", 3.0, "fc", None),
    ("a person jumps on their right leg.", 3.0, "fc", None),
    ("a person jumps up.", 3.0, "fc", None),
    ("a person jumps forward lightly, taking two steps.", 2.3, "fc", None),
    ("a person performs a taekwondo kick, extending their leg forcefully.", 2.0, "fc", None),
    ("a person performs a side kick.", 2.0, "fc", None),
    ("a person does a front kick followed by a spin.", 3.0, "fc", None),
    ("a person climbs upward, moving up a slope.", 2.0, "fc", None),
    ("a person practices tai chi, performing slow, controlled movements.", 9.0, "fc", None),
    ("a person skips forward.", 3.0, "fc", None),
    ("a person hops on one foot.", 3.0, "fc", None),
    ("a person does a cartwheel.", 3.0, "fc", None),
    ("a person performs a standing long jump.", 2.0, "fc", None),
    ("a person runs and jumps over an obstacle.", 3.0, "fc", None),
    ("a person dodges to the side quickly.", 2.0, "fc", None),
    # ── Terminal constraint: walks to a destination (20) ─────────────────
    ("a person walks to a destination.", 4.0, "fc+t", [2.5, 0.0]),
    ("a person walks to the other side of the room.", 4.0, "fc+t", [3.0, 0.0]),
    ("a person walks towards a point ahead of them.", 3.0, "fc+t", [2.0, 0.0]),
    ("a person walks diagonally to a target position.", 4.0, "fc+t", [2.0, 2.0]),
    ("a person walks to a spot on their left.", 3.0, "fc+t", [0.0, 2.0]),
    ("a person walks to a spot on their right.", 3.0, "fc+t", [0.0, -2.0]),
    ("a person runs to a destination and stops.", 3.0, "fc+t", [3.5, 0.0]),
    ("a person jogs towards a goal and decelerates to a stop.", 4.0, "fc+t", [4.0, 0.0]),
    ("a person walks briskly to a meeting point.", 3.0, "fc+t", [2.5, 1.0]),
    ("a person walks casually to a chair and sits down.", 4.0, "fc+t", [2.0, 0.5]),
    ("a person moves towards the camera.", 3.0, "fc+t", [1.5, 0.0]),
    ("a person walks to the far end of the stage.", 5.0, "fc+t", [5.0, 0.0]),
    ("a person walks to a nearby object and bends down.", 3.0, "fc+t", [1.5, 0.5]),
    ("a person approaches and stops at a specific location.", 4.0, "fc+t", [2.0, -1.0]),
    ("a person walks to a corner of the room.", 4.0, "fc+t", [2.5, -2.5]),
    ("a person strides purposefully to a destination.", 4.0, "fc+t", [3.0, 1.0]),
    ("a person hurries to a specific point.", 3.0, "fc+t", [3.0, 0.0]),
    ("a person saunters to a position and turns around.", 5.0, "fc+t", [2.0, 0.0]),
    ("a person runs across the field to the opposite end.", 4.0, "fc+t", [6.0, 0.0]),
    ("a person walks backwards to a target position behind them.", 3.0, "fc+t", [-1.5, 0.0]),
]


def _curated_to_prompt_cfg(entry: tuple) -> Dict:
    text, duration, ctype, terminal_xz = entry
    if ctype == "fc+t":
        cfg = {
            "prompt": text,
            "duration": duration,
            "constraint": ["foot_contact", "terminal"],
            "terminal_xz": terminal_xz,
        }
    else:
        cfg = {
            "prompt": text,
            "duration": duration,
            "constraint": ["foot_contact"],
        }
    return cfg


# ---------------------------------------------------------------------------
# HumanML3D loader
# ---------------------------------------------------------------------------

def _load_humanml3d(
    data_path: str,
    n_per_category: int = 15,
    seed: int = 42,
) -> List[Dict]:
    """
    Load prompts from HumanML3D annotation directory.

    Expected layout:
        <data_path>/
            texts/          # one .txt per motion, each line: "text#tokens"
            test.txt        # newline-separated motion IDs for test split

    Categories (keyword-based split):
        walk, run, dance, sport, interaction, other
    """
    test_split = os.path.join(data_path, "test.txt")
    texts_dir  = os.path.join(data_path, "texts")

    if not os.path.exists(test_split):
        # Try alternate paths
        for candidate in ["test_humanml.txt", "split/test.txt"]:
            alt = os.path.join(data_path, candidate)
            if os.path.exists(alt):
                test_split = alt
                break
        else:
            raise FileNotFoundError(f"test.txt not found under {data_path}")

    with open(test_split) as f:
        ids = [l.strip() for l in f if l.strip()]

    print(f"HumanML3D test split: {len(ids)} motions")

    CATEGORIES = {
        "walk":        ["walk", "stroll", "limp", "amble", "wander"],
        "run":         ["run", "jog", "sprint", "dash", "gallop"],
        "dance":       ["dance", "bachata", "jazz", "salsa", "ballet"],
        "sport":       ["jump", "kick", "throw", "catch", "climb", "swim", "skate"],
        "interaction": ["pick", "carry", "push", "pull", "open", "close"],
        "other":       [],
    }

    buckets: Dict[str, List[tuple]] = {k: [] for k in CATEGORIES}

    for mid in ids:
        txt_path = os.path.join(texts_dir, mid + ".txt")
        if not os.path.exists(txt_path):
            continue
        with open(txt_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            continue
        # Use first annotation
        raw = lines[0].split("#")[0].strip().lower()
        full = lines[0].split("#")[0].strip()

        assigned = "other"
        for cat, kws in CATEGORIES.items():
            if cat == "other":
                continue
            if any(kw in raw for kw in kws):
                assigned = cat
                break

        # Estimate duration: look for frame count in filename or use default
        # HumanML3D uses 20fps internally, motions are 2-10s
        duration = 4.0

        buckets[assigned].append((full, duration, mid))

    rng = random.Random(seed)
    prompts = []
    for cat, items in buckets.items():
        rng.shuffle(items)
        selected = items[:n_per_category]
        for text, duration, mid in selected:
            constraints = _assign_constraints(text)
            cfg: Dict = {
                "prompt": text,
                "duration": duration,
                "constraint": constraints,
                "source_id": mid,
            }
            if "terminal" in constraints:
                cfg["terminal_xz"] = [2.5, 0.0]
            prompts.append(cfg)
        print(f"  {cat:12s}: {len(selected):3d} prompts")

    return prompts


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _filter_prompts(prompts: List[Dict], filter_type: Optional[str]) -> List[Dict]:
    if filter_type is None:
        return prompts
    if filter_type == "terminal":
        return [p for p in prompts if "terminal" in p["constraint"]]
    if filter_type == "foot_contact_only":
        return [p for p in prompts if p["constraint"] == ["foot_contact"]]
    return prompts


# ---------------------------------------------------------------------------
# Statistics printer
# ---------------------------------------------------------------------------

def _print_stats(prompts: List[Dict]) -> None:
    from collections import Counter
    cats = Counter()
    for p in prompts:
        key = "+".join(p["constraint"])
        cats[key] += 1
    total_dur = sum(p["duration"] for p in prompts)
    print(f"\nTotal prompts : {len(prompts)}")
    print(f"Total duration: {total_dur:.0f}s ({total_dur/60:.1f} min)")
    print("Constraint breakdown:")
    for k, v in cats.most_common():
        print(f"  {k:30s}: {v}")
    durations = [p["duration"] for p in prompts]
    print(f"Duration range: {min(durations):.1f}s – {max(durations):.1f}s  "
          f"(mean {sum(durations)/len(durations):.1f}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sample FlowSteer-Motion eval prompts")
    parser.add_argument("--output", default="eval/prompts/full_eval.json",
                        help="Output JSON path")
    parser.add_argument("--humanml3d_path", default=None,
                        help="Path to HumanML3D data dir (texts/ + test.txt). "
                             "If omitted, uses built-in curated set.")
    parser.add_argument("--n_per_category", type=int, default=15,
                        help="[HumanML3D mode] prompts per action category")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter", default=None,
                        choices=["terminal", "foot_contact_only"],
                        help="Output only a subset by constraint type")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle output order")
    args = parser.parse_args()

    if args.humanml3d_path:
        print(f"Loading from HumanML3D: {args.humanml3d_path}")
        prompts = _load_humanml3d(args.humanml3d_path, args.n_per_category, args.seed)
    else:
        print("Using built-in curated prompt set")
        prompts = [_curated_to_prompt_cfg(e) for e in _CURATED]

    prompts = _filter_prompts(prompts, args.filter)

    if args.shuffle:
        random.Random(args.seed).shuffle(prompts)

    # Add sequential index
    for i, p in enumerate(prompts):
        p["idx"] = i + 1

    _print_stats(prompts)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(prompts)} prompts → {args.output}")


if __name__ == "__main__":
    main()
