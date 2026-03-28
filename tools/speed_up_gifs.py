#!/usr/bin/env python3
"""
Speed up GIF animations in docs slightly, without touching site text/CSS transitions.

What this script does
---------------------
- Finds source GIFs under docs/_static and docs/archive/scripts/_static
- Reads per-frame durations (ms) and reduces them by a uniform factor (default: 1.25x faster)
- Writes back the GIFs in-place with updated durations, preserving frame order and loop count

Notes
-----
- We intentionally do not modify any CSS/JS; only the embedded GIF file timings change.
- Pillow (PIL) reports frame durations in milliseconds; we scale by 1/speed and round.
- We set a small floor on durations to avoid edge cases where browsers clamp very small delays.

Usage
-----
python tools/speed_up_gifs.py                # default 1.25x faster on default dirs
python tools/speed_up_gifs.py --factor 1.2   # custom factor
python tools/speed_up_gifs.py docs/_static   # custom path(s)

Implementation choices
----------------------
- Per-frame durations are preserved proportionally, so pauses remain but are slightly shorter.
- We save with save_all=True and preserve loop=0 (infinite) if present; disposal is left to Pillow.
- We avoid optimizing palette to minimize visual changes; optimization can change appearance.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image, ImageSequence


def iter_gifs(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_dir():
            yield from (x for x in p.rglob("*.gif") if "_build" not in x.parts)
        elif p.is_file() and p.suffix.lower() == ".gif":
            if "_build" not in p.parts:
                yield p


def read_frames_and_durations(img_path: Path) -> Tuple[List[Image.Image], List[int], dict]:
    im = Image.open(img_path)
    frames: List[Image.Image] = []
    durations_ms: List[int] = []
    # Preserve basic info if present
    base_info = {
        key: im.info.get(key)
        for key in ("loop", "background", "transparency", "version")
        if key in im.info
    }

    for i, frame in enumerate(ImageSequence.Iterator(im)):
        # PIL updates im.info on seek; frame.info may not carry duration reliably.
        # Use the container's info after seeking to each frame.
        try:
            im.seek(i)
        except Exception:
            pass
        duration = int(im.info.get("duration", 100))  # ms
        # Copy the visual content of each frame so saving does not depend on lazy seeks.
        frames.append(frame.copy())
        durations_ms.append(duration)

    return frames, durations_ms, base_info


def scale_durations(durations_ms: List[int], factor: float, min_ms: int = 40) -> List[int]:
    scaled: List[int] = []
    for d in durations_ms:
        # Speed up by reducing duration per frame: new = old / factor
        new_d = max(min_ms, int(round(d / factor)))
        # GIF encodes durations in 10ms units; round to nearest 10ms to avoid bloat.
        new_d = int(round(new_d / 10.0)) * 10
        scaled.append(max(10, new_d))
    return scaled


def balance_durations(
    durations_ms: List[int],
    compress: float = 0.5,
    min_ms: int = 120,
    max_ms: int = 1500,
) -> List[int]:
    """
    Rebalance frame durations to reduce extremes while preserving sequence rhythm.

    new = median + (old - median) * compress
    then clamp to [min_ms, max_ms] and round to nearest 10ms.

    - compress in (0,1): smaller → stronger pull toward median
    - min_ms: ensure very fast text frames linger a bit more
    - max_ms: cap long holds so illustrations don't stall
    """
    if not durations_ms:
        return durations_ms
    med = statistics.median(durations_ms)
    out: List[int] = []
    for d in durations_ms:
        nd = med + (d - med) * compress
        nd = max(min_ms, min(max_ms, int(round(nd))))
        nd = int(round(nd / 10.0)) * 10
        out.append(max(10, nd))
    return out


def summarize(label: str, durations: List[int]) -> str:
    if not durations:
        return f"{label}: 0 frames"
    return (
        f"{label}: {len(durations)} frames | "
        f"mean={statistics.mean(durations):.1f}ms, "
        f"median={statistics.median(durations):.1f}ms, "
        f"min={min(durations)}ms, max={max(durations)}ms"
    )


def process_gif(path: Path, factor: float, dry_run: bool = False) -> None:
    frames, orig_durations, base_info = read_frames_and_durations(path)

    if not frames:
        print(f"SKIP (no frames): {path}")
        return

    new_durations = scale_durations(orig_durations, factor)

    print(f"\n{path}")
    print("  " + summarize("original", orig_durations))
    print("  " + summarize("new     ", new_durations))

    if dry_run:
        return

    # Save back in-place with updated durations.
    # Keep palette/mode of first frame as-is to minimize visual diffs.
    first = frames[0]
    rest = frames[1:]
    save_kwargs = dict(
        save_all=True,
        append_images=rest,
        duration=new_durations,
        loop=base_info.get("loop", 0) if base_info.get("loop") is not None else 0,
        disposal=2,  # restore to background between frames helps avoid trails
        optimize=False,
    )
    # Attempt to carry transparency if present
    if base_info.get("transparency") is not None:
        save_kwargs["transparency"] = base_info["transparency"]

    tmp_path = path.with_suffix(".tmp.gif")
    first.save(tmp_path, format="GIF", **save_kwargs)
    tmp_path.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Adjust GIF frame durations (scale or balance).")
    parser.add_argument("paths", nargs="*", type=Path, help="Directories/files to process")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--factor", type=float, help="Scale: speed-up factor (>1 = faster)")
    mode.add_argument(
        "--balance", action="store_true", help="Balance durations by compressing extremes"
    )
    parser.add_argument("--compress", type=float, default=0.5, help="Balance: pull toward median (0–1)")
    parser.add_argument("--min-ms", type=int, default=120, help="Balance: minimum per-frame duration")
    parser.add_argument("--max-ms", type=int, default=1500, help="Balance: maximum per-frame duration")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing GIFs")
    args = parser.parse_args()

    default_roots = [Path("docs/_static"), Path("docs/archive/scripts/_static")]
    targets = args.paths if args.paths else default_roots

    gifs = sorted(set(iter_gifs(targets)))
    if not gifs:
        print("No GIFs found in:")
        for t in targets:
            print(f" - {t}")
        return

    # Default to a gentle speed-up if no mode specified
    if not args.balance and not args.factor:
        args.factor = 1.25

    if args.balance:
        print(
            f"Processing {len(gifs)} GIF(s) with balance: compress={args.compress}, "
            f"min_ms={args.min_ms}, max_ms={args.max_ms}"
        )
        for g in gifs:
            frames, orig_durations, base_info = read_frames_and_durations(g)
            new_durations = balance_durations(
                orig_durations,
                compress=args.compress,
                min_ms=args.min_ms,
                max_ms=args.max_ms,
            )
            print(f"\n{g}")
            print("  " + summarize("original", orig_durations))
            print("  " + summarize("balanced", new_durations))
            if args.dry_run:
                continue
            # Save via process_gif path by temporarily monkey-patching scale behavior
            # but here we re-use saving logic directly for clarity
            if not frames:
                continue
            first, rest = frames[0], frames[1:]
            save_kwargs = dict(
                save_all=True,
                append_images=rest,
                duration=new_durations,
                loop=base_info.get("loop", 0) if base_info.get("loop") is not None else 0,
                disposal=2,
                optimize=False,
            )
            if base_info.get("transparency") is not None:
                save_kwargs["transparency"] = base_info["transparency"]
            tmp_path = g.with_suffix(".tmp.gif")
            first.save(tmp_path, format="GIF", **save_kwargs)
            tmp_path.replace(g)
    else:
        print(
            f"Processing {len(gifs)} GIF(s) with factor={args.factor} (1.0=unchanged, 1.25=25% faster)"
        )
        for g in gifs:
            process_gif(g, factor=args.factor, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
