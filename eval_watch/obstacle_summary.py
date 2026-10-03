"""Score and one-line summary of an obstacle_eval.py result (stage 1).

  score  = mean share of robots that crossed, over every obstacle kind and height (0..1)
  climbs = tallest height that at least --min of the robots cross, with every lower height passing too

Prints "<score>|<text>". With a second file (the same checkpoint with the map zeroed) the text ends with
that score.

  python3 eval_watch/obstacle_summary.py eval_watch/<tag>.json [eval_watch/<tag>_nomap.json]
"""
import json
import sys


def score(path: str, min_ok: float = 0.8):
    r = json.load(open(path))
    cells, parts, climbs = [], [], {}
    for kind, rows in r["kinds"].items():
        if kind == "flat":
            parts.append(f"flat {rows[0]['crossed']:.2f}")
            continue
        cells += [x["crossed"] for x in rows]
        limit = 0.0
        for x in rows:
            if x["crossed"] < min_ok:
                break
            limit = x["height_cm"]
        climbs[kind] = limit
        parts.append(f"{kind} " + " ".join(f"{x['crossed']:.2f}" for x in rows) + f" -> climbs {limit:.0f} cm")
    return (sum(cells) / max(len(cells), 1)), "; ".join(parts), climbs


if __name__ == "__main__":
    s, text, _ = score(sys.argv[1])
    if len(sys.argv) > 2:
        try:
            s0, _, c0 = score(sys.argv[2])
            text += f"; with the map zeroed: score {s0:.3f}, climbs " + "/".join(f"{v:.0f}" for v in c0.values()) + " cm"
        except Exception:
            text += "; map-zeroed test failed"
    print(f"{s:.3f}|{text}")
