# Is the K-Bot foot collision sole FLAT or CURVED? (2026-07-16, user challenge
# to the rocking hypothesis.) Reads the collision MESH vertices from the legs
# USD and profiles the bottom surface: if the lowest vertices are coplanar
# within ~1 mm and span the footprint, the sole is flat (rocking would be
# edge-tilting, not curvature); if min-z varies along the foot length, it's
# curved and the hull rocks by construction.
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom  # noqa: E402

from isaaclab_assets.robots.kscale_legs import KBOT_LEGS_USD  # noqa: E402

stage = Usd.Stage.Open(KBOT_LEGS_USD)
lines = []
for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
    p = str(prim.GetPath())
    if "FOOT" not in p.upper() or "/collisions/" not in p or prim.GetTypeName() != "Mesh":
        continue
    pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
    if not pts:
        continue
    # foot z extents differ in sign L/R (mirrored) — find the sole axis by the
    # larger |extent| side on z
    # z is the LATERAL axis (L/R feet mirror in z); vertical is y. Profile every
    # face anyway — the sole is whichever face is flat AND spans the footprint.
    import collections
    lines.append(p)
    for ax, name in ((0, "x"), (1, "y"), (2, "z")):
        vals = sorted(v[ax] for v in pts)
        for face, ref in ((f"{name}-min", vals[0]), (f"{name}-max", vals[-1])):
            near = [v for v in pts if abs(v[ax] - ref) < 0.001]
            xs = [v[0] for v in near] or [0]
            bins = collections.defaultdict(lambda: 1e9)
            for v in pts:
                b = round(v[0] / 0.03) * 0.03  # 3 cm bins along foot length
                bins[b] = min(bins[b], abs(v[ax] - ref))
            prof = " ".join(f"{b:+.2f}:{d*1000:.0f}" for b, d in sorted(bins.items()))
            lines.append(
                f"  {face}={ref:+.4f} 1mm-verts={len(near)} xspan=[{min(xs):+.3f},{max(xs):+.3f}]"
                f" | clearance(mm) along x: {prof}"
            )
out = "\n".join(lines) if lines else "(no foot collision meshes found)"
print("[sole]\n" + out)
with open("eval_watch/sole_flatness_report.txt", "w") as f:
    f.write(out + "\n")
simulation_app.close()
