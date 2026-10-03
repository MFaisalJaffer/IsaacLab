# Inspect the K-Bot legs USD: foot collision geometry + PhysX torsional/patch
# attrs. Part of the 2026-07-15 drift investigation (sim drifts 2x the rig at
# the same effective mu; question is whether torsionalPatchRadius or geometry
# is the missing grip). Boots Kit headless because pxr requires it.
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics  # noqa: E402

from isaaclab_assets.robots.kscale_legs import KBOT_LEGS_USD  # noqa: E402

print(f"[inspect] usd: {KBOT_LEGS_USD}")
stage = Usd.Stage.Open(KBOT_LEGS_USD)
lines = []
for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
    path = str(prim.GetPath())
    if "FOOT" not in path.upper():
        continue
    has_col = prim.HasAPI(UsdPhysics.CollisionAPI)
    info = {"type": str(prim.GetTypeName()), "collisionAPI": has_col,
            "schemas": [str(s) for s in prim.GetAppliedSchemas()]}
    for a in (
        "physics:approximation",
        "physxCollision:torsionalPatchRadius",
        "physxCollision:minTorsionalPatchRadius",
        "physxCollision:contactOffset",
        "physxCollision:restOffset",
    ):
        at = prim.GetAttribute(a)
        if at:
            info[a.split(":")[-1]] = (at.Get(), "authored" if at.HasAuthoredValue() else "default")
    # geometry extent for size context
    ext = prim.GetAttribute("extent")
    if ext and ext.Get():
        info["extent"] = ext.Get()
    lines.append(f"{path}\n    {info}")
out = "\n".join(lines) if lines else "(no FOOT collision prims found)"
print(f"[inspect]\n{out}")
with open("eval_watch/foot_collision_report.txt", "w") as f:
    f.write(out + "\n")
simulation_app.close()
