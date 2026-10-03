# Author PhysX torsional patch friction onto the K-Bot foot collision hulls and
# save as a SEPARATE usd (robot_torsional.usd) — the training USD is untouched.
# Part of the 2026-07-15 drift investigation: sim statue drifts 13.3 cm/30s vs
# 6.8 on the rig at the same effective mu; hypothesis = foot convex hull grips
# through ~1 merged friction anchor -> free yaw-pivot -> ratchet drift. The rig
# foot has MuJoCo spin friction 0.02 (m units ~ moment arm): torsionalPatchRadius
# 0.02 is the physically-mapped equivalent (torque ~ mu * N * radius).
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--radius", type=float, default=0.02)
parser.add_argument("--min_radius", type=float, default=0.005)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import shutil

from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402

from isaaclab_assets.robots.kscale_legs import KBOT_LEGS_USD  # noqa: E402

out_path = os.path.join(os.path.dirname(KBOT_LEGS_USD), "robot_torsional.usd")
shutil.copyfile(KBOT_LEGS_USD, out_path)
stage = Usd.Stage.Open(out_path)
# foot collision prims are INSTANCE PROXIES (uneditable) — de-instance the foot
# link subtrees in this copy so the collision prims become authorable
for prim in stage.Traverse():
    if "FOOT" in str(prim.GetPath()).upper() and prim.IsInstance():
        prim.SetInstanceable(False)
        print(f"[author] de-instanced {prim.GetPath()}")
n = 0
for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
    if "FOOT" not in str(prim.GetPath()).upper():
        continue
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    if prim.IsInstanceProxy():
        print(f"[author] SKIP instance proxy (cannot author): {prim.GetPath()}")
        continue
    api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    api.CreateTorsionalPatchRadiusAttr().Set(args.radius)
    api.CreateMinTorsionalPatchRadiusAttr().Set(args.min_radius)
    print(f"[author] set torsionalPatchRadius={args.radius} min={args.min_radius} on {prim.GetPath()}")
    n += 1
stage.GetRootLayer().Save()
print(f"[author] authored {n} prims -> {out_path}")
with open("eval_watch/author_torsional_result.txt", "w") as f:
    f.write(f"{n} prims -> {out_path}\n")
simulation_app.close()
