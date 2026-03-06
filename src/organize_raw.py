from pathlib import Path
import shutil

# -----------------------------
# PATHS
# -----------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"

print("RAW exists:", RAW.exists())
print("RAW path:", RAW)

if not RAW.exists():
    print("❌ RAW folder not found")
    exit()

# -----------------------------
# MOVE FILES UP
# -----------------------------

moved = 0
skipped = 0

for sub in RAW.iterdir():

    if not sub.is_dir():
        continue

    for f in sub.glob("*.*"):

        dest = RAW / f.name

        if dest.exists():
            print("⚠️ Already exists:", dest.name)
            skipped += 1
            continue

        shutil.move(str(f), str(dest))
        moved += 1

    # remove empty folder
    try:
        sub.rmdir()
        print("🗑 Removed folder:", sub.name)
    except:
        pass

print("\n✅ DONE")
print("Moved:", moved)
print("Skipped:", skipped)
