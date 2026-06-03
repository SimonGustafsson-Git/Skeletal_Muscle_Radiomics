# Segment one image with MuscleMap


# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------

import subprocess
from pathlib import Path


# ---------------------------------------------------------------------
# INPUTS
# ---------------------------------------------------------------------
MUSCLEMAP_DIR = Path("")
MM_SEGMENT = "mm_segment"

IMAGE_PATH = Path("")
OUTPUT_DIR = Path("")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Missing image: {IMAGE_PATH}")

    cmd = [MM_SEGMENT, "-i", str(IMAGE_PATH), "-o", str(OUTPUT_DIR)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=MUSCLEMAP_DIR, check=True)


if __name__ == "__main__":
    main()
