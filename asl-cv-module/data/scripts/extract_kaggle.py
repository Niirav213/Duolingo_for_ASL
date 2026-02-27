"""
data/scripts/extract_kaggle.py
-------------------------------
Batch processes the Kaggle ASL Alphabet dataset images through
MediaPipe and saves landmark feature vectors as .npy files.

This bridges the gap between raw Kaggle images and your training pipeline.

Full flow:
    Kaggle images → MediaPipe → .npy files → preprocess.py → train_static.py

Dataset expected structure:
    data/datasets/raw_images/
        A/
            A1.jpg
            A2.jpg
            ...
        B/
            B1.jpg
            ...

Output structure:
    data/datasets/raw/
        A/
            session1.npy   ← shape (N, 178)
        B/
            session1.npy
        ...

Usage:
    # Local
    python data/scripts/extract_kaggle.py

    # On Kaggle notebook — change INPUT_DIR to:
    # /kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train
"""

import cv2
import sys
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm

# So Python can find your modules when run from project root
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import mediapipe as mp
except ImportError:
    print("[ERROR] MediaPipe not installed. Run: pip install mediapipe")
    sys.exit(1)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

# Resolve project root relative to this script file
# This works no matter where you run the script from on Windows or Mac/Linux
SCRIPT_DIR   = Path(__file__).resolve().parent          # data/scripts/
DATA_DIR     = SCRIPT_DIR.parent                        # data/
PROJECT_ROOT = SCRIPT_DIR.parent.parent                 # asl-cv-module/

# Where your raw Kaggle images live
INPUT_DIR = DATA_DIR / "datasets" / "raw_images"

# Where extracted .npy files will be saved
OUTPUT_DIR = DATA_DIR / "datasets" / "raw"

# How many images to process per sign (None = all)
# Set to 200 for faster testing, None for full dataset
MAX_PER_SIGN = None

# Batch size — how many samples to save per .npy file
BATCH_SIZE = 1000


# ─────────────────────────────────────────────
# Feature Extraction (mirrors extractor.py)
# ─────────────────────────────────────────────

FINGER_ANGLE_TRIPLETS = [
    (1, 2, 3),   (2, 3, 4),       # Thumb
    (5, 6, 7),   (6, 7, 8),       # Index
    (9, 10, 11), (10, 11, 12),    # Middle
    (13, 14, 15),(14, 15, 16),    # Ring
    (17, 18, 19),(18, 19, 20),    # Pinky
    (5, 0, 9),   (9, 0, 13),      # Splay
    (13, 0, 17), (5, 0, 17),
    (0, 5, 9),
]

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_MCPS = [2, 5, 9, 13, 17]


def angle_at_joint(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def extract_features_from_image(image_path: str, hands) -> np.ndarray | None:
    """
    Run MediaPipe on a single image and return a 178-dim feature vector.
    Returns None if no hand is detected.

    Args:
        image_path: Path to the image file
        hands: MediaPipe Hands instance (reused for efficiency)

    Returns:
        np.ndarray of shape (178,) or None
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    # MediaPipe requires RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    # Use first detected hand
    lms = result.multi_hand_landmarks[0].landmark
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)  # (21, 3)

    # ── 1. Joint angles (15,) ──
    angles = np.array([
        angle_at_joint(pts[a], pts[b], pts[c])
        for a, b, c in FINGER_ANGLE_TRIPLETS
    ], dtype=np.float32)

    # ── 2. Normalized positions (63,) ──
    norm_pts = pts - pts[0]                              # wrist as origin
    scale = np.linalg.norm(norm_pts[9]) + 1e-6          # normalize by hand size
    norm_pts = (norm_pts / scale).flatten().astype(np.float32)

    # ── 3. Palm orientation (3,) ──
    v1 = pts[5] - pts[0]
    v2 = pts[17] - pts[0]
    normal = np.cross(v1, v2)
    orientation = (normal / (np.linalg.norm(normal) + 1e-6)).astype(np.float32)

    # ── 4. Finger extensions (5,) ──
    wrist = pts[0]
    extensions = np.array([
        float(np.clip(
            np.linalg.norm(pts[tip] - wrist) / (np.linalg.norm(pts[mcp] - wrist) + 1e-6) - 1.0,
            0.0, 1.0
        ))
        for tip, mcp in zip(FINGER_TIPS, FINGER_MCPS)
    ], dtype=np.float32)

    # ── Assemble 178-dim vector ──
    # Left hand features are zero-padded (single hand dataset)
    # Layout matches extractor.py exactly:
    # [0:15]    right angles
    # [15:30]   left angles (zeros)
    # [30:93]   right normalized
    # [93:156]  left normalized (zeros)
    # [156:159] right orientation
    # [159:162] left orientation (zeros)
    # [162:165] right position (zeros — no pose data)
    # [165:168] left position (zeros)
    # [168:173] right extensions
    # [173:178] left extensions (zeros)

    return np.concatenate([
        angles,             # 15  right angles
        np.zeros(15),       # 15  left angles
        norm_pts,           # 63  right normalized
        np.zeros(63),       # 63  left normalized
        orientation,        # 3   right orientation
        np.zeros(3),        # 3   left orientation
        np.zeros(3),        # 3   right position (no pose)
        np.zeros(3),        # 3   left position
        extensions,         # 5   right extensions
        np.zeros(5),        # 5   left extensions
    ]).astype(np.float32)   # total = 178


# ─────────────────────────────────────────────
# Main Extraction Loop
# ─────────────────────────────────────────────

def extract_dataset(
    input_dir: Path,
    output_dir: Path,
    max_per_sign: int = None,
    batch_size: int = 1000,
):
    """
    Process all sign folders and save extracted features as .npy files.

    Args:
        input_dir:    Root folder with sign subfolders
        output_dir:   Where to save .npy output files
        max_per_sign: Max images per sign (None = all)
        batch_size:   How many samples per .npy file
    """
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        print("  Make sure you've placed the Kaggle dataset images at:")
        print(f"  {input_dir}/A/, {input_dir}/B/, etc.")
        return

    # Initialize MediaPipe — static_image_mode=True for photos
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        model_complexity=1,
    )

    sign_folders = sorted([
        f for f in input_dir.iterdir()
        if f.is_dir()
    ])

    if not sign_folders:
        print(f"[ERROR] No subfolders found in {input_dir}")
        return

    print(f"[Extract] Found {len(sign_folders)} sign folders")
    print(f"[Extract] Output → {output_dir}\n")

    total_extracted = 0
    total_failed = 0
    summary = {}

    for sign_folder in sign_folders:
        sign = sign_folder.name.upper()

        # Get all image files
        image_files = [
            f for f in sign_folder.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ]

        if max_per_sign:
            image_files = image_files[:max_per_sign]

        if not image_files:
            print(f"[{sign}] No images found, skipping.")
            continue

        print(f"[{sign}] Processing {len(image_files)} images...")

        features_list = []
        failed = 0

        for img_path in tqdm(image_files, desc=f"  {sign}", leave=False):
            feat = extract_features_from_image(str(img_path), hands)
            if feat is not None:
                features_list.append(feat)
            else:
                failed += 1

        extracted = len(features_list)
        total_extracted += extracted
        total_failed += failed
        summary[sign] = {"extracted": extracted, "failed": failed}

        if not features_list:
            print(f"  [WARN] No features extracted for '{sign}' — skipping.")
            continue

        # Save as .npy files in batches
        sign_out_dir = output_dir / sign
        sign_out_dir.mkdir(parents=True, exist_ok=True)

        features_array = np.array(features_list, dtype=np.float32)

        # Split into batch files if large
        num_batches = max(1, len(features_array) // batch_size + 1)
        batches = np.array_split(features_array, num_batches)

        for i, batch in enumerate(batches):
            out_path = sign_out_dir / f"session{i+1}.npy"
            np.save(out_path, batch)

        print(f"  ✓ {extracted} extracted | {failed} failed → saved to {sign_out_dir}/")

    hands.close()

    # ── Summary ──
    print("\n" + "="*50)
    print("EXTRACTION COMPLETE")
    print("="*50)
    print(f"Total extracted : {total_extracted}")
    print(f"Total failed    : {total_failed}")
    detection_rate = total_extracted / (total_extracted + total_failed) * 100 if (total_extracted + total_failed) > 0 else 0
    print(f"Detection rate  : {detection_rate:.1f}%")
    print(f"\nPer-sign summary:")
    for sign, stats in summary.items():
        print(f"  {sign:10s} → {stats['extracted']:5d} extracted | {stats['failed']:4d} failed")

    print(f"\nNext step: python data/scripts/preprocess.py")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe features from Kaggle ASL dataset images."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_DIR),
        help=f"Path to dataset root folder (default: {INPUT_DIR})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Path to output folder (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=MAX_PER_SIGN,
        help="Max images per sign. Useful for quick testing (default: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Samples per .npy file (default: {BATCH_SIZE})"
    )
    args = parser.parse_args()

    extract_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        max_per_sign=args.max,
        batch_size=args.batch_size,
    )