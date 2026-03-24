import sys
sys.path.append(r"c:\Users\kisho\Duolingo_for_ASL\asl-cv-module")

from core.detector import DetectionResult
from core.extractor import FeatureVector
from models.rule_based import RuleBasedASLClassifier
from api.pipeline import ASLPipeline
import numpy as np

features = FeatureVector()
features.is_valid = True

# Mock a valid FeatureVector with 178 elements
# Let's mock a perfect 'A' sign (thumb alongside, fingers curled)
v = np.zeros(178, dtype=np.float32)

# Set right finger extensions
# [168:173] right finger extensions: thumb, index, middle, ring, pinky
v[168] = 0.2  # thumb (curled)
v[169] = 0.1  # index (curled)
v[170] = 0.1  # middle (curled)
v[171] = 0.1  # ring (curled)
v[172] = 0.1  # pinky (curled)

features.vector = v

# Test RuleBased directly
classifier = RuleBasedASLClassifier()
label, conf = classifier.predict(features.vector)
print(f"RuleBased Output: label={label}, conf={conf}")

# Test Pipeline directly
pipeline = ASLPipeline(load_static=True, load_dynamic=False)
features.right_hand_angles = np.zeros(15)
features.right_hand_position = np.zeros(3)
features.right_hand_orientation = np.zeros(3)
features.right_finger_extensions = v[168:173]
features.right_hand_normalized = np.zeros(63)

# Mock detection for scorer to avoid NoneType errors
from core.scorer import ScoreResult
res = pipeline.scorer.score(features, "A")
print(f"Scorer Overall Score: {res.overall_score}")

# Test the exact override logic from pipeline.py line 144
target_sign = "A"
detected_sign = label
confidence = conf

sign_matches = (detected_sign.upper() == target_sign.upper()) and confidence > 0.5
print(f"Sign matches: {sign_matches}")

if res.overall_score == 0.0 and sign_matches:
    res.overall_score = confidence * 100
    res.is_correct = True
elif res.overall_score == 0.0 and detected_sign:
    res.overall_score = confidence * 30
    res.is_correct = False

print(f"Final pipeline override result: score={res.overall_score}, correct={res.is_correct}")
