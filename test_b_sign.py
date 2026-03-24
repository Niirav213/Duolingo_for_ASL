import sys
sys.path.append(r"c:\Users\kisho\Duolingo_for_ASL\asl-cv-module")
import numpy as np
from core.extractor import FeatureVector
from models.rule_based import RuleBasedASLClassifier
from api.pipeline import ASLPipeline
from core.scorer import ScoreResult

# Mock features for B where thumb is slightly extended (e.g. 0.4 > 0.35)
v = np.zeros(178, dtype=np.float32)
# Thumb
v[168] = 0.4 
# Index, middle, ring, pinky all extended
v[169] = 0.9
v[170] = 0.9
v[171] = 0.9
v[172] = 0.9

classifier = RuleBasedASLClassifier()
label, conf = classifier.predict(v)
print(f"RuleBased -> {label}: {conf}")

# Let's test pipeline logic
target_sign = "B"
detected_sign = label
confidence = conf
score_result = ScoreResult()

sign_matches = (detected_sign.upper() == target_sign.upper()) and confidence >= 0.35
print(f"Sign Matches: {sign_matches}")

if score_result.overall_score == 0.0 and sign_matches:
    score_result.overall_score = confidence * 100
    score_result.is_correct = True

print(f"Pipeline -> is_correct={score_result.is_correct}, score={score_result.overall_score}")
