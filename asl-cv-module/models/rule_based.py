"""
models/rule_based.py
--------------------
Rule-based ASL fingerspelling classifier (A-Z).
Uses finger extension values and joint angles from the feature extractor
to classify static ASL hand signs without needing a trained ML model.

Each ASL letter has a distinct hand shape defined by which fingers are
extended/curled and the thumb position. This classifier encodes those
rules directly.
"""

import numpy as np
from typing import Optional


class RuleBasedASLClassifier:
    """
    Classifies ASL fingerspelling (A-Z) using hand landmark geometry.
    
    Uses:
    - finger_extensions: [thumb, index, middle, ring, pinky] each 0.0-1.0
    - normalized landmarks: 21 points × 3 coords = 63 values
    - joint angles: 15 angle values
    """

    def predict(self, feature_vector: np.ndarray) -> tuple[str, float]:
        """
        Predict ASL letter from a 178-dim feature vector.
        
        Feature layout (from extractor.py):
          [0:15]    right hand joint angles
          [15:30]   left hand joint angles  
          [30:93]   right hand normalized positions (21×3)
          [93:156]  left hand normalized positions
          [156:159] right hand orientation
          [159:162] left hand orientation
          [162:165] right hand body position
          [165:168] left hand body position
          [168:173] right finger extensions
          [173:178] left finger extensions
        """
        # Extract right hand features
        angles = feature_vector[0:15]
        positions = feature_vector[30:93].reshape(21, 3)
        extensions = feature_vector[168:173]
        
        # If no meaningful data, return unknown
        if np.sum(np.abs(extensions)) < 0.01:
            return "", 0.0
        
        thumb_ext = extensions[0]
        index_ext = extensions[1]
        middle_ext = extensions[2]
        ring_ext = extensions[3]
        pinky_ext = extensions[4]
        
        # Thresholds
        EXT = 0.35    # finger is extended
        CURL = 0.25   # finger is curled
        
        # Helper: check fingertip positions relative to each other
        thumb_tip = positions[4]
        index_tip = positions[8]
        middle_tip = positions[12]
        ring_tip = positions[16]
        pinky_tip = positions[20]
        
        index_mcp = positions[5]
        middle_mcp = positions[9]
        index_pip = positions[6]
        
        # Thumb-index distance
        thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
        thumb_middle_dist = np.linalg.norm(thumb_tip - middle_tip)
        thumb_ring_dist = np.linalg.norm(thumb_tip - ring_tip)
        thumb_pinky_dist = np.linalg.norm(thumb_tip - pinky_tip)
        
        # Index-middle distance
        index_middle_dist = np.linalg.norm(index_tip - middle_tip)
        
        # Classify based on finger states
        scores = {}
        
        # ── A: Fist with thumb alongside (all fingers curled, thumb slightly out)
        if index_ext < EXT and middle_ext < EXT and ring_ext < EXT and pinky_ext < EXT:
            if thumb_ext < 0.5:
                scores['A'] = 0.85
        
        # ── B: All four fingers extended, thumb curled across palm
        if index_ext >= EXT and middle_ext >= EXT and ring_ext >= EXT and pinky_ext >= EXT:
            if thumb_ext < EXT:
                scores['B'] = 0.85
        
        # ── C: Curved hand (all fingers partially extended, forming a C shape)
        if 0.15 < index_ext < 0.6 and 0.15 < middle_ext < 0.6:
            if 0.15 < ring_ext < 0.6 and 0.15 < pinky_ext < 0.6:
                scores['C'] = 0.70
        
        # ── D: Index extended, other three curled touching thumb
        if index_ext >= EXT and middle_ext < EXT and ring_ext < EXT and pinky_ext < EXT:
            if thumb_ext < EXT:
                scores['D'] = 0.82
        
        # ── E: All fingers curled into palm, thumb across
        if index_ext < CURL and middle_ext < CURL and ring_ext < CURL and pinky_ext < CURL:
            if thumb_ext < CURL:
                scores['E'] = 0.75
        
        # ── F: Index-thumb touching, other three extended
        if middle_ext >= EXT and ring_ext >= EXT and pinky_ext >= EXT:
            if thumb_index_dist < 0.15 and index_ext < EXT:
                scores['F'] = 0.80
        
        # ── G: Index pointing sideways, thumb extended
        if index_ext >= EXT and middle_ext < EXT and ring_ext < EXT and pinky_ext < EXT:
            if thumb_ext >= EXT:
                # G is distinguished from L by hand orientation (pointing sideways)
                scores['G'] = 0.70
        
        # ── H: Index and middle extended sideways
        if index_ext >= EXT and middle_ext >= EXT and ring_ext < EXT and pinky_ext < EXT:
            scores['H'] = 0.72
        
        # ── I: Only pinky extended
        if pinky_ext >= EXT and index_ext < EXT and middle_ext < EXT and ring_ext < EXT:
            scores['I'] = 0.85
        
        # ── J: Same as I but with a downward motion (static: same as I)
        # J requires motion tracking, using I shape as base
        
        # ── K: Index and middle extended, spread apart, thumb touching middle
        if index_ext >= EXT and middle_ext >= EXT and ring_ext < EXT and pinky_ext < EXT:
            if index_middle_dist > 0.1:
                scores['K'] = 0.70
        
        # ── L: Index extended up, thumb extended out (L shape)
        if index_ext >= EXT and middle_ext < EXT and ring_ext < EXT and pinky_ext < EXT:
            if thumb_ext >= EXT:
                scores['L'] = 0.80
        
        # ── M: Three fingers over thumb (fist with thumb under three fingers)
        if index_ext < EXT and middle_ext < EXT and ring_ext < EXT and pinky_ext < EXT:
            if thumb_ext < CURL:
                scores['M'] = 0.60
        
        # ── N: Two fingers over thumb
        if index_ext < EXT and middle_ext < EXT and ring_ext < EXT and pinky_ext < EXT:
            scores['N'] = 0.55
        
        # ── O: All fingers curved to touch thumb (O shape)
        if 0.05 < index_ext < 0.4 and 0.05 < middle_ext < 0.4:
            if thumb_index_dist < 0.12 or thumb_middle_dist < 0.12:
                scores['O'] = 0.75
        
        # ── P: Similar to K but pointing down
        if index_ext >= EXT and middle_ext >= EXT and ring_ext < EXT:
            # P has index pointing down
            if index_tip[1] > index_mcp[1]:  # tip below MCP = pointing down
                scores['P'] = 0.70
        
        # ── Q: Similar to G but pointing down
        if index_ext >= EXT and middle_ext < EXT and ring_ext < EXT:
            if thumb_ext >= EXT and index_tip[1] > index_mcp[1]:
                scores['Q'] = 0.70
        
        # ── R: Index and middle crossed
        if index_ext >= EXT and middle_ext >= EXT and ring_ext < EXT and pinky_ext < EXT:
            if index_middle_dist < 0.06:  # fingers very close = crossed
                scores['R'] = 0.75
        
        # ── S: Fist with thumb across front of fingers
        if index_ext < EXT and middle_ext < EXT and ring_ext < EXT and pinky_ext < EXT:
            if thumb_ext >= CURL and thumb_ext < 0.5:
                scores['S'] = 0.70
        
        # ── T: Fist with thumb between index and middle
        if index_ext < EXT and middle_ext < EXT and ring_ext < EXT and pinky_ext < EXT:
            scores['T'] = 0.50
        
        # ── U: Index and middle extended together (close)
        if index_ext >= EXT and middle_ext >= EXT and ring_ext < EXT and pinky_ext < EXT:
            if index_middle_dist < 0.08:
                scores['U'] = 0.78
        
        # ── V: Index and middle extended and spread (peace sign)
        if index_ext >= EXT and middle_ext >= EXT and ring_ext < EXT and pinky_ext < EXT:
            if index_middle_dist >= 0.08:
                scores['V'] = 0.85
        
        # ── W: Index, middle, ring extended and spread
        if index_ext >= EXT and middle_ext >= EXT and ring_ext >= EXT and pinky_ext < EXT:
            scores['W'] = 0.85
        
        # ── X: Index finger hooked (partially curled)
        if 0.1 < index_ext < EXT and middle_ext < EXT and ring_ext < EXT and pinky_ext < EXT:
            scores['X'] = 0.70
        
        # ── Y: Thumb and pinky extended, rest curled (hang loose)
        if thumb_ext >= EXT and pinky_ext >= EXT:
            if index_ext < EXT and middle_ext < EXT and ring_ext < EXT:
                scores['Y'] = 0.90
        
        # ── Z: Index finger traces Z shape (requires motion, static: index pointed)
        # Static approximation: index pointing with slight hook
        
        # Pick the best match
        if not scores:
            # Fallback: use raw extension pattern to guess
            return self._fallback_classify(extensions), 0.60
        
        best_letter = max(scores, key=scores.get)
        best_confidence = scores[best_letter]
        
        return best_letter, best_confidence
    
    def _fallback_classify(self, extensions: np.ndarray) -> str:
        """Simple fallback based on number of extended fingers."""
        extended = sum(1 for e in extensions if e >= 0.35)
        
        if extended == 0:
            return 'A'  # Fist
        elif extended == 1:
            if extensions[1] >= 0.35:  # index
                return 'D'
            elif extensions[4] >= 0.35:  # pinky
                return 'I'
            else:
                return 'A'
        elif extended == 2:
            if extensions[0] >= 0.35 and extensions[4] >= 0.35:
                return 'Y'
            elif extensions[0] >= 0.35 and extensions[1] >= 0.35:
                return 'L'
            else:
                return 'V'
        elif extended == 3:
            return 'W'
        elif extended == 4:
            return 'B'
        else:  # 5
            return 'B'
