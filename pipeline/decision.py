"""
Module MDP de decision adaptative.

Etat attendu:
{
  N:{count,score}, S:{count,score}, E:{count,score}, W:{count,score},
  pedestrians:int, emergency:bool, emergency_zone:str|None
}
"""

from __future__ import annotations

import time

import cv2

from pipeline.config import PHASE_DURATIONS, THRESHOLD_HIGH, THRESHOLD_MEDIUM, VEHICLE_WEIGHTS


class MDPDecision:
    """Politique MDP simple pour choisir phase et duree."""

    def __init__(self, thresholds=None, durations=None):
        self.thresholds = thresholds or {
            "HIGH": THRESHOLD_HIGH,
            "MEDIUM": THRESHOLD_MEDIUM,
        }
        self.durations = durations or PHASE_DURATIONS
        self.history = []
        self.decision_count = 0

        print("✅ MDPDecision initialise")
        print(f"   Seuils : HIGH={self.thresholds['HIGH']}, MEDIUM={self.thresholds['MEDIUM']}")

    def compute_score(self, state, direction):
        """Score NS = score(N)+score(S), Score EW = score(E)+score(W)."""
        if direction == "NS":
            base = state.get("N", {}).get("score", 0.0) + state.get("S", {}).get("score", 0.0)
        elif direction == "EW":
            base = state.get("E", {}).get("score", 0.0) + state.get("W", {}).get("score", 0.0)
        else:
            base = state.get(direction, {}).get("score", 0.0)

        # Les pietons sont une pression globale de securite, ajoutee aux deux axes.
        return float(base + state.get("pedestrians", 0) * VEHICLE_WEIGHTS.get("person", 1.5))

    def _duration_for_score(self, score):
        if score >= self.thresholds["HIGH"]:
            return self.durations["max"]
        if score >= self.thresholds["MEDIUM"]:
            return self.durations["medium"]
        return self.durations["min"]

    def decide(self, state):
        """
        SI emergency: EMERGENCY 45s.
        SINON: choisir NS/EW par score dominant puis duree selon seuils.
        """
        self.decision_count += 1
        score_ns = self.compute_score(state, "NS")
        score_ew = self.compute_score(state, "EW")

        if state.get("emergency", False):
            decision = {
                "phase": "EMERGENCY",
                "duration": self.durations.get("emergency", self.durations["max"]),
                "score_NS": score_ns,
                "score_EW": score_ew,
                "reason": "Vehicule urgence detecte",
                "timestamp": time.time(),
                "green_dirs": ["N", "S", "E", "W"],
                "red_dirs": [],
            }
            self.history.append(decision)
            return decision

        if score_ns >= score_ew:
            phase = "NS"
            dominant = score_ns
            reason = "Axe Nord/Sud dominant"
            green_dirs = ["N", "S"]
            red_dirs = ["E", "W"]
        else:
            phase = "EW"
            dominant = score_ew
            reason = "Axe Est/Ouest dominant"
            green_dirs = ["E", "W"]
            red_dirs = ["N", "S"]

        decision = {
            "phase": phase,
            "duration": self._duration_for_score(dominant),
            "score_NS": score_ns,
            "score_EW": score_ew,
            "reason": reason,
            "timestamp": time.time(),
            "green_dirs": green_dirs,
            "red_dirs": red_dirs,
        }
        self.history.append(decision)
        return decision

    def draw_decision(self, frame, decision, anonymized=0):
        """Overlay bas de frame avec phase, duree, score et privacy."""
        h, w = frame.shape[:2]
        emergency = decision["phase"] == "EMERGENCY"
        bg = (0, 0, 180) if emergency else (20, 90, 20)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 82), (w, h), bg, -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        score = max(decision["score_NS"], decision["score_EW"])
        line1 = (
            f"MDP: Phase {decision['phase']} | Duree {decision['duration']}s | "
            f"Score {score:.1f}"
        )
        line2 = (
            f"NS={decision['score_NS']:.1f} | EW={decision['score_EW']:.1f} | "
            f"Privacy-by-Design: {anonymized} anonymise(s)"
        )
        cv2.putText(frame, line1, (18, h - 48), cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 1)
        cv2.putText(frame, line2, (18, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)
        return frame

    def get_stats(self):
        return {
            "total_decisions": self.decision_count,
            "history_size": len(self.history),
        }


class MDPDecisionModule(MDPDecision):
    """Alias de compatibilite avec l'ancien pipeline."""

    pass


if __name__ == "__main__":
    mdp = MDPDecision()
    sample = {
        "N": {"count": 4, "score": 4.0},
        "S": {"count": 3, "score": 3.0},
        "E": {"count": 1, "score": 1.0},
        "W": {"count": 1, "score": 1.0},
        "pedestrians": 2,
        "emergency": False,
        "emergency_zone": None,
    }
    print(mdp.decide(sample))
