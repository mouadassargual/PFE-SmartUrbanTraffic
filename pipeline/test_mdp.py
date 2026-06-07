# test_mdp.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.decision import MDPDecision

mdp = MDPDecision()

scenarios = [
    {
        "name": "Dense Nord",
        "state": {
            "N": {"count": 12, "score": 12.0},
            "S": {"count": 2, "score": 2.0},
            "E": {"count": 1, "score": 1.0},
            "W": {"count": 1, "score": 1.0},
            "pedestrians": 2,
            "emergency": False,
            "emergency_zone": None,
        },
        "expected": "NS"
    },
    {
        "name": "Axe EW dominant malgre N eleve",
        "state": {
            "N": {"count": 8, "score": 8.0},
            "S": {"count": 0, "score": 0.0},
            "E": {"count": 5, "score": 5.0},
            "W": {"count": 6, "score": 6.0},
            "pedestrians": 0,
            "emergency": False,
            "emergency_zone": None,
        },
        "expected": "EW"
    },
    {
        "name": "Trafic equilibre",
        "state": {
            "N": {"count": 4, "score": 4.0},
            "S": {"count": 4, "score": 4.0},
            "E": {"count": 4, "score": 4.0},
            "W": {"count": 4, "score": 4.0},
            "pedestrians": 0,
            "emergency": False,
            "emergency_zone": None,
        },
        "expected": "NS"
    },
    {
        "name": "Urgence Est",
        "state": {
            "N": {"count": 2, "score": 2.0},
            "S": {"count": 2, "score": 2.0},
            "E": {"count": 3, "score": 3.0},
            "W": {"count": 1, "score": 1.0},
            "pedestrians": 0,
            "emergency": True,
            "emergency_zone": "E",
        },
        "expected": "EMERGENCY"
    },
    {
        "name": "Pietons seuls",
        "state": {
            "N": {"count": 0, "score": 0.0},
            "S": {"count": 0, "score": 0.0},
            "E": {"count": 0, "score": 0.0},
            "W": {"count": 0, "score": 0.0},
            "pedestrians": 4,
            "emergency": False,
            "emergency_zone": None,
        },
        "expected": "PEDESTRIAN"
    },
    {
        "name": "Aucun trafic",
        "state": {
            "N": {"count": 0, "score": 0.0},
            "S": {"count": 0, "score": 0.0},
            "E": {"count": 0, "score": 0.0},
            "W": {"count": 0, "score": 0.0},
            "pedestrians": 0,
            "emergency": False,
            "emergency_zone": None,
        },
        "expected": "ALL_RED"
    },
]

print("=" * 50)
print(f"TEST MDP — {len(scenarios)} scénarios")
print("=" * 50)

all_passed = True
for s in scenarios:
    result = mdp.decide(s["state"])
    status = "✅ OK" if result["phase"] == s["expected"] else "❌ FAIL"
    if result["phase"] != s["expected"]:
        all_passed = False
    print(f"{status} | {s['name']}")
    print(f"     Phase produite  : {result['phase']}")
    print(f"     Phase attendue  : {s['expected']}")
    print(f"     Durée           : {result['duration']}s")
    print(f"     Raison          : {result['reason']}")
    print()

print("=" * 50)
if all_passed:
    print(f"✅ TOUS LES SCÉNARIOS VALIDÉS — {len(scenarios)}/{len(scenarios)}")
else:
    print("❌ CERTAINS SCÉNARIOS ONT ÉCHOUÉ")
print("=" * 50)
