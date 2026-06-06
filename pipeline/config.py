"""
Configuration centrale du système Smart Traffic Agadir
PFE M2 IA Embarquée — Mouad ASSARGUAL
FSA Aït Melloul, Université Ibn Zohr
"""

import os
import numpy as np

# ═══════════════════════════════════════════════════════
# CHEMINS
# ═══════════════════════════════════════════════════════
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
DATA_DIR    = os.path.join(BASE_DIR, "data")
VIDEOS_DIR  = os.path.join(DATA_DIR, "videos")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════
# MODÈLES
# ═══════════════════════════════════════════════════════
YOLO_MODEL        = os.path.join(MODELS_DIR, "YOLO26n_step2_800_best.onnx")
YOLO26N_MODEL     = os.path.join(MODELS_DIR, "YOLO26n_step2_800_best.onnx")
YOLO26S_MODEL     = os.path.join(MODELS_DIR, "YOLO26s_step2_960_best.onnx")
FACE_MODEL        = os.path.join(MODELS_DIR, "mobilenet_ssd_face.onnx")
ONNX_MODELS = {
    "yolo26n_800": YOLO26N_MODEL,
    "yolo26s_960": YOLO26S_MODEL,
}
CONF_THRESH       = 0.35
IOU_THRESH        = 0.45
IMG_SIZE          = 800

# ═══════════════════════════════════════════════════════
# CLASSES
# ═══════════════════════════════════════════════════════
CLASS_NAMES = ['bus', 'car', 'emergency_vehicle',
               'motorcycle', 'person', 'truck']

CLASS_COLORS = {
    'bus'              : (255, 165,   0),   # Orange
    'car'              : (  0, 255,   0),   # Vert
    'emergency_vehicle': (  0,   0, 255),   # Rouge
    'motorcycle'       : (255, 255,   0),   # Jaune
    'person'           : (255,   0, 255),   # Magenta
    'truck'            : (  0, 255, 255),   # Cyan
}

# ═══════════════════════════════════════════════════════
# MDP — PROCESSUS DE DÉCISION MARKOVIEN
# ═══════════════════════════════════════════════════════

# Poids par type de véhicule (encombrement)
VEHICLE_WEIGHTS = {
    'car'              : 1.0,
    'motorcycle'       : 0.5,
    'truck'            : 2.0,
    'bus'              : 3.0,
    'person'           : 1.5,   # Usager vulnérable
    'emergency_vehicle': 100.0, # Priorité absolue
}

# Durées de phase (secondes)
PHASE_DURATIONS = {
    'min'      : 15,
    'medium'   : 30,
    'max'      : 45,
    'emergency': 45,
    'pedestrian': 30,
    'all_red'  : 15,
}

# Seuils de décision
THRESHOLD_HIGH   = 10  # Score élevé → 45s
THRESHOLD_MEDIUM =  5  # Score moyen → 30s

# Phases de feux
PHASES = {
    'NS' : {'green': ['N', 'S'], 'red': ['E', 'W']},
    'EW' : {'green': ['E', 'W'], 'red': ['N', 'S']},
}

# ═══════════════════════════════════════════════════════
# ZONES POLYGONALES — PROFIL NE8TH (BELLEVUE)
# ═══════════════════════════════════════════════════════
ZONE_PROFILES = {
    'ne8th' : {
        'N': np.array([[594,46],[538,47],[481,241],
                       [868,227],[598,46]], np.int32),
        'E': np.array([[873,228],[1118,508],[1275,291],
                       [1235,250],[873,227]], np.int32),
        'S': np.array([[1118,509],[542,678],[913,695],
                       [1268,694],[1119,510]], np.int32),
        'W': np.array([[3,355],[479,240],[418,592],
                       [3,531],[3,357]], np.int32),
    },
    '116th' : {
        'N': np.array([[447,25],[716,146],[452,202],
                       [406,38],[439,25]], np.int32),
        'E': np.array([[791,157],[1032,311],[1104,136],
                       [991,77],[788,159]], np.int32),
        'S': np.array([[1087,404],[763,716],[1263,704],
                       [1269,614],[1089,403]], np.int32),
        'W': np.array([[400,250],[503,681],[12,707],
                       [7,414],[396,250]], np.int32),
    },
}

# Zone active par défaut
DEFAULT_PROFILE = 'ne8th'

# ═══════════════════════════════════════════════════════
# ROIs PIÉTONS — DÉTECTION HAUTE RÉSOLUTION OPTIONNELLE
# ═══════════════════════════════════════════════════════
#
# Coordonnées relatives (x1, y1, x2, y2) dans l'image complète.
# Ces zones restent volontairement larges : elles couvrent trottoirs,
# bords de route, îlots et zones d'approche, pas seulement les passages piétons.
PERSON_ROI_PROFILES = {
    'ne8th': [
        ('left_sidewalk',   (0.00, 0.25, 0.42, 0.92)),
        ('center_crossing', (0.28, 0.22, 0.74, 0.88)),
        ('right_sidewalk',  (0.58, 0.20, 1.00, 0.88)),
    ],
    '116th': [
        ('left_sidewalk',   (0.00, 0.25, 0.45, 0.95)),
        ('center_crossing', (0.28, 0.18, 0.76, 0.90)),
        ('right_sidewalk',  (0.55, 0.15, 1.00, 0.88)),
    ],
}
PERSON_ROI_ENABLED_DEFAULT = False
PERSON_ROI_EVERY          = 5
PERSON_ROI_DEDUP_IOU      = 0.50

# ═══════════════════════════════════════════════════════
# ANONYMISATION — PRIVACY BY DESIGN
# ═══════════════════════════════════════════════════════
BLUR_KERNEL_SIZE = (51, 51)   # Taille du flou gaussien
BLUR_SIGMA       = 30          # Intensité du flou
FACE_CONF_THRESH = 0.5         # Seuil détection visage

# ═══════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════
DASHBOARD_HOST   = '0.0.0.0'
DASHBOARD_PORT   = 5000
STREAM_FPS       = 10

# ═══════════════════════════════════════════════════════
# SCÉNARIOS DE VALIDATION MDP
# ═══════════════════════════════════════════════════════
TEST_SCENARIOS = {
    'normal'     : {'N':5, 'S':2, 'E':3, 'W':1, 'p':0, 'e':0},
    'pedestrian' : {'N':3, 'S':3, 'E':2, 'W':2, 'p':5, 'e':0},
    'emergency'  : {'N':8, 'S':1, 'E':4, 'W':2, 'p':2, 'e':1},
    'rush_hour'  : {'N':12,'S':8, 'E':10,'W':6, 'p':3, 'e':0},
    'empty'      : {'N':0, 'S':0, 'E':0, 'W':0, 'p':0, 'e':0},
}

if __name__ == '__main__':
    print("Configuration Smart Traffic Agadir ✅")
    print(f"Base dir   : {BASE_DIR}")
    print(f"Models dir : {MODELS_DIR}")
    print(f"Classes    : {CLASS_NAMES}")
    print(f"Profils    : {list(ZONE_PROFILES.keys())}")
