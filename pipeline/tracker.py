"""
IoU Tracker — Suivi de véhicules entre frames
Algorithme léger adapté au Raspberry Pi (pas de dépendance externe)
PFE M2 IA Embarquée — Mouad ASSARGUAL
FSA Aït Melloul, Université Ibn Zohr
"""

import cv2
import numpy as np
from pipeline.config import CLASS_NAMES, VEHICLE_WEIGHTS


class Track:
    """Représente un objet suivi à travers les frames"""

    _id_counter = 0

    def __init__(self, bbox, class_name, conf):
        Track._id_counter += 1
        self.id         = Track._id_counter
        self.bbox       = bbox          # [x1, y1, x2, y2]
        self.class_name = class_name
        self.conf       = conf
        self.age        = 1             # nombre de frames depuis création
        self.hits       = 1             # nombre de fois détecté
        self.misses     = 0             # frames consécutives sans détection
        self.active     = True

    def update(self, bbox, conf):
        """Met à jour la position et les stats du track"""
        self.bbox   = bbox
        self.conf   = conf
        self.hits  += 1
        self.misses = 0
        self.age   += 1
        self.active = True

    def mark_missed(self):
        """Incrémente le compteur de frames manquées"""
        self.misses += 1
        self.age    += 1

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class IoUTracker:
    """
    IoU Tracker simple et efficace pour Edge AI.

    Principe :
      1. Pour chaque nouvelle détection, calculer l'IoU avec tous les tracks actifs
      2. Associer détection ↔ track si IoU > seuil
      3. Les tracks non associés après MAX_MISSES frames sont supprimés
      4. Les nouvelles détections sans correspondance créent un nouveau track

    Complexité : O(N×M) avec N=tracks, M=détections → négligeable sur Pi
    """

    def __init__(self, iou_thresh=0.3, max_misses=3, min_hits=1):
        """
        Args:
            iou_thresh : seuil IoU pour associer détection ↔ track (0.3 recommandé)
            max_misses : nombre de frames sans détection avant suppression du track
            min_hits   : hits minimum avant de considérer un track comme confirmé
        """
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses
        self.min_hits   = min_hits
        self.tracks     = []            # liste des tracks actifs
        self.frame_count = 0

    # ─────────────────────────────────────────────────────────
    # Calcul IoU
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def compute_iou(boxA, boxB):
        """
        Calcule l'Intersection over Union entre deux bounding boxes.

        Args:
            boxA, boxB : [x1, y1, x2, y2]
        Returns:
            float IoU ∈ [0, 1]
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter   = inter_w * inter_h

        if inter == 0:
            return 0.0

        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = areaA + areaB - inter

        return inter / union if union > 0 else 0.0

    @staticmethod
    def iou_matrix(tracks, detections):
        """
        Calcule la matrice IoU (N_tracks × N_detections).
        """
        if not tracks or not detections:
            return np.zeros((len(tracks), len(detections)))

        mat = np.zeros((len(tracks), len(detections)))
        for i, t in enumerate(tracks):
            for j, d in enumerate(detections):
                mat[i, j] = IoUTracker.compute_iou(t.bbox, d['bbox'])
        return mat

    # ─────────────────────────────────────────────────────────
    # Association greedy (suffisante pour trafic)
    # ─────────────────────────────────────────────────────────

    def _associate(self, detections):
        """
        Association greedy détections ↔ tracks par IoU décroissant.

        Returns:
            matched    : liste de (track_idx, det_idx)
            unmatched_tracks : indices de tracks sans détection
            unmatched_dets   : indices de détections sans track
        """
        if not self.tracks or not detections:
            return [], list(range(len(self.tracks))), list(range(len(detections)))

        iou_mat = self.iou_matrix(self.tracks, detections)

        matched           = []
        used_tracks       = set()
        used_dets         = set()

        # Trier par IoU décroissant → greedy matching
        order = np.argsort(-iou_mat, axis=None)
        for flat_idx in order:
            t_idx, d_idx = divmod(flat_idx, len(detections))
            if iou_mat[t_idx, d_idx] < self.iou_thresh:
                break
            if t_idx in used_tracks or d_idx in used_dets:
                continue
            # Vérifier même classe
            if self.tracks[t_idx].class_name != detections[d_idx]['class_name']:
                continue
            matched.append((t_idx, d_idx))
            used_tracks.add(t_idx)
            used_dets.add(d_idx)

        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_tracks]
        unmatched_dets   = [j for j in range(len(detections))  if j not in used_dets]

        return matched, unmatched_tracks, unmatched_dets

    # ─────────────────────────────────────────────────────────
    # API principale
    # ─────────────────────────────────────────────────────────

    def update(self, detections):
        """
        Met à jour les tracks avec les nouvelles détections.

        Args:
            detections : liste de dicts issus de YOLODetector.detect()

        Returns:
            liste de Track actifs et confirmés (hits >= min_hits)
        """
        self.frame_count += 1

        matched, unmatched_tracks, unmatched_dets = self._associate(detections)

        # Mettre à jour les tracks associés
        for t_idx, d_idx in matched:
            det = detections[d_idx]
            self.tracks[t_idx].update(det['bbox'], det['conf'])

        # Incrémenter les misses des tracks non associés
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].mark_missed()

        # Créer de nouveaux tracks pour les détections non associées
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            self.tracks.append(
                Track(det['bbox'], det['class_name'], det['conf'])
            )

        # Supprimer les tracks trop anciens
        self.tracks = [
            t for t in self.tracks
            if t.misses <= self.max_misses
        ]

        # Retourner uniquement les tracks confirmés
        return [t for t in self.tracks if t.hits >= self.min_hits]

    def get_active_tracks(self):
        """Retourne tous les tracks actifs confirmés"""
        return [t for t in self.tracks if t.hits >= self.min_hits]

    def count_by_class(self, confirmed_tracks=None):
        """
        Compte les objets par classe parmi les tracks actifs.

        Returns:
            dict {class_name: count}
        """
        tracks = confirmed_tracks or self.get_active_tracks()
        counts = {c: 0 for c in CLASS_NAMES}
        for t in tracks:
            if t.class_name in counts:
                counts[t.class_name] += 1
        return counts

    def reset(self):
        """Remet à zéro le tracker (nouveau flux vidéo)"""
        self.tracks      = []
        self.frame_count = 0
        Track._id_counter = 0

    def get_stats(self):
        return {
            'frame'        : self.frame_count,
            'active_tracks': len(self.get_active_tracks()),
            'total_tracks' : Track._id_counter,
        }


class ZoneTracker:
    """Comptage par zones polygonales N/S/E/W avec scores MDP pondérés."""

    def __init__(self, zones_dict, vehicle_weights=None):
        self.zones = zones_dict
        self.vehicle_weights = vehicle_weights or VEHICLE_WEIGHTS
        self.transport_classes = ["car", "motorcycle", "truck", "bus", "emergency_vehicle"]
        self.cumulative = {
            direction: {
                "count": 0,
                "score": 0.0,
                "pedestrians": 0,
                "by_class": {class_name: 0 for class_name in self.transport_classes},
            }
            for direction in ("N", "S", "E", "W")
        }
        self.emergency_count = 0

    @staticmethod
    def point_in_zone(cx, cy, zone_polygon):
        """Retourne True si le point est dans le polygone."""
        return cv2.pointPolygonTest(zone_polygon, (int(cx), int(cy)), False) >= 0

    @staticmethod
    def compute_iou(bbox1, bbox2):
        """Intersection over Union."""
        return IoUTracker.compute_iou(bbox1, bbox2)

    @staticmethod
    def _center(item):
        if hasattr(item, "center"):
            return item.center
        x1, y1, x2, y2 = item["bbox"]
        return (x1 + x2) / 2, (y1 + y2) / 2

    @staticmethod
    def _class_name(item):
        return item.class_name if hasattr(item, "class_name") else item.get("class_name")

    def update(self, detections):
        """
        Calcule l'état courant par zone.

        Args:
            detections: liste de dicts de détection ou liste de Track.

        Returns:
            {
              N:{count,score,pedestrians,by_class}, S:{...}, E:{...}, W:{...},
              pedestrians:int, emergency:bool, emergency_zone:str|None
            }
        """
        state = {
            direction: {
                "count": 0,
                "score": 0.0,
                "pedestrians": 0,
                "by_class": {class_name: 0 for class_name in self.transport_classes},
            }
            for direction in ("N", "S", "E", "W")
        }
        pedestrians = 0
        emergency = False
        emergency_zone = None

        for item in detections:
            class_name = self._class_name(item)
            if class_name is None:
                continue

            cx, cy = self._center(item)

            if class_name == "person":
                pedestrians += 1
                for direction, polygon in self.zones.items():
                    if self.point_in_zone(cx, cy, polygon):
                        state[direction]["pedestrians"] += 1
                        break
                continue

            item_zone = None
            for direction, polygon in self.zones.items():
                if self.point_in_zone(cx, cy, polygon):
                    item_zone = direction
                    break
            if item_zone is None:
                continue

            weight = float(self.vehicle_weights.get(class_name, 1.0))
            state[item_zone]["count"] += 1
            state[item_zone]["score"] += weight
            state[item_zone]["by_class"].setdefault(class_name, 0)
            state[item_zone]["by_class"][class_name] += 1

            if class_name == "emergency_vehicle":
                emergency = True
                emergency_zone = item_zone
                self.emergency_count += 1

        for direction in ("N", "S", "E", "W"):
            self.cumulative[direction]["count"] += state[direction]["count"]
            self.cumulative[direction]["score"] += state[direction]["score"]
            self.cumulative[direction]["pedestrians"] += state[direction]["pedestrians"]
            for class_name, count in state[direction]["by_class"].items():
                self.cumulative[direction]["by_class"].setdefault(class_name, 0)
                self.cumulative[direction]["by_class"][class_name] += count

        state["pedestrians"] = pedestrians
        state["emergency"] = emergency
        state["emergency_zone"] = emergency_zone
        return state

    def draw_zones(self, frame, state, phase=None):
        """Dessine les polygones et compteurs officiels, sans simuler les feux."""
        zone_colors = {
            "N": (255, 180, 40),
            "E": (60, 220, 255),
            "S": (80, 255, 120),
            "W": (255, 80, 220),
        }
        overlay = frame.copy()
        for direction, polygon in self.zones.items():
            color = zone_colors.get(direction, (255, 255, 255))
            cv2.fillPoly(overlay, [polygon], color)
            cv2.polylines(frame, [polygon], True, color, 2)

            moments = cv2.moments(polygon)
            if moments["m00"] == 0:
                continue
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            zone_state = state.get(direction, {"count": 0, "score": 0.0})
            by_class = zone_state.get("by_class", {})
            text_parts = [
                f"{name[:3]}:{by_class.get(name, 0)}"
                for name in self.transport_classes
                if by_class.get(name, 0) > 0
            ]
            if zone_state.get("pedestrians", 0) > 0:
                text_parts.append(f"per:{zone_state['pedestrians']}")
            class_text = " ".join(text_parts)
            cv2.circle(frame, (cx, cy - 34), 13, color, -1)
            cv2.putText(
                frame,
                f"{direction} c:{zone_state['count']} s:{zone_state['score']:.1f}",
                (cx - 58, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
            )
            if class_text:
                cv2.putText(
                    frame,
                    class_text,
                    (cx - 58, cy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                )

        cv2.addWeighted(overlay, 0.13, frame, 0.87, 0, frame)
        return frame

    def get_stats(self):
        return {
            "zone_stats": self.cumulative,
            "emergency_count": self.emergency_count,
        }


# ─────────────────────────────────────────────────────────
# Test autonome
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Test IoUTracker...")

    tracker = IoUTracker(iou_thresh=0.3, max_misses=3)

    # Simuler 3 frames avec des détections légèrement décalées
    frames_dets = [
        # Frame 1 — 2 véhicules + 1 piéton
        [
            {'bbox': [100, 100, 200, 200], 'class_name': 'motorcycle', 'conf': 0.9},
            {'bbox': [400, 300, 550, 450], 'class_name': 'car',        'conf': 0.85},
            {'bbox': [700, 200, 740, 320], 'class_name': 'person',     'conf': 0.8},
        ],
        # Frame 2 — même objets, légèrement déplacés
        [
            {'bbox': [105, 102, 205, 202], 'class_name': 'motorcycle', 'conf': 0.88},
            {'bbox': [410, 305, 560, 455], 'class_name': 'car',        'conf': 0.87},
            {'bbox': [702, 205, 742, 325], 'class_name': 'person',     'conf': 0.79},
        ],
        # Frame 3 — moto disparue, nouvel objet d'urgence
        [
            {'bbox': [415, 308, 565, 458], 'class_name': 'car',              'conf': 0.86},
            {'bbox': [705, 208, 745, 328], 'class_name': 'person',           'conf': 0.81},
            {'bbox': [900, 400, 1050, 500],'class_name': 'emergency_vehicle','conf': 0.95},
        ],
    ]

    for i, dets in enumerate(frames_dets):
        active = tracker.update(dets)
        counts = tracker.count_by_class(active)
        print(f"\n  Frame {i+1} — {len(active)} tracks actifs :")
        for cls, cnt in counts.items():
            if cnt > 0:
                print(f"    {cls:<20}: {cnt}")
        for t in active:
            print(f"    ID={t.id:>3} | {t.class_name:<20} | "
                  f"hits={t.hits} misses={t.misses}")

    print(f"\n  Stats finales : {tracker.get_stats()}")
    print("\nTest IoUTracker ✅")
