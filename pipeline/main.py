"""
Orchestrateur principal Smart Traffic Agadir.

Pipeline: detection -> ROI person -> anonymisation -> tracking -> zones -> MDP
Production Raspberry Pi 5: ONNX Runtime CPU, OpenCV, Flask optionnel.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pipeline.anonymizer import Anonymizer
from pipeline.config import (
    CLASS_COLORS,
    CONF_THRESH,
    DASHBOARD_PORT,
    DEFAULT_PROFILE,
    DEFAULT_SCENE,
    IMG_SIZE,
    PERSON_ROI_EVERY,
    PERSON_ROI_PROFILES,
    RESULTS_DIR,
    SCENE_PROFILES,
    VEHICLE_WEIGHTS,
    YOLO_MODEL,
    ZONE_PROFILES,
)
from pipeline.dashboard import Dashboard
from pipeline.decision import MDPDecision
from pipeline.detector import YOLODetector
from pipeline.tracker import IoUTracker, ZoneTracker


HEADER_TEXT = "SMART TRAFFIC AGADIR"
ANALYSIS_ZONE_COLORS = {
    "N": (255, 180, 40),
    "E": (60, 220, 255),
    "S": (80, 255, 120),
    "W": (255, 80, 220),
}
VEHICLE_ROI_CLASSES = ("bus", "car", "emergency_vehicle", "motorcycle", "truck")


def load_zone_polygons(zones_json):
    data = json.loads(Path(zones_json).read_text())
    zones = data.get("zones", data)
    required = ["N", "E", "S", "W"]
    missing = [zone for zone in required if zone not in zones or len(zones[zone]) < 3]
    if missing:
        raise ValueError(f"Zones JSON incomplet, polygones manquants/invalides: {missing}")
    return {
        zone: np.array(zones[zone], np.int32)
        for zone in required
    }


def draw_header_footer(frame, fps, frame_id, model_name):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (12, 20, 28), -1)
    cv2.rectangle(overlay, (0, h - 26), (w, h), (12, 20, 28), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.putText(frame, HEADER_TEXT, (18, 28), cv2.FONT_HERSHEY_DUPLEX, 0.68, (245, 245, 245), 1)
    footer = f"FPS:{fps:.1f} | Frame:{frame_id} | Modele:{model_name} | Raspberry Pi 5"
    cv2.putText(frame, footer, (18, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
    return frame


def draw_person_rois(frame, rois):
    h, w = frame.shape[:2]
    for name, rel_box in rois:
        x1 = int(max(0, min(1, rel_box[0])) * w)
        y1 = int(max(0, min(1, rel_box[1])) * h)
        x2 = int(max(0, min(1, rel_box[2])) * w)
        y2 = int(max(0, min(1, rel_box[3])) * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
        cv2.putText(frame, name, (x1 + 4, max(58, y1 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
    return frame


def tracks_to_person_detections(tracks):
    return [
        {"bbox": track.bbox, "class_name": "person", "conf": track.conf}
        for track in tracks
        if track.class_name == "person"
    ]


def person_detections(detections):
    return [
        det
        for det in detections
        if det.get("class_name") == "person"
    ]


def count_detections_by_class(detections):
    counts = {}
    for det in detections:
        class_name = det.get("class_name", "unknown")
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts


def count_tracks_by_class(tracks):
    counts = {}
    for track in tracks:
        counts[track.class_name] = counts.get(track.class_name, 0) + 1
    return counts


def draw_tracking_boxes(frame, tracks, colors):
    """Dessine les tracks confirmes sans ajouter de logique MDP sur la video."""
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.bbox)
        color = colors.get(track.class_name, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"#{track.id} {track.class_name} {track.conf:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
        )
    return frame


def draw_pipeline_overlay(frame, fps, frame_id, anonymized, total_anonymized, tracks_count):
    """Overlay minimal: detection/anonymisation/tracking seulement."""
    lines = [
        f"FPS: {fps:.1f}",
        f"Frame: {frame_id}",
        f"Tracks: {tracks_count}",
        f"Anon/frame: {anonymized}",
        f"Anon total: {total_anonymized}",
    ]
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (250, 126), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (18, 34 + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
        )
    return frame


def draw_analysis_frame(frame, zone_tracker, state, decision, tracks):
    """
    Vue d'analyse officielle pour le dashboard.

    Elle affiche la preuve visuelle de l'IA: polygones officiels, detections
    et image anonymisee. Les scores et la decision MDP restent dans l'UI.
    """
    canvas = frame.copy()
    zones = getattr(zone_tracker, "zones", {}) or {}
    if zones:
        zone_overlay = canvas.copy()
        for direction, polygon in zones.items():
            pts = np.asarray(polygon, dtype=np.int32)
            color = ANALYSIS_ZONE_COLORS.get(direction, (255, 255, 255))
            cv2.fillPoly(zone_overlay, [pts], color)
        cv2.addWeighted(zone_overlay, 0.16, canvas, 0.84, 0, canvas)
        for direction, polygon in zones.items():
            pts = np.asarray(polygon, dtype=np.int32)
            color = ANALYSIS_ZONE_COLORS.get(direction, (255, 255, 255))
            cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)

    for track in tracks:
        x1, y1, x2, y2 = map(int, track.bbox)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        color = CLASS_COLORS.get(track.class_name, (255, 255, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"#{track.id} {track.class_name} {track.conf:.2f}"
        label_y = max(18, y1 - 6)
        cv2.putText(
            canvas,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
        )
        cv2.circle(canvas, (cx, cy), 5, color, -1)

    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (10, 18, 28), -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
    anonymized = int(state.get("anonymized_total", state.get("anonymized", 0)))
    text = (
        f"IMAGE ANALYSEE | Detections YOLO + anonymisation | "
        f"{len(tracks)} track(s) | {anonymized} personne(s) protegee(s)"
    )
    cv2.putText(canvas, text, (16, 31), cv2.FONT_HERSHEY_DUPLEX, 0.56, (255, 255, 255), 1)
    return canvas


class SmartTrafficPipeline:
    """Pipeline complet de gestion intelligente du trafic urbain."""

    def __init__(self, args):
        self.args = args
        self.active_scene_id = args.scene or (
            args.profile if args.profile in SCENE_PROFILES else DEFAULT_SCENE
        )
        if args.scene or (args.video is None and args.camera is None):
            self._apply_scene_args(args, self.active_scene_id)
        self.profile = args.profile
        self.zones = load_zone_polygons(args.zones_json) if args.zones_json else ZONE_PROFILES[self.profile]
        self.person_rois = PERSON_ROI_PROFILES.get(self.profile, [])
        self.model_name = Path(args.model).name
        self.frame_id = 0
        self.processed_frame_count = 0
        self.results = []
        self.decision_history = []
        self.emergency_count = 0
        self.dropped_frames = 0
        self.run_start_time = None
        self.playback_clock_start = None

        print("\n" + "=" * 56)
        print("  Smart Traffic Agadir - Pipeline demarrage")
        print("=" * 56)

        class_conf = {}
        for class_name, arg_name in (
            ("bus", "bus_conf"),
            ("car", "car_conf"),
            ("emergency_vehicle", "emergency_conf"),
            ("motorcycle", "motorcycle_conf"),
            ("person", "person_conf"),
            ("truck", "truck_conf"),
        ):
            value = getattr(args, arg_name, None)
            if value is not None:
                class_conf[class_name] = value
        self.detector = YOLODetector(args.model, conf=args.conf, imgsz=args.imgsz, class_conf=class_conf)
        self.anonymizer = Anonymizer(face_model_path=args.face_model, method=args.anon_method)
        self.iou_tracker = IoUTracker(iou_thresh=0.3, max_misses=3, min_hits=1)
        self.zone_tracker = ZoneTracker(self.zones, VEHICLE_WEIGHTS)
        self.mdp = MDPDecision()

        self.dashboard = None
        self.dashboard_enabled = bool(args.dashboard and not args.no_dash)
        if self.dashboard_enabled:
            self.dashboard = Dashboard(port=args.port)
            self.dashboard.set_current_scene(self.active_scene_id)
            self.dashboard.start()

        self.source = self._resolve_source(args)
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la source: {self.source}")
        if self.dashboard is not None:
            self.dashboard.set_source_video(self.source)

        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.playback_clock_start = time.perf_counter()
        self.writer = self._make_writer(args.save)
        if self.dashboard is not None:
            self.dashboard.update_video_metadata(
                self._video_meta(current_second=0.0),
                model_info=self._model_info(),
            )

        print(f"\n  Source  : {self.source}")
        print(f"  FPS src : {self.src_fps:.1f}")
        print(f"  Scene   : {self.active_scene_id}")
        print(f"  Profil  : {self.profile}")
        if args.zones_json:
            print(f"  Zones   : {args.zones_json}")
        print(f"  ROI pers: {'oui' if args.person_roi else 'non'}")
        if args.person_roi:
            print(f"            {len(self.person_rois)} zone(s), toutes les {args.roi_every} frame(s)")
        print(f"  ROI veh : {'oui' if args.vehicle_roi else 'non'}")
        if args.vehicle_roi:
            print(f"            zones MDP, toutes les {args.vehicle_roi_every} frame(s)")
        print(f"  Anon.   : {args.anon_method}")
        print(f"  Display : {'oui' if args.show else 'non'}")
        print(f"  Dashbrd : {'oui (port ' + str(args.port) + ')' if self.dashboard_enabled else 'non'}")
        print(f"  Analyse : {args.analysis_mode}")
        print(f"  Lecture : {'timeline source FPS, frames sautees si retard' if args.realtime_playback else 'toutes les frames traitees'}")
        print(f"  Stride  : {args.vid_stride} frame(s) source par inference")
        if args.save:
            print(f"  Output  : {args.save}")

    @staticmethod
    def _apply_scene_args(args, scene_id):
        scene = SCENE_PROFILES.get(scene_id)
        if not scene:
            return
        args.video = scene.get("video") or args.video
        args.profile = scene.get("profile", scene_id)
        args.zones_json = scene.get("zones_json") or args.zones_json

    @staticmethod
    def _resolve_source(args):
        if args.camera is not None:
            return int(args.camera)
        if args.video is None:
            raise ValueError("Fournir --video chemin|0 ou --camera index")
        return int(args.video) if str(args.video).isdigit() else args.video

    def _make_writer(self, save_path):
        if not save_path:
            return None
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(save_path, fourcc, min(10.0, self.src_fps), (width, height))

    def _model_info(self):
        return {
            "name": self.model_name,
            "imgsz": self.args.imgsz,
            "profile": self.profile,
            "scene": self.active_scene_id,
            "scene_label": SCENE_PROFILES.get(self.active_scene_id, {}).get("label", self.active_scene_id),
            "zones_json": self.args.zones_json,
        }

    def _video_meta(self, current_second=None):
        if current_second is None:
            current_second = self.frame_id / max(self.src_fps, 1e-9)
        return {
            "fps": self.src_fps,
            "total_frames": self.total_frames,
            "current_second": current_second,
            "playback_mode": "source_video_review" if self.args.analysis_mode == "on_demand" else (
                "source_fps_timeline" if self.args.realtime_playback else "processed_all_frames"
            ),
            "dropped_frames": self.dropped_frames,
            "vid_stride": self.args.vid_stride,
            "processed_frames": self.processed_frame_count,
            "scene": self.active_scene_id,
        }

    def _load_scene(self, scene_id):
        scene = SCENE_PROFILES.get(scene_id)
        if not scene:
            print(f"  Scene inconnue: {scene_id}")
            return False

        source = scene.get("video")
        if not source or not Path(source).exists():
            print(f"  Scene {scene_id}: video introuvable -> {source}")
            return False

        profile = scene.get("profile", scene_id)
        zones_json = scene.get("zones_json")
        zones = (
            load_zone_polygons(zones_json)
            if zones_json and Path(zones_json).exists()
            else ZONE_PROFILES[profile]
        )

        if getattr(self, "cap", None) is not None:
            self.cap.release()

        self.active_scene_id = scene_id
        self.profile = profile
        self.args.profile = profile
        self.args.video = source
        self.args.zones_json = zones_json
        self.source = source
        self.zones = zones
        self.person_rois = PERSON_ROI_PROFILES.get(profile, [])
        self.iou_tracker.reset()
        self.zone_tracker = ZoneTracker(self.zones, VEHICLE_WEIGHTS)
        self.mdp = MDPDecision()
        self.decision_history = []
        self.emergency_count = 0
        self.frame_id = 0
        self.processed_frame_count = 0
        self.dropped_frames = 0
        self.anonymizer.reset_counter()

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la source: {self.source}")

        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.playback_clock_start = time.perf_counter()

        if self.dashboard is not None:
            self.dashboard.set_current_scene(scene_id)
            self.dashboard.set_source_video(self.source)
            self.dashboard.update_video_metadata(
                self._video_meta(current_second=0.0),
                model_info=self._model_info(),
            )

        print(f"  Scene dashboard -> {scene_id} ({self.source})")
        return True

    def _handle_dashboard_scene_change(self):
        if not (self.dashboard_enabled and self.dashboard is not None):
            return False

        request_data = self.dashboard.pop_scene_request()
        if not request_data:
            return False

        scene_id = request_data.get("scene")
        if not scene_id:
            return False

        if not self._load_scene(scene_id):
            return False

        target_pos = int(SCENE_PROFILES.get(scene_id, {}).get("frame_index", 0) or 0)
        if self.total_frames > 0:
            target_pos = min(target_pos, self.total_frames - 1)
        return self.analyze_selected_frame(max(0, target_pos))

    def _seek_target_position(self, seek):
        target_frame = seek.get("frame")
        target_second = seek.get("second")
        try:
            if target_frame is not None:
                target_pos = int(float(target_frame)) - 1
            elif target_second is not None:
                target_pos = int(float(target_second) * self.src_fps)
            else:
                return None
        except (TypeError, ValueError):
            return None

        target_pos = max(0, target_pos)
        if self.total_frames > 0:
            target_pos = min(target_pos, self.total_frames - 1)
        return target_pos

    def _handle_dashboard_seek(self):
        if not (self.dashboard_enabled and self.dashboard is not None):
            return False

        seek = self.dashboard.pop_seek_request()
        if not seek:
            return False

        target_pos = self._seek_target_position(seek)
        if target_pos is None:
            return False

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
        self.frame_id = target_pos
        self.iou_tracker.reset()
        self._reset_realtime_clock()
        print(f"  Seek dashboard -> frame {target_pos + 1} ({target_pos / max(self.src_fps, 1e-9):.2f}s)")
        return True

    def _reset_realtime_clock(self):
        if self.playback_clock_start is None:
            return
        self.playback_clock_start = time.perf_counter() - (self.frame_id / max(self.src_fps, 1e-9))

    def _sync_realtime_playback(self):
        """Synchronise une video fichier au temps reel en sautant les frames en retard."""
        if not self.args.realtime_playback or isinstance(self.source, int):
            return
        if self.playback_clock_start is None:
            return

        elapsed = time.perf_counter() - self.playback_clock_start
        target_frame_id = int(elapsed * self.src_fps) + 1
        if self.total_frames > 0:
            target_frame_id = min(target_frame_id, self.total_frames)
        if self.args.max_frames:
            target_frame_id = min(target_frame_id, self.args.max_frames)

        next_frame_id = self.frame_id + 1
        if target_frame_id <= next_frame_id:
            return

        skipped = target_frame_id - next_frame_id
        self.dropped_frames += skipped
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_id - 1)
        self.frame_id = target_frame_id - 1

    def _skip_vid_stride_frames(self):
        """Saute les frames source non inferées quand --vid-stride > 1."""
        stride = int(max(1, self.args.vid_stride))
        if stride <= 1:
            return 0

        frames_to_skip = stride - 1
        if self.args.max_frames:
            frames_to_skip = min(frames_to_skip, max(0, self.args.max_frames - self.frame_id))
        skipped = 0
        for _ in range(frames_to_skip):
            if not self.cap.grab():
                break
            skipped += 1
        self.frame_id += skipped
        self.dropped_frames += skipped
        return skipped

    def _zone_bbox_rois(self, frame):
        """Construit des ROI rectangulaires autour des polygones MDP."""
        h, w = frame.shape[:2]
        rois = []
        padding = max(0, int(self.args.vehicle_roi_padding))
        for zone_name, polygon in self.zones.items():
            x, y, box_w, box_h = cv2.boundingRect(polygon.astype(np.int32))
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + box_w + padding)
            y2 = min(h, y + box_h + padding)
            if x2 > x1 and y2 > y1:
                rois.append((zone_name, (x1, y1, x2, y2)))
        return rois

    def process_frame(self, frame):
        t0 = time.perf_counter()

        detections = self.detector.detect(frame)
        if self.args.person_roi:
            detections = self.detector.detect_roi(
                frame,
                self.person_rois,
                every_n=self.args.roi_every,
                base_detections=detections,
                frame_id=self.frame_id,
                class_names=("person",),
                source_prefix="person_roi",
            )
        if self.args.vehicle_roi:
            detections = self.detector.detect_roi(
                frame,
                self._zone_bbox_rois(frame),
                every_n=self.args.vehicle_roi_every,
                base_detections=detections,
                frame_id=self.frame_id,
                class_names=VEHICLE_ROI_CLASSES,
                source_prefix="vehicle_roi",
            )

        frame, anonymized = self.anonymizer.anonymize(frame, person_detections(detections))
        tracks = self.iou_tracker.update(detections)

        state = self.zone_tracker.update(tracks)
        state["frame_id"] = self.frame_id
        state["anonymized"] = anonymized
        state["anonymized_total"] = self.anonymizer.total_anonymized
        state["detections_total"] = len(detections)
        state["detections_by_class"] = count_detections_by_class(detections)
        state["active_tracks"] = len(tracks)
        state["tracks_by_class"] = count_tracks_by_class(tracks)
        state["frame_size"] = {"width": frame.shape[1], "height": frame.shape[0]}
        state["polygons"] = {
            direction: polygon.astype(int).tolist()
            for direction, polygon in self.zones.items()
        }
        state["model_info"] = self._model_info()

        decision = self.mdp.decide(state)
        if decision["phase"] == "EMERGENCY":
            self.emergency_count += 1
        self.decision_history.append(decision)

        fps = 1.0 / max(time.perf_counter() - t0, 1e-9)
        return detections, tracks, frame, state, decision, anonymized, fps

    def render(self, frame, detections, tracks, state, decision, anonymized, fps):
        frame = draw_header_footer(frame, fps, self.frame_id, self.model_name)
        if self.args.person_roi and getattr(self.args, "draw_person_roi", False):
            frame = draw_person_rois(frame, self.person_rois)
        frame = draw_tracking_boxes(frame, tracks, CLASS_COLORS)
        frame = draw_pipeline_overlay(
            frame,
            fps,
            self.frame_id,
            anonymized,
            self.anonymizer.total_anonymized,
            len(tracks),
        )
        frame = self.anonymizer.draw_privacy_overlay(frame, anonymized)
        if self.args.draw_zones:
            frame = self.zone_tracker.draw_zones(frame, state, decision["phase"])
        if self.args.draw_mdp_overlay:
            frame = self.mdp.draw_decision(frame, decision, self.anonymizer.total_anonymized)
        return frame

    def analyze_selected_frame(self, target_pos):
        """Analyse une seule frame choisie dans le lecteur source."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
        ok, frame = self.cap.read()
        if not ok:
            print(f"  Analyse impossible: frame {target_pos + 1} non lisible.")
            return False

        self.frame_id = target_pos + 1
        self.iou_tracker.reset()
        detections, tracks, processed, state, decision, anonymized, fps = self.process_frame(frame)
        rendered = self.render(
            processed.copy(),
            detections,
            tracks,
            state,
            decision,
            anonymized,
            fps,
        )
        analysis_frame = draw_analysis_frame(
            processed.copy(),
            self.zone_tracker,
            state,
            decision,
            tracks,
        )
        if self.dashboard is not None:
            self.dashboard.update(
                rendered,
                state,
                decision,
                fps,
                analysis_frame=analysis_frame,
                video_meta=self._video_meta(current_second=target_pos / max(self.src_fps, 1e-9)),
            )
        self._record_frame(fps, state, decision, tracks, anonymized)
        print(
            f"  Analyse F{self.frame_id:05d} | FPS_IA={fps:.1f} | "
            f"Phase={decision['phase']} | Tracks={len(tracks)} | Anon={self.anonymizer.total_anonymized}"
        )
        return True

    def run_dashboard_review(self):
        print("\n  Dashboard review demarre - lecteur source + analyse a la demande\n")
        try:
            while True:
                if self._handle_dashboard_scene_change():
                    continue
                seek = self.dashboard.pop_seek_request() if self.dashboard is not None else None
                if seek:
                    target_pos = self._seek_target_position(seek)
                    if target_pos is not None:
                        self.analyze_selected_frame(target_pos)
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n  Interruption clavier.")
        finally:
            self.cleanup()

    def run(self):
        if self.dashboard_enabled and self.args.analysis_mode == "on_demand":
            self.run_dashboard_review()
            return

        print("\n  Pipeline demarree - [q] pour quitter\n")
        fps_window = deque(maxlen=30)
        self.run_start_time = time.perf_counter()

        try:
            while True:
                if self._handle_dashboard_scene_change():
                    fps_window.clear()
                    continue
                if self._handle_dashboard_seek():
                    fps_window.clear()

                if self.args.max_frames and self.frame_id >= self.args.max_frames:
                    print(f"  Limite de {self.args.max_frames} frames atteinte.")
                    break

                ok, frame = self.cap.read()
                if not ok:
                    print("  Fin du flux video.")
                    break

                self.frame_id += 1
                self.processed_frame_count += 1
                detections, tracks, processed, state, decision, anonymized, fps = self.process_frame(frame)
                fps_window.append(fps)
                avg_fps = float(np.mean(fps_window)) if fps_window else fps
                effective_fps = avg_fps * max(1, int(self.args.vid_stride))

                rendered = self.render(
                    processed.copy(),
                    detections,
                    tracks,
                    state,
                    decision,
                    anonymized,
                    avg_fps,
                )
                analysis_frame = None
                if (
                    self.dashboard_enabled
                    and self.dashboard is not None
                    and self.dashboard.needs_analysis_snapshot()
                ):
                    analysis_frame = draw_analysis_frame(
                        processed.copy(),
                        self.zone_tracker,
                        state,
                        decision,
                        tracks,
                    )

                if self.dashboard_enabled and self.dashboard is not None:
                    self.dashboard.update(
                        rendered,
                        state,
                        decision,
                        avg_fps,
                        analysis_frame=analysis_frame,
                        video_meta=self._video_meta(),
                    )

                if self.writer is not None:
                    self.writer.write(rendered)

                if self.args.show:
                    cv2.imshow("Smart Traffic Agadir", rendered)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("  Arret demande.")
                        break

                self._record_frame(avg_fps, state, decision, tracks, anonymized)

                if self.frame_id % 30 == 0:
                    fps_text = f"FPS={avg_fps:.1f}"
                    if self.args.vid_stride > 1:
                        fps_text += f" | FPS_effectif~{effective_fps:.1f}"
                    print(
                        f"  [F{self.frame_id:05d}] {fps_text} | "
                        f"Phase={decision['phase']} Duree={decision['duration']}s | "
                        f"Tracks={len(tracks)} | "
                        f"ROI(p/v)={self.detector.last_person_roi_count}/{self.detector.last_vehicle_roi_count} | "
                        f"Anon={self.anonymizer.total_anonymized}"
                    )

                if self.args.max_frames and self.frame_id >= self.args.max_frames:
                    print(f"  Limite de {self.args.max_frames} frames atteinte.")
                    break

                self._skip_vid_stride_frames()
                self._sync_realtime_playback()

        except KeyboardInterrupt:
            print("\n  Interruption clavier.")
        finally:
            self.cleanup()

    def _record_frame(self, fps, state, decision, tracks, anonymized):
        self.results.append(
            {
                "frame": self.frame_id,
                "scene": self.active_scene_id,
                "processed_frame": self.processed_frame_count,
                "fps": round(float(fps), 2),
                "effective_source_fps": round(float(fps) * max(1, int(self.args.vid_stride)), 2),
                "zones": {key: state.get(key, {}) for key in ("N", "S", "E", "W")},
                "pedestrians": state.get("pedestrians", 0),
                "emergency": state.get("emergency", False),
                "emergency_zone": state.get("emergency_zone"),
                "decision": decision,
                "tracks": len(tracks),
                "roi_person_detections": self.detector.last_person_roi_count,
                "roi_vehicle_detections": self.detector.last_vehicle_roi_count,
                "anonymized": anonymized,
            }
        )

    def cleanup(self):
        self.cap.release()
        if self.writer is not None:
            self.writer.release()
        if self.args.show:
            cv2.destroyAllWindows()
        if self.dashboard is not None:
            self.dashboard.stop()
        self._save_report()

    def _save_report(self):
        if not self.results:
            return
        os.makedirs(RESULTS_DIR, exist_ok=True)
        avg_fps = float(np.mean([row["fps"] for row in self.results]))
        elapsed = None
        avg_processed_fps = None
        avg_effective_source_fps = None
        if self.run_start_time is not None:
            elapsed = max(time.perf_counter() - self.run_start_time, 1e-9)
            avg_processed_fps = self.processed_frame_count / elapsed
            avg_effective_source_fps = self.frame_id / elapsed
        tracker_stats = self.iou_tracker.get_stats()
        zone_stats = self.zone_tracker.get_stats()
        report = {
            "total_frames": self.frame_id,
            "scene": self.active_scene_id,
            "source_frames_covered": self.frame_id,
            "processed_frames": self.processed_frame_count,
            "vid_stride": int(self.args.vid_stride),
            "avg_fps": round(avg_fps, 2),
            "avg_processed_fps": round(avg_processed_fps, 2) if avg_processed_fps else None,
            "avg_effective_source_fps": (
                round(avg_effective_source_fps, 2) if avg_effective_source_fps else None
            ),
            "elapsed_seconds": round(elapsed, 2) if elapsed else None,
            "total_anonymized": self.anonymizer.total_anonymized,
            "decisions_count": self.mdp.decision_count,
            "emergency_count": self.emergency_count,
            "zone_stats": zone_stats["zone_stats"],
            "model_info": {
                "path": self.args.model,
                "name": self.model_name,
                "imgsz": self.args.imgsz,
                "conf": self.args.conf,
                "profile": self.profile,
                "scene": self.active_scene_id,
            },
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "total_tracks": tracker_stats["total_tracks"],
            "frames": self.results,
        }
        out_path = Path(RESULTS_DIR) / "pipeline_results.json"
        out_path.write_text(json.dumps(report, indent=2))

        print(f"\n  Resultats -> {out_path}")
        print(f"  FPS moyen    : {avg_fps:.2f}")
        if avg_effective_source_fps and self.args.vid_stride > 1:
            print(f"  FPS effectif : {avg_effective_source_fps:.2f} frames source/s")
        print(f"  Frames src   : {self.frame_id}")
        print(f"  Frames IA    : {self.processed_frame_count}")
        print(f"  Anonymises   : {self.anonymizer.total_anonymized} detections")
        print(f"  Decisions    : {self.mdp.decision_count}")
        print(f"  Emergencies  : {self.emergency_count}")
        print(f"  Tracks total : {tracker_stats['total_tracks']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Smart Traffic Agadir - Pipeline")
    parser.add_argument("--model", type=str, default=YOLO_MODEL, help="Chemin ONNX")
    parser.add_argument("--video", type=str, default=None, help="Chemin video ou 0 webcam")
    parser.add_argument("--camera", type=int, default=None, help="Index webcam alternatif")
    parser.add_argument("--profile", choices=list(ZONE_PROFILES.keys()), default=DEFAULT_PROFILE)
    parser.add_argument("--scene", choices=list(SCENE_PROFILES.keys()), default=None, help="Scene dashboard preconfiguree")
    parser.add_argument("--zones-json", type=str, default=None, help="Polygones zones N/E/S/W generes par scripts/design_zone_polygons.py")
    parser.add_argument("--conf", type=float, default=CONF_THRESH)
    parser.add_argument(
        "--person-conf",
        type=float,
        default=None,
        help="Seuil de confiance specifique pour person, utile pour petits pietons",
    )
    parser.add_argument("--car-conf", type=float, default=None, help="Seuil specifique pour car")
    parser.add_argument("--truck-conf", type=float, default=None, help="Seuil specifique pour truck")
    parser.add_argument("--bus-conf", type=float, default=None, help="Seuil specifique pour bus")
    parser.add_argument("--motorcycle-conf", type=float, default=None, help="Seuil specifique pour motorcycle")
    parser.add_argument(
        "--emergency-conf",
        type=float,
        default=None,
        help="Seuil specifique pour emergency_vehicle",
    )
    parser.add_argument("--imgsz", "--img-size", dest="imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--show", action="store_true", help="Afficher fenetre OpenCV")
    parser.add_argument("--no-display", action="store_true", help="Alias compatibilite: force show=False")
    parser.add_argument("--save", "--output", dest="save", type=str, default=None, help="Sauvegarder video annotee")
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT)
    parser.add_argument("--dashboard", action="store_true", help="Activer dashboard Flask")
    parser.add_argument("--no-dash", action="store_true", help="Desactiver dashboard")
    parser.add_argument("--person-roi", action="store_true", help="Activer ROI person")
    parser.add_argument("--draw-person-roi", action="store_true", help="Debug: dessiner les ROI pietons sur la video")
    parser.add_argument("--roi-every", "--person-roi-every", dest="roi_every", type=int, default=PERSON_ROI_EVERY)
    parser.add_argument("--vehicle-roi", action="store_true", help="Activer ROI haute resolution sur les zones vehicules")
    parser.add_argument("--vehicle-roi-every", type=int, default=5, help="Frequence ROI vehicules")
    parser.add_argument("--vehicle-roi-padding", type=int, default=24, help="Padding pixels autour des zones vehicules")
    parser.add_argument("--anon-method", choices=Anonymizer.METHODS, default="gaussian")
    parser.add_argument("--face-model", type=str, default=None, help="Modele visage ONNX optionnel")
    parser.add_argument("--draw-zones", action="store_true", help="Debug: dessiner les polygones MDP sur la video")
    parser.add_argument("--draw-mdp-overlay", action="store_true", help="Debug: dessiner l'overlay MDP sur la video")
    parser.add_argument(
        "--analysis-mode",
        choices=["on_demand", "continuous"],
        default="on_demand",
        help="on_demand: analyser seulement l'instant choisi dans le dashboard; continuous: traiter le flux en continu",
    )
    parser.add_argument(
        "--realtime-playback",
        "--real-time",
        action="store_true",
        help="Respecter la timeline du FPS source; saute les frames que le Pi ne peut pas traiter",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limite benchmark")
    parser.add_argument(
        "--vid-stride",
        type=int,
        default=1,
        help="Traiter une frame source toutes les N frames; ex: 2 = une frame sur deux",
    )

    args = parser.parse_args()
    if args.no_display:
        args.show = False
    if args.video is None and args.camera is None and DEFAULT_SCENE not in SCENE_PROFILES:
        parser.error("fournir --video chemin|0, --camera index ou une scene valide")
    if args.roi_every < 1:
        parser.error("--roi-every doit etre >= 1")
    if args.vid_stride < 1:
        parser.error("--vid-stride doit etre >= 1")
    return args


if __name__ == "__main__":
    SmartTrafficPipeline(parse_args()).run()
