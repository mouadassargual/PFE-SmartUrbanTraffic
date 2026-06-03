"""
Détection YOLO ONNX pour Raspberry Pi 5.

Production: ONNX Runtime CPUExecutionProvider uniquement.
Formats supportés:
- post-NMS: (1, N, 6) ou (N, 6) avec x1,y1,x2,y2,conf,cls
- brut YOLO: (1, nc+4, anchors) ou (anchors, nc+4) avec cx,cy,w,h + scores
"""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

from pipeline.config import (
    CLASS_NAMES,
    CONF_THRESH,
    IMG_SIZE,
    IOU_THRESH,
    PERSON_ROI_DEDUP_IOU,
    YOLO_MODEL,
)


class YOLODetector:
    """Détecteur YOLO via ONNX Runtime CPU."""

    def __init__(
        self,
        model_path=None,
        conf=CONF_THRESH,
        imgsz=None,
        img_size=None,
        class_conf=None,
    ):
        self.model_path = model_path or YOLO_MODEL
        self.conf = conf
        self.class_conf = class_conf or {}
        requested_imgsz = int(imgsz or img_size or IMG_SIZE)
        self.imgsz = requested_imgsz
        self.img_size = self.imgsz
        self.classes = CLASS_NAMES
        self.frame_id = 0
        self.last_roi_count = 0
        self.last_person_roi_count = 0
        self.last_vehicle_roi_count = 0

        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.output_names = [output.name for output in self.session.get_outputs()]
        input_shape = input_meta.shape
        if len(input_shape) == 4:
            static_h, static_w = input_shape[2], input_shape[3]
            if isinstance(static_h, int) and isinstance(static_w, int):
                if static_h != static_w:
                    raise ValueError(
                        f"Modele ONNX non carre non supporte: {static_w}x{static_h}"
                    )
                if static_h != requested_imgsz:
                    print(
                        "⚠️  Taille ONNX statique detectee: "
                        f"{static_h}x{static_w}. "
                        f"--imgsz {requested_imgsz} ignore."
                    )
                self.imgsz = static_h
                self.img_size = self.imgsz

        print(f"✅ YOLODetector chargé : {self.model_path}")
        print(f"   Classes : {self.classes}")
        print(f"   Entrée  : {self.imgsz}x{self.imgsz}")
        if self.class_conf:
            print(f"   Seuils  : global={self.conf:.2f}, classes={self.class_conf}")

    def _threshold_for_class(self, cls_id):
        if 0 <= int(cls_id) < len(self.classes):
            cls_name = self.classes[int(cls_id)]
            return float(self.class_conf.get(cls_name, self.conf))
        return float(self.conf)

    def preprocess(self, frame):
        """Resize imgsz, BGR->RGB, normalisation, CHW + batch."""
        image = cv2.resize(frame, (self.imgsz, self.imgsz))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        return np.expand_dims(image, axis=0)

    def _scale_xyxy(self, box, orig_w, orig_h):
        scale_x = orig_w / self.imgsz
        scale_y = orig_h / self.imgsz
        x1, y1, x2, y2 = box
        return [
            int(np.clip(x1 * scale_x, 0, orig_w)),
            int(np.clip(y1 * scale_y, 0, orig_h)),
            int(np.clip(x2 * scale_x, 0, orig_w)),
            int(np.clip(y2 * scale_y, 0, orig_h)),
        ]

    def _postprocess_raw_yolo(self, preds, orig_h, orig_w):
        boxes = preds[:, :4]
        scores = preds[:, 4:]
        cls_ids = np.argmax(scores, axis=1)
        confs = np.max(scores, axis=1)

        thresholds = np.array(
            [self._threshold_for_class(cls_id) for cls_id in cls_ids],
            dtype=np.float32,
        )
        keep = confs >= thresholds
        boxes = boxes[keep]
        confs = confs[keep]
        cls_ids = cls_ids[keep]
        if len(boxes) == 0:
            return []

        scale_x = orig_w / self.imgsz
        scale_y = orig_h / self.imgsz
        x1 = np.clip((boxes[:, 0] - boxes[:, 2] / 2) * scale_x, 0, orig_w)
        y1 = np.clip((boxes[:, 1] - boxes[:, 3] / 2) * scale_y, 0, orig_h)
        x2 = np.clip((boxes[:, 0] + boxes[:, 2] / 2) * scale_x, 0, orig_w)
        y2 = np.clip((boxes[:, 1] + boxes[:, 3] / 2) * scale_y, 0, orig_h)

        bboxes_cv = [
            [int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])]
            for i in range(len(x1))
        ]
        nms_conf = min([self.conf, *self.class_conf.values()]) if self.class_conf else self.conf
        indices = cv2.dnn.NMSBoxes(bboxes_cv, confs.tolist(), nms_conf, IOU_THRESH)
        indices = np.array(indices).reshape(-1) if len(indices) else []
        if len(indices) == 0:
            return []

        detections = []
        for i in indices:
            cls_id = int(cls_ids[i])
            if 0 <= cls_id < len(self.classes):
                detections.append(
                    {
                        "bbox": [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
                        "conf": float(confs[i]),
                        "class_id": cls_id,
                        "class_name": self.classes[cls_id],
                    }
                )
        return detections

    def _postprocess_nms_free(self, preds, orig_h, orig_w):
        detections = []
        for det in preds:
            if len(det) < 6:
                continue
            x1, y1, x2, y2, conf, cls_id = det[:6]
            cls_id = int(cls_id)
            if not 0 <= cls_id < len(self.classes):
                continue
            if float(conf) < self._threshold_for_class(cls_id):
                continue
            detections.append(
                {
                    "bbox": self._scale_xyxy((x1, y1, x2, y2), orig_w, orig_h),
                    "conf": float(conf),
                    "class_id": cls_id,
                    "class_name": self.classes[cls_id],
                }
            )
        return detections

    def postprocess(self, outputs, orig_h, orig_w):
        """Détection automatique du format de sortie ONNX."""
        output = outputs[0]
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]

        nc4 = len(self.classes) + 4
        if output.ndim == 2 and output.shape[0] == nc4:
            return self._postprocess_raw_yolo(output.T, orig_h, orig_w)
        if output.ndim == 2 and output.shape[1] == nc4:
            return self._postprocess_raw_yolo(output, orig_h, orig_w)
        if output.ndim == 2 and output.shape[1] >= 6:
            return self._postprocess_nms_free(output, orig_h, orig_w)

        raise ValueError(f"Format sortie ONNX non supporté: {output.shape}")

    def _detect_frame(self, frame):
        """Inférence sans modifier le compteur de frame public."""
        orig_h, orig_w = frame.shape[:2]
        inp = self.preprocess(frame)
        outputs = self.session.run(self.output_names, {self.input_name: inp})
        return self.postprocess(outputs, orig_h, orig_w)

    def detect(self, frame):
        """Détection complète sur une frame BGR."""
        self.frame_id += 1
        return self._detect_frame(frame)

    @staticmethod
    def compute_iou(bbox1, bbox2):
        """Intersection over Union pour deux boxes [x1,y1,x2,y2]."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
        area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _roi_to_pixels(roi, width, height):
        """ROI relative 0..1 ou absolue pixels vers x1,y1,x2,y2."""
        if isinstance(roi, dict):
            roi = roi.get("box", roi.get("bbox"))
        x1, y1, x2, y2 = roi
        if max(x1, y1, x2, y2) <= 1.0:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        return (
            int(max(0, min(width - 1, x1))),
            int(max(0, min(height - 1, y1))),
            int(max(0, min(width, x2))),
            int(max(0, min(height, y2))),
        )

    def merge_detections(self, detections, iou_thresh=PERSON_ROI_DEDUP_IOU):
        """Déduplique les boxes de même classe en gardant la meilleure confiance."""
        kept = []
        for det in sorted(detections, key=lambda item: item["conf"], reverse=True):
            duplicate = False
            for prev in kept:
                if prev["class_name"] != det["class_name"]:
                    continue
                if self.compute_iou(prev["bbox"], det["bbox"]) > iou_thresh:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)
        return kept

    def detect_roi(
        self,
        frame,
        zones,
        every_n=5,
        base_detections=None,
        frame_id=None,
        class_names=("person",),
        source_prefix="roi",
    ):
        """
        Détection supplémentaire dans des zones ROI pour améliorer les objets petits.

        Si `base_detections` est fourni, retourne directement base + ROI dédupliqués.
        Sinon retourne uniquement les détections ROI.
        """
        current_frame = self.frame_id if frame_id is None else frame_id
        if every_n <= 0 or current_frame % every_n != 0:
            self.last_roi_count = 0
            if source_prefix == "person_roi":
                self.last_person_roi_count = 0
            if source_prefix == "vehicle_roi":
                self.last_vehicle_roi_count = 0
            return base_detections or []

        h, w = frame.shape[:2]
        roi_detections = []
        allowed_classes = set(class_names) if class_names is not None else None
        for zone in zones:
            if isinstance(zone, tuple) and len(zone) == 2:
                zone_name, roi = zone
            else:
                zone_name, roi = "person_roi", zone

            x1, y1, x2, y2 = self._roi_to_pixels(roi, w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            for det in self._detect_frame(crop):
                if allowed_classes is not None and det["class_name"] not in allowed_classes:
                    continue
                bx1, by1, bx2, by2 = det["bbox"]
                det["bbox"] = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
                det["source"] = f"{source_prefix}:{zone_name}"
                roi_detections.append(det)

        self.last_roi_count = len(roi_detections)
        if source_prefix == "person_roi":
            self.last_person_roi_count = len(roi_detections)
        if source_prefix == "vehicle_roi":
            self.last_vehicle_roi_count = len(roi_detections)
        if base_detections is None:
            return roi_detections
        return self.merge_detections(base_detections + roi_detections)

    def draw(self, frame, detections, colors):
        """Dessine rectangles et labels classe + confiance."""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class_name"]
            conf = det["conf"]
            color = colors.get(cls_name, (0, 255, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            y_label = max(15, y1 - 5)
            cv2.putText(frame, label, (x1, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame


if __name__ == "__main__":
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    print("YOLODetector module prêt. Fournir un modèle ONNX réel pour tester l'inférence.")
