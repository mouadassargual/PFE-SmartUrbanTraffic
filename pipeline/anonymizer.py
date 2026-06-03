"""
Anonymisation Privacy by Design.

Niveau 1: floutage des bounding boxes YOLO class==person.
Niveau 2 optionnel: détection visage MobileNet-SSD/ONNX si un modèle existe.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from pipeline.config import BLUR_KERNEL_SIZE, BLUR_SIGMA, FACE_CONF_THRESH, FACE_MODEL


class Anonymizer:
    """Anonymisation gaussienne, pixelate ou black-box pour les personnes."""

    METHODS = ("gaussian", "pixelate", "black")

    def __init__(
        self,
        face_model_path=None,
        method="gaussian",
        kernel_size=BLUR_KERNEL_SIZE,
        sigma=BLUR_SIGMA,
        pixel_size=15,
        face_conf=FACE_CONF_THRESH,
    ):
        # Compatibilité: Anonymizer("gaussian")
        if face_model_path in self.METHODS and method == "gaussian":
            method = face_model_path
            face_model_path = None

        if method not in self.METHODS:
            raise ValueError(f"Méthode '{method}' inconnue. Choisir parmi {self.METHODS}")

        self.method = method
        self.kernel_size = self._odd_kernel(kernel_size)
        self.sigma = sigma
        self.pixel_size = pixel_size
        self.face_conf = face_conf
        self.total_anonymized = 0
        self.face_net = None

        face_model_path = face_model_path if face_model_path is not None else FACE_MODEL
        if face_model_path and os.path.exists(face_model_path):
            try:
                self.face_net = cv2.dnn.readNetFromONNX(face_model_path)
                print(f"✅ Modèle visage chargé : {face_model_path}")
            except cv2.error as exc:
                print(f"⚠️ Modèle visage ignoré ({exc})")

        mode = "YOLO person + faces" if self.face_net is not None else "YOLO person only"
        print(f"✅ Anonymizer initialisé — méthode : {self.method} | {mode}")

    @staticmethod
    def _odd_kernel(kernel_size):
        if isinstance(kernel_size, int):
            kx = ky = kernel_size
        else:
            kx, ky = kernel_size
        if kx % 2 == 0:
            kx += 1
        if ky % 2 == 0:
            ky += 1
        return (max(3, kx), max(3, ky))

    @staticmethod
    def _clamp(frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w))
        y2 = max(0, min(int(y2), h))
        return x1, y1, x2, y2

    def blur_region(self, frame, x1, y1, x2, y2):
        """Gaussian blur kernel 51x51 sigma 30, avec clamp des coordonnées."""
        x1, y1, x2, y2 = self._clamp(frame, x1, y1, x2, y2)
        if x2 <= x1 or y2 <= y1:
            return frame
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, self.kernel_size, self.sigma)
        return frame

    def _pixelate_region(self, frame, x1, y1, x2, y2):
        x1, y1, x2, y2 = self._clamp(frame, x1, y1, x2, y2)
        if x2 <= x1 or y2 <= y1:
            return frame
        roi = frame[y1:y2, x1:x2]
        h, w = roi.shape[:2]
        if h < self.pixel_size or w < self.pixel_size:
            return self.blur_region(frame, x1, y1, x2, y2)
        small = cv2.resize(
            roi,
            (max(1, w // self.pixel_size), max(1, h // self.pixel_size)),
            interpolation=cv2.INTER_LINEAR,
        )
        frame[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return frame

    def _black_region(self, frame, x1, y1, x2, y2):
        x1, y1, x2, y2 = self._clamp(frame, x1, y1, x2, y2)
        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = 0
        return frame

    def anonymize_region(self, frame, x1, y1, x2, y2):
        """Anonymise une région selon la méthode choisie."""
        if self.method == "gaussian":
            return self.blur_region(frame, x1, y1, x2, y2)
        if self.method == "pixelate":
            return self._pixelate_region(frame, x1, y1, x2, y2)
        if self.method == "black":
            return self._black_region(frame, x1, y1, x2, y2)
        return frame

    def detect_faces(self, frame):
        """Détection visage optionnelle, robuste aux sorties MobileNet-SSD classiques."""
        if self.face_net is None:
            return []

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        self.face_net.setInput(blob)
        output = self.face_net.forward()
        preds = output.reshape(-1, output.shape[-1])

        faces = []
        for det in preds:
            if len(det) < 7:
                continue
            conf = float(det[2])
            if conf < self.face_conf:
                continue
            x1, y1, x2, y2 = det[3:7]
            faces.append([int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)])
        return faces

    def anonymize(self, frame, detections):
        """
        Floute les personnes YOLO puis les visages si le modèle visage est disponible.

        Returns:
            frame anonymisée, compteur d'anonymisations de la frame.
        """
        count = 0
        for det in detections:
            if det.get("class_name") != "person":
                continue
            x1, y1, x2, y2 = det["bbox"]
            self.anonymize_region(frame, x1, y1, x2, y2)
            count += 1

        for x1, y1, x2, y2 in self.detect_faces(frame):
            self.blur_region(frame, x1, y1, x2, y2)
            count += 1

        self.total_anonymized += count
        return frame, count

    def draw_privacy_overlay(self, frame, count):
        """Overlay magenta indiquant le nombre d'anonymisations."""
        h, w = frame.shape[:2]
        text = f"{count} personne(s) anonymisee(s)"
        overlay = frame.copy()
        x1 = max(10, w - 370)
        y1 = 54
        x2 = min(w - 10, x1 + 360)
        y2 = y1 + 40
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (180, 0, 180), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, "ANON", (x1 + 12, y1 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, text, (x1 + 96, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return frame

    def reset_counter(self):
        self.total_anonymized = 0

    def get_stats(self):
        return {
            "total_anonymized": self.total_anonymized,
            "method": self.method,
            "face_model_enabled": self.face_net is not None,
        }


if __name__ == "__main__":
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    dets = [{"bbox": [100, 100, 260, 500], "class_name": "person", "conf": 0.9}]
    anonymizer = Anonymizer(face_model_path=None)
    _, count = anonymizer.anonymize(frame, dets)
    print(f"Anonymisations test: {count}")
