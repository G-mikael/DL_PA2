import numpy as np
import copy
from scipy.optimize import linear_sum_assignment
from typing import List, Dict
from src.metrics.tracker_metrics import compute_iou

class NaiveTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1
        self.tracks = {}  # track_id -> {'last_box': box, 'age': int, 'time_since_update': int}

    def update(self, detections: List[dict], frame_idx: int) -> List[dict]:
        """
        Recebe as detecções não-rotuladas de um único quadro e devolve
        as detecções associadas a um ID persistente.
        """
        det_boxes = [(d['bb_left'], d['bb_top'], d['bb_width'], d['bb_height']) for d in detections]
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]['last_box'] for tid in track_ids]

        assigned_dets = set()
        assigned_tracks = set()
        results = []

        if len(track_boxes) > 0 and len(det_boxes) > 0:
            # Matriz de Custo baseado em IoU (1 - IoU)
            cost_matrix = np.zeros((len(track_ids), len(det_boxes)))
            for i, t_box in enumerate(track_boxes):
                for j, d_box in enumerate(det_boxes):
                    cost_matrix[i, j] = 1.0 - compute_iou(t_box, d_box)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= (1.0 - self.iou_threshold):
                    tid = track_ids[r]
                    self.tracks[tid]['last_box'] = det_boxes[c]
                    self.tracks[tid]['time_since_update'] = 0
                    assigned_tracks.add(tid)
                    assigned_dets.add(c)

                    res = copy.deepcopy(detections[c])
                    res['id'] = tid
                    results.append(res)

        # 1. Novas tracks para detecções não associadas
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'last_box': det_boxes[j],
                    'time_since_update': 0
                }
                res = copy.deepcopy(det)
                res['id'] = tid
                results.append(res)

        # 2. Envelhecimento e remoção de tracks mortas
        dead_tracks = []
        for tid in self.tracks:
            if tid not in assigned_tracks:
                self.tracks[tid]['time_since_update'] += 1
                if self.tracks[tid]['time_since_update'] > self.max_age:
                    dead_tracks.append(tid)

        for tid in dead_tracks:
            del self.tracks[tid]

        return results