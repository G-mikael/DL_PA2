import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple

def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Calcula IoU entre duas caixas no formato [bb_left, bb_top, width, height]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    denominator = float(boxAArea + boxBArea - interArea)
    if denominator == 0:
        return 0.0
    return interArea / denominator

class EvaluatorMOT:
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def evaluate(self, gt_list: List[dict], pred_list: List[dict]) -> Dict[str, float]:
        """
        Avalia as trajetórias preditas contra o Ground Truth.
        Formato das entradas: lista de dicts com keys 'frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height'
        """
        # Organiza por quadros
        frames = sorted(list(set([d['frame'] for d in gt_list] + [d['frame'] for d in pred_list])))
        
        gt_by_frame = {f: [] for f in frames}
        pred_by_frame = {f: [] for f in frames}
        
        for item in gt_list:
            gt_by_frame[item['frame']].append(item)
        for item in pred_list:
            pred_by_frame[item['frame']].append(item)

        gt_ids = sorted(list(set([d['id'] for d in gt_list if d['id'] != -1])))
        pred_ids = sorted(list(set([d['id'] for d in pred_list if d['id'] != -1])))
        
        gt_map = {gid: idx for idx, gid in enumerate(gt_ids)}
        pred_map = {pid: idx for idx, pid in enumerate(pred_ids)}

        # 1. CÁLCULO DE IDF1 (Atribuição Global 1-para-1)
        overlap_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=int)

        for f in frames:
            gts = gt_by_frame[f]
            preds = pred_by_frame[f]
            for g in gts:
                for p in preds:
                    box_g = (g['bb_left'], g['bb_top'], g['bb_width'], g['bb_height'])
                    box_p = (p['bb_left'], p['bb_top'], p['bb_width'], p['bb_height'])
                    if compute_iou(box_g, box_p) >= self.iou_threshold:
                        overlap_matrix[gt_map[g['id']], pred_map[p['id']]] += 1

        # Resolução do problema de casamento global (maximizar coincidencia)
        row_ind, col_ind = linear_sum_assignment(-overlap_matrix)
        
        idtp = 0
        for r, c in zip(row_ind, col_ind):
            idtp += overlap_matrix[r, c]

        len_gt_total = len(gt_list)
        len_pred_total = len(pred_list)

        idfp = len_pred_total - idtp
        idfn = len_gt_total - idtp

        idf1 = (2 * idtp) / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else 0.0


        # 2. CONTAGEM DE ID SWITCHES E FRAGMENTAÇÕES
        id_switches = 0
        fragmentations = 0

        # Mapeamento do último estado rastreado por ID do GT
        last_pred_for_gt = {}    # gt_id -> pred_id
        last_frame_for_gt = {}   # gt_id -> frame_idx
        gt_was_lost = {gid: False for gid in gt_ids}

        for f in frames:
            gts = gt_by_frame[f]
            preds = pred_by_frame[f]

            # Matching ganancioso por quadro para rastrear associações locais
            frame_matches = {} # gt_id -> pred_id
            matched_preds = set()

            for g in gts:
                best_iou = self.iou_threshold
                best_p_id = None
                box_g = (g['bb_left'], g['bb_top'], g['bb_width'], g['bb_height'])

                for p in preds:
                    if p['id'] in matched_preds:
                        continue
                    box_p = (p['bb_left'], p['bb_top'], p['bb_width'], p['bb_height'])
                    iou = compute_iou(box_g, box_p)
                    if iou > best_iou:
                        best_iou = iou
                        best_p_id = p['id']

                if best_p_id is not None:
                    frame_matches[g['id']] = best_p_id
                    matched_preds.add(best_p_id)

            # Análise de trocas e fragmentações
            for g in gts:
                gid = g['id']
                current_p_id = frame_matches.get(gid, None)

                if current_p_id is not None:
                    if gid in last_pred_for_gt:
                        # Casamento existia anteriormente
                        prev_p_id = last_pred_for_gt[gid]
                        if current_p_id != prev_p_id:
                            id_switches += 1
                        elif gt_was_lost[gid]:
                            fragmentations += 1
                    
                    last_pred_for_gt[gid] = current_p_id
                    last_frame_for_gt[gid] = f
                    gt_was_lost[gid] = False
                else:
                    if gid in last_pred_for_gt:
                        gt_was_lost[gid] = True

        return {
            'IDF1': idf1,
            'IDTP': idtp,
            'IDFP': idfp,
            'IDFN': idfn,
            'IDSW': id_switches,
            'Frag': fragmentations
        }