import numpy as np
import copy
import torch
from scipy.optimize import linear_sum_assignment
from src.metrics.tracker_metrics import compute_iou
from src.models.rnn_motion import RNNMotionModel

class RNNTracker:
    def __init__(self, model_path, iou_threshold=0.3, max_age=5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1
        self.tracks = {}
        
        # Carregar modelo PyTorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RNNMotionModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def update(self, detections, frame_idx):
        det_boxes = [(d['bb_left'], d['bb_top'], d['bb_width'], d['bb_height']) for d in detections]
        track_ids = list(self.tracks.keys())
        
        # 1. PREDIÇÃO: O modelo "adivinha" onde o objeto está agora
        pred_boxes = []
        for tid in track_ids:
            trk = self.tracks[tid]
            with torch.no_grad():
                # Preparar tensor: (batch=1, seq_len=1, 4)
                x_in = (torch.from_numpy(np.array([[trk["last_box"]]], dtype=np.float32)) / 1000.0)
                x_in = x_in.to(self.device).float()
                
                pred_out, new_hidden = self.model(x_in, trk['hidden'])
                
                pred_box_norm = pred_out[0, 0].cpu().numpy()
                pred_box = pred_box_norm * 1000.0 # Desnormalizar
                
            trk['pred_box'] = pred_box
            trk['next_hidden'] = new_hidden
            pred_boxes.append(pred_box)

        assigned_dets = set()
        assigned_tracks = set()
        results = []

        # 2. ASSOCIAÇÃO: Compara a PREDIÇÃO com a DETECÇÃO via Húngaro
        if len(track_ids) > 0 and len(det_boxes) > 0:
            cost_matrix = np.zeros((len(track_ids), len(det_boxes)))
            for i, p_box in enumerate(pred_boxes):
                for j, d_box in enumerate(det_boxes):
                    cost_matrix[i, j] = 1.0 - compute_iou(p_box, d_box)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= (1.0 - self.iou_threshold):
                    tid = track_ids[r]
                    # Atualiza com a observação real (Teacher Forcing em tempo de inferência)
                    self.tracks[tid]['last_box'] = det_boxes[c]
                    self.tracks[tid]['hidden'] = self.tracks[tid]['next_hidden']
                    self.tracks[tid]['time_since_update'] = 0
                    
                    assigned_tracks.add(tid)
                    assigned_dets.add(c)
                    
                    res = copy.deepcopy(detections[c])
                    res['id'] = tid
                    results.append(res)

        # 3. OCLUSÃO E MORTES
        dead_tracks = []
        for tid in track_ids:
            if tid not in assigned_tracks:
                # O objeto não foi detectado. Roda no escuro!
                # Usa a própria predição como "last_box" para o próximo passo
                self.tracks[tid]['last_box'] = self.tracks[tid]['pred_box']
                self.tracks[tid]['hidden'] = self.tracks[tid]['next_hidden']
                self.tracks[tid]['time_since_update'] += 1
                
                if self.tracks[tid]['time_since_update'] > self.max_age:
                    dead_tracks.append(tid)

        for tid in dead_tracks:
            del self.tracks[tid]

        # 4. NOVOS OBJETOS
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'last_box': det_boxes[j],
                    'hidden': None, # Estado oculto começa vazio
                    'time_since_update': 0
                }
                res = copy.deepcopy(det)
                res['id'] = tid
                results.append(res)

        return results