import copy
import numpy as np

class DetectorSimulator:
    def __init__(self, drop_prob: float = 0.1, noise_std: float = 2.0, fp_rate: float = 0.05):
        self.drop_prob = drop_prob     # Probabilidade de p% de descartar (FN)
        self.noise_std = noise_std     # Ruído gaussiano nas coordenadas
        self.fp_rate = fp_rate         # Injeção de Falsos Positivos

    def corrupt(self, gt_annotations: list, img_size=(128, 128)) -> list:
        detections = []
        
        for ann in gt_annotations:
            # Descarte de detecção (False Negative)
            if np.random.rand() < self.drop_prob:
                continue
                
            det = copy.deepcopy(ann)
            
            # Perturbação de coordenadas
            det['bb_left'] += np.random.normal(0, self.noise_std)
            det['bb_top'] += np.random.normal(0, self.noise_std)
            det['bb_width'] += np.random.normal(0, self.noise_std / 2)
            det['bb_height'] += np.random.normal(0, self.noise_std / 2)
            
            # Garante limites válidos
            det['bb_left'] = max(0, min(img_size[0], det['bb_left']))
            det['bb_top'] = max(0, min(img_size[1], det['bb_top']))
            det['id'] = -1  # O detector não sabe a identidade
            
            detections.append(det)

        # Injeção de Falsos Positivos (FP)
        if len(gt_annotations) > 0:
            max_frame = max(a['frame'] for a in gt_annotations)
            for f in range(1, max_frame + 1):
                if np.random.rand() < self.fp_rate:
                    detections.append({
                        'frame': f,
                        'id': -1,
                        'bb_left': np.random.uniform(0, img_size[0] - 20),
                        'bb_top': np.random.uniform(0, img_size[1] - 20),
                        'bb_width': np.random.uniform(10, 20),
                        'bb_height': np.random.uniform(10, 20),
                        'conf': np.random.uniform(0.3, 0.7),
                        'class': 1,
                        'visibility': 1.0
                    })
                    
        return detections