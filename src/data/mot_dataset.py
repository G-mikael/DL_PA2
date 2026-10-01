import os
import pandas as pd
from typing import List, Dict

class MOT17Sequence:
    def __init__(self, seq_path: str, detector_name: str = "SDP"):
        """
        seq_path: Caminho para a pasta da sequência (ex: 'data/MOT17/train/MOT17-02-SDP')
        """
        self.seq_path = seq_path
        self.seq_name = os.path.basename(seq_path)
        self.detector_name = detector_name
        
        self.gt_path = os.path.join(seq_path, "gt", "gt.txt")
        self.det_path = os.path.join(seq_path, "det", "det.txt")

    def load_gt(self) -> List[dict]:
        """Carrega anotações Ground Truth com tratamento flexível de colunas."""
        if not os.path.exists(self.gt_path):
            return []
        
        df = pd.read_csv(self.gt_path, header=None)
        
        # Mapeamento base para as 6 primeiras colunas obrigatórias
        base_cols = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'visibility']
        num_cols = df.shape[1]
        
        if num_cols >= len(base_cols):
            df = df.iloc[:, :len(base_cols)]
            df.columns = base_cols
        else:
            # Caso tenha menos colunas, atribui dinamicamente até onde existir
            df.columns = base_cols[:num_cols]

        # Garantir presença dos campos esperados
        if 'class' in df.columns:
            df = df[(df['class'] == 1) & (df['id'] != -1)]
        else:
            df = df[df['id'] != -1]

        return df.to_dict('records')

    def load_public_detections(self) -> List[dict]:
        """Carrega as detecções públicas (det.txt) tratando variações de 7 a 10 colunas."""
        if not os.path.exists(self.det_path):
            return []
            
        df = pd.read_csv(self.det_path, header=None)
        num_cols = df.shape[1]

        # Nomes padrão para as colunas presentes
        standard_cols = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x1', 'x2', 'x3']
        df.columns = standard_cols[:num_cols]

        # Garantir atributos necessários para o pipeline do tracker
        df['id'] = -1
        df['class'] = 1
        if 'conf' not in df.columns:
            df['conf'] = 1.0

        return df.to_dict('records')