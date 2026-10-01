import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class SyntheticObject:
    obj_id: int
    pos: np.ndarray      # [x, y]
    vel: np.ndarray      # [vx, vy]
    axes: Tuple[int, int] # (raio_maior, raio_menor)
    depth: int           # Camada de profundidade (quanto menor, mais próximo da câmera)
    color: Tuple[int, int, int]

class SyntheticVideoGenerator:
    def __init__(
        self,
        img_size: Tuple[int, int] = (128, 128),
        num_frames: int = 45,
        num_objects: int = 8,
        speed_scale: float = 2.0,
        occlusion_duration: int = 10
    ):
        self.img_size = img_size
        self.num_frames = num_frames
        self.num_objects = num_objects
        self.speed_scale = speed_scale
        self.occlusion_duration = occlusion_duration

    def generate(self) -> Tuple[np.ndarray, List[dict]]:
        """
        Retorna:
            frames: Array (N, H, W, 3) com as imagens do vídeo sintético
            gt_annotations: Lista de dicionários no formato ground-truth
        """
        frames = []
        gt_annotations = []
        
        # Inicializa objetos com posições e profundidades aleatórias
        objects = []
        for i in range(1, self.num_objects + 1):
            pos = np.random.uniform(20, self.img_size[0] - 20, size=2)
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0.5, 1.5) * self.speed_scale
            vel = np.array([np.cos(angle), np.sin(angle)]) * speed
            
            axes = (np.random.randint(8, 16), np.random.randint(6, 12))
            depth = np.random.randint(1, 100)
            color = tuple(map(int, np.random.randint(50, 255, size=3)))
            
            objects.append(SyntheticObject(i, pos, vel, axes, depth, color))

        for frame_idx in range(1, self.num_frames + 1):
            frame = np.zeros((*self.img_size, 3), dtype=np.uint8)
            
            # Ordena por profundidade (maior depth desenhado primeiro -> objetos com menor depth cobrem)
            objects.sort(key=lambda x: x.depth, reverse=True)
            
            for obj in objects:
                # Atualiza posição com rebate nas bordas
                obj.pos += obj.vel
                for d in range(2):
                    if obj.pos[d] < 10 or obj.pos[d] > self.img_size[d] - 10:
                        obj.vel[d] *= -1

                # Desenha elipse no quadro
                center = (int(obj.pos[0]), int(obj.pos[1]))
                cv2.ellipse(frame, center, obj.axes, 0, 0, 360, obj.color, -1)
                
                # Bounding box [bb_left, bb_top, bb_width, bb_height]
                w, h = obj.axes[0] * 2, obj.axes[1] * 2
                bb_left = max(0, obj.pos[0] - obj.axes[0])
                bb_top = max(0, obj.pos[1] - obj.axes[1])
                
                gt_annotations.append({
                    'frame': frame_idx,
                    'id': obj.obj_id,
                    'bb_left': bb_left,
                    'bb_top': bb_top,
                    'bb_width': w,
                    'bb_height': h,
                    'conf': 1,
                    'class': 1,
                    'visibility': 1.0
                })
                
            frames.append(frame)
            
        return np.array(frames), gt_annotations