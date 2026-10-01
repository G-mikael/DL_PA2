import sys
from pathlib import Path

# pasta raiz do projeto (para importar módulos locais)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
OUTPUT_DIR = ROOT_DIR / "outputs" / "synthetic_experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
import copy
from src.synthetic.generator import SyntheticVideoGenerator
from src.synthetic.simulator import DetectorSimulator
from src.metrics.tracker_metrics import EvaluatorMOT
from src.tracking.naive_tracker import NaiveTracker



def run_experiment(num_objects=8, speed_scale=1.0, drop_prob=0.0, noise_std=0.0):
    """Executa o pipeline completo da Parte 0 para uma configuração específica."""
    # 1. Gerar Vídeo e GT
    generator = SyntheticVideoGenerator(
        img_size=(128, 128),
        num_frames=60,
        num_objects=num_objects,
        speed_scale=speed_scale
    )
    _, gt_annotations = generator.generate()

    # 2. Simular Resposta do Detector
    simulator = DetectorSimulator(
        drop_prob=drop_prob,
        noise_std=noise_std,
        fp_rate=0.01 if drop_prob > 0 else 0.0
    )
    detections = simulator.corrupt(gt_annotations)

    # 3. Rodar Rastreador Ingênuo (Frame a Frame)
    tracker = NaiveTracker(iou_threshold=0.2, max_age=3)
    frames = sorted(list(set([d['frame'] for d in detections])))
    
    pred_annotations = []
    for f in frames:
        frame_dets = [d for d in detections if d['frame'] == f]
        tracked_dets = tracker.update(frame_dets, f)
        pred_annotations.extend(tracked_dets)

    # 4. Avaliar com Métricas Próprias
    evaluator = EvaluatorMOT(iou_threshold=0.3)
    metrics = evaluator.evaluate(gt_annotations, pred_annotations)
    return metrics

def main():
    print("--- 1. TESTE DO PISO FÁCIL ---")
    easy_metrics = run_experiment(num_objects=3, speed_scale=0.5, drop_prob=0.0, noise_std=0.0)
    print(f"Piso Fácil -> IDF1: {easy_metrics['IDF1']:.4f} | IDSW: {easy_metrics['IDSW']} | Frag: {easy_metrics['Frag']}")
    
    # ENSAIO DE RUPTURA (Estresse de Velocidade e Oclusão/Ruído)

    print("\n--- 2. EXECUTANDO ENSAIO DE RUPTURA ---")
    speeds = np.linspace(0.5, 4.0, 8)
    drop_rates = [0.0, 0.1, 0.25]
    
    results = {drop: [] for drop in drop_rates}

    for drop in drop_rates:
        for spd in speeds:
            # Roda 5 repetições para estabilidade estatística
            idf1_acc = []
            for _ in range(5):
                m = run_experiment(num_objects=10, speed_scale=spd, drop_prob=drop, noise_std=spd * 0.5)
                idf1_acc.append(m['IDF1'])
            
            avg_idf1 = np.mean(idf1_acc)
            results[drop].append(avg_idf1)
            print(f"Drop Rate: {drop*100:.0f}% | Speed Scale: {spd:.1f} => IDF1 Médio: {avg_idf1:.3f}")

    # PLOTAGEM DO GRÁFICO

    plt.figure(figsize=(9, 5))
    for drop in drop_rates:
        plt.plot(speeds, results[drop], marker='o', linewidth=2, label=f'Descarte de Detecção (FN) = {int(drop*100)}%')
    
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Limiar Crítico (IDF1 = 0.90)')
    plt.title('Ensaio de Ruptura da Associação Ingênua (Parte 0)')
    plt.xlabel('Fator de Velocidade / Movimento dos Objetos')
    plt.ylabel('IDF1 Score')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ruptura_parte0.png', dpi=300)
    print("\n✅ Gráfico 'ruptura_parte0.png' gerado com sucesso!")

if __name__ == "__main__":
    main()