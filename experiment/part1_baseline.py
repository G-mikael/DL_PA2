import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# pasta raiz do projeto (para importar módulos locais)
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
OUTPUT_DIR = ROOT_DIR / "outputs" / "part1_experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from src.data.mot_dataset import MOT17Sequence
from src.tracking.naive_tracker import NaiveTracker
from src.metrics.tracker_metrics import EvaluatorMOT
from src.detection.map_metric import compute_frame_ap

def run_part1_experiment(seq_path: str, output_plot_path: str = OUTPUT_DIR / "part1_descolamento.png"):
    # 1. Carregar dados
    sequence = MOT17Sequence(seq_path, detector_name="SDP")
    gt_data = sequence.load_gt()
    det_data = sequence.load_public_detections()

    if not gt_data or not det_data:
        print(f"Erro ao carregar os dados em: {seq_path}")
        return

    frames = sorted(list(set([d['frame'] for d in gt_data] + [d['frame'] for d in det_data])))
    
    # 2. Executar Associação Ingênua e Medir mAP por Quadro
    tracker = NaiveTracker(iou_threshold=0.3, max_age=5)
    pred_data = []
    
    ap_per_frame = []
    gt_by_frame = {f: [d for d in gt_data if d['frame'] == f] for f in frames}
    det_by_frame = {f: [d for d in det_data if d['frame'] == f] for f in frames}

    for f in frames:
        current_dets = det_by_frame.get(f, [])
        current_gts = gt_by_frame.get(f, [])

        # Medir mAP espacial do detetor congelado neste quadro
        frame_ap = compute_frame_ap(current_gts, current_dets, iou_threshold=0.5)
        ap_per_frame.append(frame_ap)

        # Atualizar rastreador
        tracked_dets = tracker.update(current_dets, f)
        pred_data.extend(tracked_dets)

    # 3. Avaliar com Métricas de Rastreamento (IDF1, IDSW, etc)
    evaluator = EvaluatorMOT(iou_threshold=0.5)
    metrics = evaluator.evaluate(gt_data, pred_data)

    mAP_global = np.mean(ap_per_frame)
    num_gt_ids = len(set([d['id'] for d in gt_data if d['id'] != -1]))
    num_pred_ids = len(set([d['id'] for d in pred_data if d['id'] != -1]))
    ratio_ids = num_pred_ids / num_gt_ids if num_gt_ids > 0 else 0
    idsw_per_gt = metrics['IDSW'] / num_gt_ids if num_gt_ids > 0 else 0

    print("=" * 50)
    print(f"RESULTADOS DA PARTE 1 - Sequência: {sequence.seq_name}")
    print(f"mAP Médio Espacial: {mAP_global:.4f}")
    print(f"IDF1 Temporal:      {metrics['IDF1']:.4f}")
    print(f"Identidades GT:     {num_gt_ids} | Previstas: {num_pred_ids} (Razão: {ratio_ids:.2f})")
    print(f"ID Switches Total:  {metrics['IDSW']} ({idsw_per_gt:.2f} por identidade)")
    print("=" * 50)

    # 4. Plotar o Gráfico de Descolamento (Dois Painéis)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Painel Superior: mAP vs IDF1
    ax1.plot(frames, ap_per_frame, label="mAP (Espacial)", color="blue", alpha=0.5)
    ax1.axhline(y=metrics['IDF1'], color="red", linestyle="--", linewidth=2, label=f"IDF1 Global ({metrics['IDF1']:.2f})")
    ax1.set_ylabel("Métrica Score")
    ax1.set_title(f"Gráfico do Descolamento - Parte 1 ({sequence.seq_name})")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Painel Inferior: Razão de Identidades e IDSWs
    ax2.axhline(y=ratio_ids, color="purple", linewidth=2, label=f"Razão Pred/GT IDs ({ratio_ids:.2f}x)")
    ax2.axhline(y=idsw_per_gt, color="orange", linestyle="-.", linewidth=2, label=f"IDSW por GT ID ({idsw_per_gt:.2f})")
    ax2.set_xlabel("Quadro (Frame)")
    ax2.set_ylabel("Rácio / Contagem")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    print(f"Gráfico do descolamento guardado em: {output_plot_path}")

if __name__ == "__main__":
    seq_example = ROOT_DIR / "data" / "MOT17" / "train" / "MOT17-02-SDP"
    if os.path.exists(seq_example):
        run_part1_experiment(seq_example)
    else:
        print(f"Por favor, descarregue a sequência {seq_example} para rodar este ensaio.")