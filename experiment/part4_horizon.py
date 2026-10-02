from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.mot_dataset import MOT17Sequence
from src.data.trajectory_dataset import TrajectoryDataset
from src.models.rnn_ablation import AblationMotionModel


def compute_gradient_norm_curve(model, sequence_tensor, max_k=16):
    """Calcula a norma analítica do gradiente (||dL_t / dh_{t-k}||) para avaliar o

    horizonte de memória e o gradiente que some (Parte 4).
    """
    model.eval()
    model.zero_grad()

    # Assegura dimensão de batch (1, seq_len, 4) e habilita gradiente na entrada se necessário
    if sequence_tensor.dim() == 2:
        x = sequence_tensor.unsqueeze(0)
    else:
        x = sequence_tensor

    hidden = None
    h_states = []
    outputs = []

    # Unroll passo a passo para reter gradientes dos estados ocultos h_t
    for t in range(x.size(1)):
        x_t = x[:, t : t + 1, :]
        out, hidden = model(x_t, hidden)
        outputs.append(out)

        # Trata tupla da LSTM (h, c) vs tensor simples de RNN/GRU
        h_t = hidden[0] if isinstance(hidden, tuple) else hidden
        h_t.retain_grad()
        h_states.append(h_t)

    # Perda escalar no último passo temporal (t = T - 1)
    loss = outputs[-1].sum()
    loss.backward()

    grad_norms = []
    T = len(h_states)

    for k in range(1, min(max_k + 1, T)):
        idx = T - k - 1
        if idx >= 0 and h_states[idx].grad is not None:
            norm = h_states[idx].grad.norm().item()
            grad_norms.append(norm)
        else:
            grad_norms.append(0.0)

    return grad_norms


def run_part4_analysis():
    # 1. Configuração de Caminhos
    root_dir = Path(__file__).resolve().parent.parent
    seq_test_path = root_dir / "data" / "MOT17" / "train" / "MOT17-11-SDP"
    output_dir = root_dir / "outputs" / "part4_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    rnn_ckpt = root_dir / "outputs" / "part3_experiments" / "RNN_T16_seed2026.pth"
    lstm_ckpt = (
        root_dir / "outputs" / "part3_experiments" / "LSTM_T16_seed2026.pth"
    )

    print(f"--- Processando Sequência Reservada (~10%): {seq_test_path.name} ---")

    # 2. Análise Empírica de Oclusão no Ground Truth (usando visibility)
    seq = MOT17Sequence(seq_test_path)
    gt_list = seq.load_gt()

    # Agrupa por ID e analisa visibilidade
    id_frames = {}
    for item in gt_list:
        obj_id = item["id"]
        if obj_id not in id_frames:
            id_frames[obj_id] = []
        id_frames[obj_id].append(item)

    occlusion_durations = []
    for obj_id, items in id_frames.items():
        sorted_items = sorted(items, key=lambda x: x["frame"])
        current_occ = 0
        for it in sorted_items:
            # Considera ocluído se visibility < 0.3
            if it.get("visibility", 1.0) < 0.3:
                current_occ += 1
            else:
                if current_occ > 0:
                    occlusion_durations.append(current_occ)
                    current_occ = 0
        if current_occ > 0:
            occlusion_durations.append(current_occ)

    if occlusion_durations:
        print(
            f"Eventos de Oclusão: {len(occlusion_durations)} | "
            f"Média: {np.mean(occlusion_durations):.2f} frames | "
            f"Máx: {np.max(occlusion_durations)} frames"
        )

    # 3. Extração de Amostra Real de Trajetória do Dataset
    traj_dataset = TrajectoryDataset(gt_list, seq_len=16)
    if len(traj_dataset) > 0:
        real_sample_x, _ = traj_dataset[0]  # Tensor (16, 4)
    else:
        real_sample_x = torch.randn(16, 4)

    # 4. Cálculo do Gradiente que Some (RNN Vanilla vs LSTM)
    rnn_model = AblationMotionModel(cell_type="RNN")
    lstm_model = AblationMotionModel(cell_type="LSTM")

    if rnn_ckpt.exists():
        rnn_model.load_state_dict(
            torch.load(rnn_ckpt, map_location=torch.device("cpu"))
        )
    if lstm_ckpt.exists():
        lstm_model.load_state_dict(
            torch.load(lstm_ckpt, map_location=torch.device("cpu"))
        )

    norms_rnn = compute_gradient_norm_curve(rnn_model, real_sample_x, max_k=15)
    norms_lstm = compute_gradient_norm_curve(
        lstm_model, real_sample_x, max_k=15
    )

    # 5. Plotagem e Salvamento do Artefato
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(norms_rnn) + 1),
        norms_rnn,
        label="RNN Vanilla",
        marker="o",
    )
    plt.plot(
        range(1, len(norms_lstm) + 1), norms_lstm, label="LSTM", marker="s"
    )
    plt.xlabel("Passos no passado (k)")
    plt.ylabel(r"Norma do Gradiente $\|\partial \mathcal{L}_t / \partial h_{t-k}\|$")
    plt.title("Parte 4: Curva de Desaparecimento do Gradiente (Analítica)")
    plt.grid(True)
    plt.legend()

    plot_path = output_dir / "curva_gradiente_desaparecimento.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gráfico salvo com sucesso em: {plot_path}")


if __name__ == "__main__":
    run_part4_analysis()