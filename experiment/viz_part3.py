import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
results_path = ROOT_DIR / "outputs" / "part3_experiments" / "ablation_results.json"
output_plot = ROOT_DIR / "outputs" / "part3_experiments" / "grafico_ablacao_small_multiples.png"

with open(results_path, "r") as f:
    results = json.load(f)

windows = [4, 8, 16, 32]
cells = ['RNN', 'GRU', 'LSTM']
colors = {'RNN': '#C44E52', 'GRU': '#55A868', 'LSTM': '#4C72B0'}

# Criando a figura com 3 subplots lado a lado, compartilhando o eixo Y
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle('Ablação - Eixo 1: Impacto do BPTT (Small Multiples)', fontsize=14, fontweight='bold')

for ax, cell in zip(axes, cells):
    means = []
    stds = []
    for w in windows:
        w_str = str(w)
        scores = results[cell][w_str]
        means.append(np.mean(scores))
        stds.append(np.std(scores))
        
    ax.errorbar(windows, means, yerr=stds, fmt='-o', 
                capsize=5, capthick=2, linewidth=2, 
                color=colors[cell])
    
    ax.set_title(f'Célula: {cell}')
    ax.set_xlabel('Tamanho da Janela BPTT')
    ax.set_xticks(windows)
    ax.grid(True, linestyle=':', alpha=0.7)

# Adiciona o rótulo do eixo Y apenas no primeiro gráfico
axes[0].set_ylabel('IDF1 Score (Média ± Desvio)')

plt.tight_layout()
plt.subplots_adjust(top=0.85) # Ajusta o espaço para o título geral
plt.savefig(output_plot, dpi=300)
print(f"✅ Gráfico Small Multiples salvo em: {output_plot}")