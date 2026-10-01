import os
import zipfile
import urllib.request
from pathlib import Path
import sys

# pasta raiz do projeto (para importar módulos locais)
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# URLs oficiais do MOT17
URL_LABELS = "https://motchallenge.net/data/MOT17Labels.zip"
URL_FULL = "https://motchallenge.net/data/MOT17.zip"

def download_and_extract(url: str, output_dir: Path):
    """Baixa um arquivo ZIP e extrai no diretório de destino se ainda não existir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_filename = output_dir / url.split("/")[-1]

    # Checkpoint simples para não rebaixar se o zip/dados já existirem
    if not zip_filename.exists():
        print(f"⬇️ Baixando {url}...")
        try:
            urllib.request.urlretrieve(url, zip_filename)
            print("✅ Download concluído!")
        except Exception as e:
            print(f"❌ Erro ao baixar os dados: {e}")
            return

    print(f"📦 Extraindo {zip_filename.name} em {output_dir}...")
    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("✅ Extração concluída com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao extrair os dados: {e}")

def prepare_mot17(full_dataset: bool = False):
    """
    Prepara o dataset MOT17.
    - full_dataset = False: Baixa apenas anotações/detecções (~10MB) -> Suficiente para Parte 1 a 4.
    - full_dataset = True: Baixa imagens completas (~5.5GB).
    """
    project_root = ROOT_DIR
    mot_dir = project_root / "data" / "MOT17"

    if full_dataset:
        print("🚀 Preparando o pacote COMPLETO do MOT17 (~5.5 GB)...")
        download_and_extract(URL_FULL, mot_dir)
    else:
        print("🚀 Preparando o pacote LEVE de anotações/detecções (~10 MB)...")
        download_and_extract(URL_LABELS, mot_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automação de Download do MOT17")
    parser.add_argument("--full", action="store_true", help="Baixa o pacote completo de 5.5GB com os quadros em PNG/JPG")
    args = parser.parse_args()

    prepare_mot17(full_dataset=args.full)