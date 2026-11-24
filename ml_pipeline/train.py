"""
Script d'entraînement (Transfer Learning / Fine-tuning) pour YOLOv8 (détection ou segmentation).

Exemple d'utilisation :
    python ml_pipeline/train.py --data ml_pipeline/dataset/data.yaml --model yolov8n.pt --epochs 6
"""

import argparse
from pathlib import Path
import shutil

import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent


def resolve_device(requested_device: str) -> str:
    """Sélectionne automatiquement CUDA si disponible lorsque `auto` est demandé."""
    if requested_device == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return requested_device


def export_training_artifacts(save_dir: Path, presentation_dir: Path) -> None:
    """Copie les courbes d'entraînement et graphiques clés dans `presentation_assets/`."""
    presentation_dir.mkdir(parents=True, exist_ok=True)

    # Correspondance entre noms générés par Ultralytics (v8.x) et noms normalisés
    artifact_mapping = {
        "results.png": ["results.png"],
        "F1_curve.png": ["BoxF1_curve.png", "F1_curve.png"],
        "P_curve.png": ["BoxP_curve.png", "P_curve.png"],
        "R_curve.png": ["BoxR_curve.png", "R_curve.png"],
        "PR_curve.png": ["BoxPR_curve.png", "PR_curve.png"],
        "confusion_matrix.png": ["confusion_matrix.png"],
        "confusion_matrix_normalized.png": ["confusion_matrix_normalized.png"],
        "val_batch0_pred.jpg": ["val_batch0_pred.jpg"],
        "val_batch0_labels.jpg": ["val_batch0_labels.jpg"],
    }

    copied = []
    for target_name, candidates in artifact_mapping.items():
        for candidate in candidates:
            src = save_dir / candidate
            if src.exists():
                shutil.copy2(src, presentation_dir / target_name)
                copied.append(target_name)
                break

    print(f"[INFO] Artefacts copiés dans {presentation_dir.resolve()} : {', '.join(copied)}")


def train_model(
    data_yaml: Path,
    model_weights: str = "yolov8n.pt",
    epochs: int = 6,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "auto",
    lr0: float = 0.01,
    patience: int = 20,
    project: Path = Path("runs/train"),
    name: str = "yolov8_finetune",
    presentation_dir: Path = Path("presentation_assets"),
    copy_best_to_root: bool = True,
) -> Path:
    """Lance le fine-tuning YOLOv8 et sauvegarde les poids et courbes d'apprentissage."""
    if not data_yaml.exists():
        raise FileNotFoundError(f"Fichier data.yaml introuvable : {data_yaml}")

    resolved_device = resolve_device(device)
    print(f"[INFO] Initialisation du modèle : {model_weights} sur périphérique '{resolved_device}'")
    model = YOLO(model_weights)

    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=resolved_device,
        lr0=lr0,
        patience=patience,
        project=str(project),
        name=name,
        plots=True,
        save=True,
    )

    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    export_training_artifacts(save_dir, presentation_dir)

    if copy_best_to_root and best_weights.exists():
        root_best = BASE_DIR / "best.pt"
        shutil.copy2(best_weights, root_best)
        print(f"[SUCCÈS] Poids fine-tunés copiés à la racine : {root_best.resolve()}")

    print(f"[SUCCÈS] Entraînement terminé. Dossier des résultats : {save_dir.resolve()}")
    if best_weights.exists():
        print(f"[SUCCÈS] Meilleurs poids sauvegardés : {best_weights.resolve()}")
    return best_weights


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tuning d'un modèle YOLOv8 (détection ou segmentation) sur dataset personnalisé."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Chemin vers le fichier data.yaml généré par prepare_dataset.py.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Poids pré-entraînés de départ (ex: yolov8n.pt, yolov8n-seg.pt).",
    )
    parser.add_argument("--epochs", type=int, default=6, help="Nombre d'époques (défaut: 6).")
    parser.add_argument("--imgsz", type=int, default=640, help="Taille d'entrée des images (défaut: 640).")
    parser.add_argument("--batch", type=int, default=8, help="Taille du batch (défaut: 8).")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Périphérique d'entraînement : 'auto', 'cpu', ou index GPU '0' (défaut: auto).",
    )
    parser.add_argument("--lr0", type=float, default=0.01, help="Taux d'apprentissage initial (défaut: 0.01).")
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Nombre d'époques sans amélioration avant arrêt anticipé (défaut: 20).",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/train"),
        help="Dossier parent des entraînements (défaut: runs/train).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolov8_finetune",
        help="Nom de l'expérience (défaut: yolov8_finetune).",
    )
    parser.add_argument(
        "--presentation-dir",
        type=Path,
        default=Path("presentation_assets"),
        help="Dossier où copier les graphiques d'entraînement (défaut: presentation_assets).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        data_yaml=args.data,
        model_weights=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        lr0=args.lr0,
        patience=args.patience,
        project=args.project,
        name=args.name,
        presentation_dir=args.presentation_dir,
    )
