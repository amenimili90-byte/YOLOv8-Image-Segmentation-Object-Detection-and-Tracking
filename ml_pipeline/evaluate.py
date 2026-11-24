"""
Script d'évaluation quantitative et de benchmark de vitesse pour YOLOv8.

Génère dans `presentation_assets/` (et `metrics/`) :
- `confusion_matrix.png`, `PR_curve.png`, `F1_curve.png`, `P_curve.png`, `R_curve.png`
- `metrics_summary.json`, `metrics_summary.md` et `metrics_summary.csv`
- `fps_benchmark.csv` et `fps_benchmark.png`

Exemple d'utilisation :
    python ml_pipeline/evaluate.py --model best.pt --data ml_pipeline/dataset/data.yaml
"""

import argparse
import json
from pathlib import Path
import shutil
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


def copy_validation_plots(val_save_dir: Path, target_dirs: List[Path]) -> None:
    """Copie les courbes de validation générées par Ultralytics dans les dossiers cibles."""
    plot_mapping = {
        "confusion_matrix.png": ["confusion_matrix.png"],
        "confusion_matrix_normalized.png": ["confusion_matrix_normalized.png"],
        "PR_curve.png": ["BoxPR_curve.png", "PR_curve.png", "MaskPR_curve.png"],
        "F1_curve.png": ["BoxF1_curve.png", "F1_curve.png", "MaskF1_curve.png"],
        "P_curve.png": ["BoxP_curve.png", "P_curve.png", "MaskP_curve.png"],
        "R_curve.png": ["BoxR_curve.png", "R_curve.png", "MaskR_curve.png"],
        "val_batch0_pred.jpg": ["val_batch0_pred.jpg"],
    }

    for target_name, candidates in plot_mapping.items():
        for candidate in candidates:
            src = val_save_dir / candidate
            if src.exists():
                for out_dir in target_dirs:
                    shutil.copy2(src, out_dir / target_name)
                break


def extract_per_class_metrics(val_metrics) -> List[Dict]:
    """Extrait Précision, Rappel, mAP@50 et mAP@50-95 pour chaque classe évaluée."""
    per_class = []
    box_metrics = val_metrics.box
    class_indices = getattr(box_metrics, "ap_class_index", [])
    names = getattr(val_metrics, "names", {})

    for pos, cls_idx in enumerate(class_indices):
        cls_id = int(cls_idx)
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        p_val, r_val, ap50_val, ap_val = box_metrics.class_result(pos)
        per_class.append(
            {
                "class_id": cls_id,
                "class_name": cls_name,
                "precision": round(float(p_val), 4),
                "recall": round(float(r_val), 4),
                "mAP50": round(float(ap50_val), 4),
                "mAP50_95": round(float(ap_val), 4),
            }
        )
    return per_class


def benchmark_inference_speed(
    model_weights: str,
    imgsz: int = 640,
    num_runs: int = 30,
    warmup_runs: int = 5,
) -> List[Dict]:
    """Mesure la latence moyenne (ms) et le débit (FPS) sur CPU et GPU CUDA (si disponible)."""
    devices_to_test = ["cpu"]
    if torch.cuda.is_available():
        devices_to_test.append("cuda:0")

    dummy_image = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    benchmark_results = []

    for dev in devices_to_test:
        model = YOLO(model_weights)
        for _ in range(warmup_runs):
            model.predict(dummy_image, imgsz=imgsz, device=dev, verbose=False)

        if dev.startswith("cuda"):
            torch.cuda.synchronize()

        latencies_ms = []
        for _ in range(num_runs):
            t_start = time.perf_counter()
            model.predict(dummy_image, imgsz=imgsz, device=dev, verbose=False)
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t_start) * 1000.0)

        avg_latency = float(np.mean(latencies_ms))
        std_latency = float(np.std(latencies_ms))
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

        label = "GPU (CUDA)" if dev.startswith("cuda") else "CPU"
        benchmark_results.append(
            {
                "device": label,
                "imgsz": imgsz,
                "runs": num_runs,
                "avg_latency_ms": round(avg_latency, 2),
                "std_latency_ms": round(std_latency, 2),
                "fps": round(fps, 2),
            }
        )

    return benchmark_results


def save_benchmark_plot(benchmark_results: List[Dict], target_dirs: List[Path]) -> None:
    """Sauvegarde un graphique comparatif des FPS par périphérique."""
    if not benchmark_results:
        return

    devices = [row["device"] for row in benchmark_results]
    fps_values = [row["fps"] for row in benchmark_results]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(devices, fps_values, color=["#3b82f6", "#10b981"][: len(devices)])
    ax.set_title("Benchmark de vitesse d'inférence YOLOv8 (FPS)")
    ax.set_ylabel("Images par seconde (FPS)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, fps_val in zip(bars, fps_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{fps_val:.2f} FPS",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.tight_layout()
    for out_dir in target_dirs:
        fig.savefig(out_dir / "fps_benchmark.png", dpi=150)
    plt.close(fig)


def write_markdown_summary(summary: Dict, output_path: Path) -> None:
    """Génère un rapport Markdown structuré `metrics_summary.md`."""
    lines = [
        "# Résumé Quantitatif d'Évaluation YOLOv8",
        "",
        f"- **Modèle évalué :** `{summary['model']}`",
        f"- **Dataset de validation :** `{summary['data']}`",
        f"- **Résolution d'entrée :** `{summary['imgsz']}×{summary['imgsz']}`",
        "",
        "## 1. Métriques Globales de Détection",
        "",
        "| Métrique | Valeur |",
        "| :--- | :---: |",
        f"| **Précision moyenne (Precision)** | `{summary['global_metrics']['box_precision']:.4f}` |",
        f"| **Rappel moyen (Recall)** | `{summary['global_metrics']['box_recall']:.4f}` |",
        f"| **mAP@50** | `{summary['global_metrics']['box_mAP50']:.4f}` |",
        f"| **mAP@50-95** | `{summary['global_metrics']['box_mAP50_95']:.4f}` |",
        "",
        "## 2. Benchmark de Vitesse d'Inférence",
        "",
        "| Périphérique | Résolution | Latence moyenne (ms/image) | Écart-type (ms) | Débit (FPS) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for row in summary["speed_benchmark"]:
        lines.append(
            f"| {row['device']} | {row['imgsz']}×{row['imgsz']} | "
            f"{row['avg_latency_ms']:.2f} | ± {row['std_latency_ms']:.2f} | **{row['fps']:.2f}** |"
        )

    lines.extend(
        [
            "",
            "## 3. Précision et Rappel par Classe",
            "",
            "| ID | Classe | Precision | Recall | mAP@50 | mAP@50-95 |",
            "| :---: | :--- | :---: | :---: | :---: | :---: |",
        ]
    )

    for cls_row in summary["per_class_metrics"]:
        lines.append(
            f"| {cls_row['class_id']} | `{cls_row['class_name']}` | "
            f"{cls_row['precision']:.4f} | {cls_row['recall']:.4f} | "
            f"{cls_row['mAP50']:.4f} | {cls_row['mAP50_95']:.4f} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_model(
    model_weights: str,
    data_yaml: str,
    imgsz: int = 640,
    device: str = "auto",
    output_dir: Path = Path("presentation_assets"),
    metrics_dir: Path = Path("metrics"),
    benchmark_runs: int = 30,
) -> Dict:
    """Exécute `model.val()`, exporte les graphiques/métriques et lance le benchmark FPS."""
    target_dirs = [output_dir, metrics_dir]
    for d in target_dirs:
        d.mkdir(parents=True, exist_ok=True)

    resolved_device = ("0" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    print(f"[INFO] Évaluation de '{model_weights}' sur '{data_yaml}' (device={resolved_device})...")
    model = YOLO(model_weights)
    val_metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        device=resolved_device,
        plots=True,
    )

    val_save_dir = Path(val_metrics.save_dir)
    copy_validation_plots(val_save_dir, target_dirs)

    global_metrics = {
        "box_precision": round(float(val_metrics.box.mp), 4),
        "box_recall": round(float(val_metrics.box.mr), 4),
        "box_mAP50": round(float(val_metrics.box.map50), 4),
        "box_mAP50_95": round(float(val_metrics.box.map), 4),
    }
    if hasattr(val_metrics, "seg") and val_metrics.seg is not None:
        global_metrics["mask_precision"] = round(float(val_metrics.seg.mp), 4)
        global_metrics["mask_recall"] = round(float(val_metrics.seg.mr), 4)
        global_metrics["mask_mAP50"] = round(float(val_metrics.seg.map50), 4)
        global_metrics["mask_mAP50_95"] = round(float(val_metrics.seg.map), 4)

    per_class_metrics = extract_per_class_metrics(val_metrics)

    print("[INFO] Exécution du benchmark de vitesse...")
    benchmark_results = benchmark_inference_speed(
        model_weights=model_weights,
        imgsz=imgsz,
        num_runs=benchmark_runs,
    )
    save_benchmark_plot(benchmark_results, target_dirs)

    summary = {
        "model": str(model_weights),
        "data": str(data_yaml),
        "imgsz": imgsz,
        "global_metrics": global_metrics,
        "speed_benchmark": benchmark_results,
        "per_class_metrics": per_class_metrics,
    }

    for d in target_dirs:
        with open(d / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        write_markdown_summary(summary, d / "metrics_summary.md")
        pd.DataFrame([global_metrics]).to_csv(d / "metrics_summary.csv", index=False)
        pd.DataFrame(per_class_metrics).to_csv(d / "per_class_metrics.csv", index=False)
        pd.DataFrame(benchmark_results).to_csv(d / "fps_benchmark.csv", index=False)

    print(f"[SUCCÈS] Tous les artefacts ont été générés dans : {output_dir.resolve()}")
    print(f"  - Precision : {global_metrics['box_precision']}")
    print(f"  - Recall    : {global_metrics['box_recall']}")
    print(f"  - mAP@50    : {global_metrics['box_mAP50']}")
    print(f"  - mAP@50-95 : {global_metrics['box_mAP50_95']}")
    for row in benchmark_results:
        print(f"  - {row['device']} : {row['avg_latency_ms']} ms/image ({row['fps']} FPS)")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Évalue un modèle YOLOv8 avec model.val() et génère métriques + benchmark FPS."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best.pt",
        help="Chemin ou nom du modèle YOLOv8 à évaluer (défaut: best.pt).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="ml_pipeline/dataset/data.yaml",
        help="Fichier data.yaml du dataset de validation (défaut: ml_pipeline/dataset/data.yaml).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Taille d'image (défaut: 640).")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Périphérique d'évaluation : 'auto', 'cpu', ou '0' (défaut: auto).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("presentation_assets"),
        help="Dossier principal de sortie (défaut: presentation_assets).",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("metrics"),
        help="Dossier secondaire des métriques (défaut: metrics).",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=30,
        help="Nombre d'itérations pour le benchmark FPS (défaut: 30).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(
        model_weights=args.model,
        data_yaml=args.data,
        imgsz=args.imgsz,
        device=args.device,
        output_dir=args.output_dir,
        metrics_dir=args.metrics_dir,
        benchmark_runs=args.benchmark_runs,
    )
