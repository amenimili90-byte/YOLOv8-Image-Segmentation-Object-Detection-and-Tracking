"""
Script de vérification, nettoyage et structuration d'un dataset au format YOLO.

Structure générée :
    output_dir/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

Exemple d'utilisation :
    python ml_pipeline/prepare_dataset.py --source-dir raw_dataset --output-dir dataset_yolo --classes person bicycle car
"""

import argparse
from pathlib import Path
import random
import shutil
from typing import List, Tuple

import cv2
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_yolo_label_file(label_path: Path, num_classes: int) -> Tuple[bool, str]:
    """
    Vérifie la validité syntaxique et géométrique d'un fichier d'annotation YOLO
    (détection 5 valeurs ou segmentation >= 7 valeurs normalisées dans [0, 1]).
    """
    if not label_path.exists():
        return False, "fichier d'annotation manquant"

    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return False, "fichier d'annotation vide"

    for line_idx, line in enumerate(content.splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue

        try:
            cls_id = int(parts[0])
        except ValueError:
            return False, f"ligne {line_idx}: class_id invalide ({parts[0]})"

        if cls_id < 0 or (num_classes > 0 and cls_id >= num_classes):
            return False, f"ligne {line_idx}: class_id hors bornes ({cls_id})"

        coords_count = len(parts) - 1
        # Détection: 4 coordonnées (cx, cy, w, h) ; Segmentation: polygones (>= 3 points -> 6+ valeurs paires)
        if coords_count != 4 and (coords_count < 6 or coords_count % 2 != 0):
            return False, f"ligne {line_idx}: nombre de coordonnées invalide ({coords_count})"

        try:
            coords = [float(val) for val in parts[1:]]
        except ValueError:
            return False, f"ligne {line_idx}: coordonnée non numérique"

        if any(val < 0.0 or val > 1.0 for val in coords):
            return False, f"ligne {line_idx}: coordonnées hors de l'intervalle [0, 1]"

        if coords_count == 4 and (coords[2] <= 0.0 or coords[3] <= 0.0):
            return False, f"ligne {line_idx}: largeur/hauteur nulle ou négative"

    return True, "ok"


def validate_image_file(image_path: Path) -> bool:
    """Vérifie qu'une image est lisible par OpenCV et non corrompue."""
    img = cv2.imread(str(image_path))
    return img is not None and img.size > 0


def discover_image_label_pairs(source_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Recherche les paires (image, label) dans `source_dir`.
    Supporte soit une arborescence `source_dir/images` + `source_dir/labels`,
    soit un dossier plat contenant images et `.txt` côte à côte.
    """
    images_subdir = source_dir / "images"
    labels_subdir = source_dir / "labels"

    search_images_root = images_subdir if images_subdir.is_dir() else source_dir
    search_labels_root = labels_subdir if labels_subdir.is_dir() else source_dir

    pairs = []
    for img_path in sorted(search_images_root.rglob("*")):
        if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
            rel_parent = img_path.relative_to(search_images_root).parent
            lbl_path = search_labels_root / rel_parent / f"{img_path.stem}.txt"
            pairs.append((img_path, lbl_path))
    return pairs


def prepare_yolo_dataset(
    source_dir: Path,
    output_dir: Path,
    class_names: List[str],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    """Nettoie, sépare en train/val et génère la structure YOLO et `data.yaml`."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Le dossier source n'existe pas : {source_dir}")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("`val_ratio` doit être strictement compris entre 0 et 1.")

    pairs = discover_image_label_pairs(source_dir)
    if not pairs:
        raise RuntimeError(f"Aucune image trouvée dans {source_dir}.")

    valid_pairs = []
    rejected = []
    for img_path, lbl_path in pairs:
        if not validate_image_file(img_path):
            rejected.append((img_path.name, "image illisible ou corrompue"))
            continue
        is_valid_lbl, reason = validate_yolo_label_file(lbl_path, len(class_names))
        if not is_valid_lbl:
            rejected.append((img_path.name, reason))
            continue
        valid_pairs.append((img_path, lbl_path))

    print(f"[INFO] Paires analysées : {len(pairs)} | Valides : {len(valid_pairs)} | Rejetées : {len(rejected)}")
    for filename, reason in rejected[:10]:
        print(f"  - Rejeté ({filename}) : {reason}")
    if len(rejected) > 10:
        print(f"  ... et {len(rejected) - 10} autres fichiers rejetés.")

    if not valid_pairs:
        raise RuntimeError("Aucune paire valide (image + label) n'a été conservée.")

    rng = random.Random(seed)
    rng.shuffle(valid_pairs)

    val_count = max(1, int(len(valid_pairs) * val_ratio)) if len(valid_pairs) > 1 else 0
    val_pairs = valid_pairs[:val_count]
    train_pairs = valid_pairs[val_count:]

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_dest_dir = output_dir / "images" / split_name
        lbl_dest_dir = output_dir / "labels" / split_name
        img_dest_dir.mkdir(parents=True, exist_ok=True)
        lbl_dest_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in split_pairs:
            shutil.copy2(img_path, img_dest_dir / img_path.name)
            shutil.copy2(lbl_path, lbl_dest_dir / lbl_path.name)

    data_yaml_path = output_dir / "data.yaml"
    data_config = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    with open(data_yaml_path, "w", encoding="utf-8") as yaml_file:
        yaml.safe_dump(data_config, yaml_file, sort_keys=False, allow_unicode=True)

    print(f"[SUCCÈS] Dataset prêt dans : {output_dir.resolve()}")
    print(f"  - Train : {len(train_pairs)} échantillons")
    print(f"  - Val   : {len(val_pairs)} échantillons")
    print(f"  - Config: {data_yaml_path.resolve()}")
    return data_yaml_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vérifie, nettoie et structure un dataset d'images/labels au format YOLO."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Dossier source contenant les images et annotations YOLO (.txt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_yolo"),
        help="Dossier de destination structuré pour YOLOv8 (défaut: dataset_yolo).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="Liste ordonnée des noms de classes (ex: --classes person bicycle car).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Proportion d'images allouée à la validation (défaut: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire pour la séparation train/val (défaut: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_yolo_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        class_names=args.classes,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
