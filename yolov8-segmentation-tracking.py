from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output_videos"
FAVICON_PATH = BASE_DIR / "favicon-yolo.ico"
BEST_MODEL_PATH = BASE_DIR / "best.pt"

# Palette BGR pour la visualisation des classes et trajectoires
COLORS = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]

AVAILABLE_MODELS = [
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
    "yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg",
]


def check_ffmpeg():
    """Retourne le chemin vers ffmpeg (PATH système, WinGet local ou imageio-ffmpeg)."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    # Vérifie le dossier d'installation WinGet sous Windows si le PATH n'est pas encore rechargé
    winget_dir = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if winget_dir.exists():
        winget_ffmpeg = next(winget_dir.glob("Gyan.FFmpeg*/**/ffmpeg.exe"), None)
        if winget_ffmpeg:
            return str(winget_ffmpeg)

    # Repli automatique sur le binaire statique embarqué par imageio-ffmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe and Path(ffmpeg_exe).exists():
            return ffmpeg_exe
    except Exception:
        pass

    return None


def show_ffmpeg_missing_warning():
    """Affiche un avertissement clair dans Streamlit si ffmpeg n'est pas installé."""
    st.error(
        "FFmpeg est requis pour encoder la vidéo annotée en H.264, mais `ffmpeg` est introuvable dans le `PATH`.\n\n"
        "Installez FFmpeg sur votre machine puis redémarrez l'application :\n"
        "- **Windows :** `winget install Gyan.FFmpeg`\n"
        "- **Linux (Ubuntu/Debian) :** `sudo apt update && sudo apt install -y ffmpeg`\n"
        "- **macOS :** `brew install ffmpeg`"
    )


@st.cache_resource
def load_model(model_weights: str) -> YOLO:
    """Charge et met en cache le modèle YOLOv8 afin d'éviter des rechargements répétés."""
    return YOLO(model_weights)


def reset_tracker_state(model: YOLO) -> None:
    """Réinitialise l'état interne du tracker ByteTrack entre deux flux vidéo distincts."""
    if getattr(model, "predictor", None) is not None:
        model.predictor = None


def result_to_json(result: Results, include_masks: bool = True):
    """Convertit un objet Results YOLOv8 (détection, segmentation, tracking) en liste JSON."""
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    track_ids = (
        result.boxes.id.int().cpu().numpy()
        if result.boxes.id is not None
        else None
    )

    result_list_json = []
    for idx in range(len(boxes_xyxy)):
        cls_id = int(classes[idx])
        entry = {
            "class_id": cls_id,
            "class": result.names[cls_id],
            "confidence": float(confidences[idx]),
            "bbox": {
                "x_min": int(boxes_xyxy[idx][0]),
                "y_min": int(boxes_xyxy[idx][1]),
                "x_max": int(boxes_xyxy[idx][2]),
                "y_max": int(boxes_xyxy[idx][3]),
            },
        }
        if track_ids is not None:
            entry["object_id"] = int(track_ids[idx])
        result_list_json.append(entry)

    if result.masks is not None:
        orig_h, orig_w = result.orig_shape[0], result.orig_shape[1]
        for idx, entry in enumerate(result_list_json):
            if idx < len(result.masks.xy):
                entry["segments"] = result.masks.xy[idx].tolist()
            if include_masks and idx < len(result.masks.data):
                resized_mask = cv2.resize(
                    result.masks.data[idx].cpu().numpy(),
                    (orig_w, orig_h),
                )
                entry["mask"] = resized_mask.tolist()

    return result_list_json


def draw_detections(
    result: Results,
    result_list_json,
    track_history=None,
):
    """Dessine masques, boîtes englobantes, IDs et trajectoires sur une copie de l'image."""
    alpha = 0.45
    image = result.orig_img.copy()
    orig_h, orig_w = image.shape[:2]

    # Superposition des masques directement depuis les tenseurs ou polygones si disponibles
    if result.masks is not None and len(result.masks.data) > 0:
        mask_overlay = image.copy()
        for idx, item in enumerate(result_list_json):
            if idx >= len(result.masks.data):
                break
            class_color = COLORS[item["class_id"] % len(COLORS)]
            mask_bin = cv2.resize(
                result.masks.data[idx].cpu().numpy(),
                (orig_w, orig_h),
            ) > 0.5
            mask_overlay[mask_bin] = class_color
        image = cv2.addWeighted(mask_overlay, alpha, image, 1 - alpha, 0)

    for item in result_list_json:
        class_color = COLORS[item["class_id"] % len(COLORS)]
        bbox = item["bbox"]
        x_min, y_min, x_max, y_max = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]

        label = (
            f"{item['class']} #{item['object_id']} ({item['confidence']:.2f})"
            if "object_id" in item
            else f"{item['class']} ({item['confidence']:.2f})"
        )

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), class_color, 2)
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y_top = max(0, y_min - text_h - baseline - 4)
        cv2.rectangle(image, (x_min, label_y_top), (x_min + text_w + 4, y_min), class_color, -1)
        cv2.putText(
            image,
            label,
            (x_min + 2, max(text_h + 2, y_min - baseline)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if "object_id" in item and track_history is not None:
            center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
            points = track_history[item["object_id"]]
            points.append(center)
            for j in range(1, len(points)):
                if points[j - 1] is None or points[j] is None:
                    continue
                thickness = max(1, int(np.sqrt(64 / float(j + 1)) * 1.5))
                cv2.line(image, points[j - 1], points[j], class_color, thickness)

    return image


def compute_class_counts_df(detections_list) -> pd.DataFrame:
    """Construit un tableau récapitulatif du nombre d'objets par classe."""
    counts = Counter(item["class"] for item in detections_list)
    if not counts:
        return pd.DataFrame(columns=["Classe", "Nombre d'objets"])
    return (
        pd.DataFrame(
            [{"Classe": cls_name, "Nombre d'objets": count} for cls_name, count in counts.items()]
        )
        .sort_values(by="Nombre d'objets", ascending=False)
        .reset_index(drop=True)
    )


def build_track_report_df(track_stats: dict) -> pd.DataFrame:
    """Construit le DataFrame CSV des pistes suivies (ID, classe, confiance moyenne, frames)."""
    if not track_stats:
        return pd.DataFrame(
            columns=["track_id", "class", "avg_confidence", "frames_present"]
        )
    rows = []
    for (track_id, cls_name), confs in track_stats.items():
        rows.append(
            {
                "track_id": track_id,
                "class": cls_name,
                "avg_confidence": round(float(np.mean(confs)), 4),
                "frames_present": len(confs),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(by=["frames_present", "track_id"], ascending=[False, True])
        .reset_index(drop=True)
    )


def image_processing(frame, model: YOLO, conf: float, device: str, include_masks: bool = True):
    """Exécute l'inférence YOLOv8 sur une image et mesure le temps d'inférence."""
    t_start = time.perf_counter()
    results = model.predict(frame, conf=conf, device=device, verbose=False)
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    result_list_json = result_to_json(results[0], include_masks=include_masks)
    result_image = draw_detections(results[0], result_list_json)
    return result_image, result_list_json, elapsed_ms


def video_processing(
    video_file: str,
    model: YOLO,
    model_name: str,
    conf: float,
    device: str,
    include_masks: bool = False,
    vid_stride: int = 1,
    progress_callback=None,
):
    """Traite une vidéo en mode streaming avec ByteTrack, génère MP4/JSON/CSV et mesure les FPS."""
    ffmpeg_executable = check_ffmpeg()
    if ffmpeg_executable is None:
        raise RuntimeError("L'exécutable ffmpeg est introuvable.")

    video_path = Path(video_file)
    video_stem = video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo source : {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or np.isnan(source_fps) or source_fps <= 0:
        source_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if frame_width <= 0 or frame_height <= 0:
        raise RuntimeError("Dimensions de la vidéo invalides (0x0).")

    out_fps = max(1.0, float(source_fps) / max(1, int(vid_stride)))
    expected_frames = max(1, (total_frames + vid_stride - 1) // vid_stride) if total_frames > 0 else None

    output_folder = OUTPUT_DIR / video_stem
    output_folder.mkdir(parents=True, exist_ok=True)

    video_file_name_out = output_folder / f"{video_stem}_{model_name}_output.mp4"
    video_file_name_out.unlink(missing_ok=True)

    result_video_json_file = output_folder / f"{video_stem}_{model_name}_output.json"
    result_video_json_file.unlink(missing_ok=True)

    reset_tracker_state(model)
    track_history = defaultdict(lambda: deque(maxlen=30))
    track_stats = defaultdict(list)
    unique_objects_by_class = defaultdict(set)
    total_detections_by_class = Counter()
    all_frames_json = []
    inference_times_ms = []

    results_stream = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        conf=conf,
        device=device,
        vid_stride=max(1, int(vid_stride)),
        verbose=False,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_file = Path(tmp_dir) / "temp_raw.mp4"
        video_writer = cv2.VideoWriter(
            str(temp_file),
            cv2.VideoWriter_fourcc(*"mp4v"),
            out_fps,
            (frame_width, frame_height),
        )
        if not video_writer.isOpened():
            raise RuntimeError("Impossible d'initialiser VideoWriter OpenCV pour temp_raw.mp4.")

        try:
            for frame_idx, result in enumerate(results_stream):
                t_start = time.perf_counter()
                result_list_json = result_to_json(result, include_masks=include_masks)
                annotated_frame = draw_detections(
                    result,
                    result_list_json,
                    track_history=track_history,
                )

                if hasattr(result, "speed") and isinstance(result.speed, dict) and "inference" in result.speed:
                    inference_times_ms.append(float(result.speed["inference"]))
                else:
                    inference_times_ms.append((time.perf_counter() - t_start) * 1000.0)

                video_writer.write(annotated_frame)
                all_frames_json.append(
                    {"frame": frame_idx, "detections": result_list_json}
                )

                for det in result_list_json:
                    cls_name = det["class"]
                    total_detections_by_class[cls_name] += 1
                    if "object_id" in det:
                        obj_id = det["object_id"]
                        track_stats[(obj_id, cls_name)].append(det["confidence"])
                        unique_objects_by_class[cls_name].add(obj_id)

                if progress_callback is not None:
                    progress_callback(frame_idx + 1, expected_frames)
        finally:
            video_writer.release()

        with open(result_video_json_file, "w", encoding="utf-8") as json_file:
            json.dump(all_frames_json, json_file, indent=2, ensure_ascii=False)

        subprocess.run(
            [
                ffmpeg_executable,
                "-y",
                "-i",
                str(temp_file),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video_file_name_out),
            ],
            check=True,
            capture_output=True,
        )

    avg_inference_ms = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0
    avg_fps = (1000.0 / avg_inference_ms) if avg_inference_ms > 0 else 0.0

    summary_rows = []
    all_classes = set(total_detections_by_class.keys()) | set(unique_objects_by_class.keys())
    for cls_name in sorted(all_classes):
        summary_rows.append(
            {
                "Classe": cls_name,
                "Objets uniques suivis": len(unique_objects_by_class[cls_name]),
                "Détections cumulées (frames)": total_detections_by_class[cls_name],
            }
        )
    summary_df = (
        pd.DataFrame(summary_rows).sort_values(by="Objets uniques suivis", ascending=False).reset_index(drop=True)
        if summary_rows
        else pd.DataFrame(columns=["Classe", "Objets uniques suivis", "Détections cumulées (frames)"])
    )
    report_csv_df = build_track_report_df(track_stats)

    return {
        "video_out": str(video_file_name_out),
        "json_out": str(result_video_json_file),
        "avg_inference_ms": avg_inference_ms,
        "avg_fps": avg_fps,
        "source_fps": float(source_fps),
        "summary_df": summary_df,
        "report_df": report_csv_df,
    }


# --- Configuration de l'interface Streamlit ---
st.set_page_config(
    page_title="YOLOv8 Détection, Segmentation & Tracking",
    layout="wide",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else None,
)

st.title("YOLOv8 — Détection, Segmentation d'Instances & Tracking (ByteTrack)")
st.caption("Application de Vision par Ordinateur — Analyse de scènes et suivi multi-objets en local")

with st.sidebar:
    st.header("Paramètres du modèle")
    model_options = (["best.pt"] if BEST_MODEL_PATH.exists() else []) + AVAILABLE_MODELS
    model_select = st.selectbox("Modèle YOLOv8", model_options, index=0)
    model_weights = str(BEST_MODEL_PATH) if model_select == "best.pt" else f"{model_select}.pt"
    model_tag = Path(model_select).stem

    cuda_available = torch.cuda.is_available()
    device_options = ["CPU", "CUDA (GPU)"] if cuda_available else ["CPU"]
    selected_device_label = st.selectbox("Périphérique d'inférence", device_options)
    if not cuda_available:
        st.info("Aucun GPU CUDA détecté : exécution sur CPU.")
    device = "cuda:0" if selected_device_label == "CUDA (GPU)" and cuda_available else "cpu"

    conf_threshold = st.slider(
        "Seuil de confiance (conf)",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
    )

    exclude_masks_json = st.checkbox(
        "Exclure les matrices `mask` du JSON exporté",
        value=True,
        help="Réduit fortement la taille du JSON exporté en conservant uniquement les polygones légers (`segments`).",
    )

    ffmpeg_path = check_ffmpeg()
    st.divider()
    st.subheader("Statut système")
    if ffmpeg_path:
        st.success("FFmpeg détecté dans le PATH.")
    else:
        st.warning("FFmpeg non détecté (requis pour l'export vidéo MP4).")

model = load_model(model_weights)

tab_image, tab_video = st.tabs(
    ["Image Processing", "Video Processing"]
)

# --- Onglet 1 : Traitement d'Image ---
with tab_image:
    st.header("Détection et Segmentation sur Image")
    image_file = st.file_uploader("Importer une image", type=["jpg", "jpeg", "png"])
    process_image_button = st.button("Analyser l'image", type="primary")

    if image_file is None and process_image_button:
        st.warning("Veuillez importer une image avant de lancer l'analyse.")

    if image_file is not None and process_image_button:
        raw_bytes = np.frombuffer(image_file.read(), np.uint8)
        img_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error("Impossible de décoder le fichier image.")
        else:
            annotated_bgr, detections_json, elapsed_ms = image_processing(
                img_bgr,
                model,
                conf=conf_threshold,
                device=device,
                include_masks=not exclude_masks_json,
            )
            fps_val = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0

            col_img, col_stats = st.columns([2, 1])
            with col_img:
                st.image(
                    cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                    caption=f"Résultat ({model_select})",
                    use_container_width=True,
                )

            with col_stats:
                st.subheader("Performances & Comptage")
                m1, m2 = st.columns(2)
                m1.metric("Temps d'inférence", f"{elapsed_ms:.1f} ms")
                m2.metric("Débit estimé", f"{fps_val:.1f} FPS")

                counts_df = compute_class_counts_df(detections_json)
                st.dataframe(counts_df, use_container_width=True, hide_index=True)

                # Exports directs
                ok_enc, img_encoded = cv2.imencode(".png", annotated_bgr)
                if ok_enc:
                    st.download_button(
                        label="Télécharger l'image annotée (PNG)",
                        data=img_encoded.tobytes(),
                        file_name=f"{Path(image_file.name).stem}_{model_tag}_annotated.png",
                        mime="image/png",
                    )

                json_str = json.dumps(detections_json, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Télécharger les détections (JSON)",
                    data=json_str.encode("utf-8"),
                    file_name=f"{Path(image_file.name).stem}_{model_tag}.json",
                    mime="application/json",
                )

                img_csv_df = pd.DataFrame(
                    [
                        {
                            "class": d["class"],
                            "confidence": round(d["confidence"], 4),
                            "x_min": d["bbox"]["x_min"],
                            "y_min": d["bbox"]["y_min"],
                            "x_max": d["bbox"]["x_max"],
                            "y_max": d["bbox"]["y_max"],
                        }
                        for d in detections_json
                    ]
                )
                st.download_button(
                    label="Télécharger le rapport (CSV)",
                    data=img_csv_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{Path(image_file.name).stem}_{model_tag}_report.csv",
                    mime="text/csv",
                )

# --- Onglet 2 : Video Processing ---
with tab_video:
    st.header("Détection, Segmentation et Suivi ByteTrack sur Vidéo")
    if ffmpeg_path is None:
        show_ffmpeg_missing_warning()

    video_file = st.file_uploader("Importer une vidéo MP4", type=["mp4"])
    vid_stride = st.slider(
        "Pas d'échantillonnage vidéo (vid_stride)",
        min_value=1,
        max_value=5,
        value=1,
        help="1 = toutes les frames. Sur CPU, passez à 2 ou 3 pour accélérer fortement l'analyse.",
    )
    process_video_button = st.button("Traiter la vidéo", type="primary")

    if process_video_button and ffmpeg_path is None:
        show_ffmpeg_missing_warning()
    elif process_video_button and video_file is None:
        st.warning("Veuillez importer une vidéo MP4 avant de lancer le traitement.")
    elif process_video_button and video_file is not None:
        progress_bar = st.progress(0.0, text="Initialisation du traitement vidéo...")

        def update_video_progress(done_frames: int, total_expected):
            if total_expected and total_expected > 0:
                ratio = min(1.0, done_frames / float(total_expected))
                progress_bar.progress(
                    ratio,
                    text=f"Analyse & suivi ByteTrack : frame {done_frames} / {total_expected} ({ratio * 100:.0f} %)",
                )
            else:
                progress_bar.progress(0.5, text=f"Analyse & suivi ByteTrack : {done_frames} frames traitées...")

        try:
            with tempfile.TemporaryDirectory() as upload_tmp_dir:
                temp_input_path = Path(upload_tmp_dir) / Path(video_file.name).name
                temp_input_path.write_bytes(video_file.getvalue())
                video_results = video_processing(
                    str(temp_input_path),
                    model=model,
                    model_name=model_tag,
                    conf=conf_threshold,
                    device=device,
                    include_masks=not exclude_masks_json,
                    vid_stride=vid_stride,
                    progress_callback=update_video_progress,
                )
            progress_bar.progress(1.0, text="Traitement terminé et encodé en H.264 !")
            st.session_state["last_video_results"] = {
                **video_results,
                "source_name": Path(video_file.name).stem,
                "model_tag": model_tag,
            }
        except subprocess.CalledProcessError as ff_err:
            st.error("Erreur lors du ré-encodage FFmpeg en H.264.")
            if ff_err.stderr:
                st.code(ff_err.stderr.decode("utf-8", errors="ignore"))
        except Exception as exc:
            st.error(f"Erreur pendant l'analyse vidéo : {exc}")
            st.exception(exc)

    if "last_video_results" in st.session_state:
        res_state = st.session_state["last_video_results"]
        out_mp4_path = Path(res_state["video_out"])
        out_json_path = Path(res_state["json_out"])
        if out_mp4_path.exists() and out_json_path.exists():
            video_bytes = out_mp4_path.read_bytes()
            json_bytes = out_json_path.read_bytes()
            csv_bytes = res_state["report_df"].to_csv(index=False).encode("utf-8")

            col_vid, col_vstats = st.columns([2, 1])
            with col_vid:
                st.video(video_bytes)

            with col_vstats:
                st.subheader("Performances & Comptage Multi-Classes")
                c1, c2, c3 = st.columns(3)
                c1.metric("Inférence moy.", f"{res_state['avg_inference_ms']:.1f} ms")
                c2.metric("Vitesse moy.", f"{res_state['avg_fps']:.1f} FPS")
                c3.metric("FPS source", f"{res_state['source_fps']:.1f}")

                st.dataframe(res_state["summary_df"], use_container_width=True, hide_index=True)

                st.subheader("Téléchargements")
                st.download_button(
                    label="Télécharger la vidéo annotée (MP4)",
                    data=video_bytes,
                    file_name=out_mp4_path.name,
                    mime="video/mp4",
                )
                st.download_button(
                    label="Télécharger les détections (JSON)",
                    data=json_bytes,
                    file_name=out_json_path.name,
                    mime="application/json",
                )
                st.download_button(
                    label="Télécharger le rapport statistique (CSV)",
                    data=csv_bytes,
                    file_name=f"{res_state['source_name']}_{res_state['model_tag']}_tracking_report.csv",
                    mime="text/csv",
                )

            if not res_state["report_df"].empty:
                with st.expander("Aperçu du rapport statistique des pistes suivies (CSV)"):
                    st.dataframe(res_state["report_df"], use_container_width=True, hide_index=True)
