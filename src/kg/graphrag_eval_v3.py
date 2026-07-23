"""
GraphRAG v3 — Expanded KG with YOLO object detection.
Adds to v2 (building color + road marking):
  - vehicles (count, types)
  - pedestrians (present, count)
  - traffic lights (present)
  - signs (present, types)

Usage:
    pip install ultralytics
    python src/kg/graphrag_eval_v3.py --set Set_34 --max_clips 5 --kg_only
    python src/kg/graphrag_eval_v3.py --set Set_34
"""

import argparse, os, sys, json, torch, cv2
import numpy as np
import networkx as nx
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from loguru import logger

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/path/to/your/data")
DATA_DIR   = BASE_DIR / "data" / "videos"
CACHE_DIR  = BASE_DIR / "results" / "kg_cache_v3"
EVAL_DIR   = BASE_DIR / "results" / "eval"
BASE_MODEL = str(BASE_DIR / "models" / "base_model")
FT_ADAPTER = str(BASE_DIR / "models" / "baseline_adapter")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ── YOLO class IDs (COCO) ──────────────────────────────────────────────────
VEHICLE_IDS     = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PEDESTRIAN_ID   = 0   # person
TRAFFIC_LIGHT   = 9
STOP_SIGN       = 11

# ── Per-set building ROI ───────────────────────────────────────────────────
SET_BUILDING_ROI = {
    "Set_33": (0.75, 0.50, 1.0, 0.75),
    "default": (0.55, 0.28, 1.0, 0.62),
}


# ═══════════════════════════ Color Extraction (from v2) ════════════════════

def hsv_to_color_name(h, s, v):
    if v < 50: return "dark"
    if v > 220 and s < 30: return "overexposed"
    if s < 30:
        if v > 200: return "white"
        if v > 130: return "light-gray"
        if v > 80:  return "gray"
        return "dark-gray"
    # Desaturated green/teal = olive-gray building (morning light on concrete)
    if 60 <= h <= 120 and s < 55 and v > 80:
        return "olive-gray"
    if 40 <= s < 150 and 15 <= h <= 70 and v > 80:
        return "sandy-beige"
    if h < 10 or h > 170: return "red"
    if 10 <= h < 25:  return "orange"
    if 25 <= h < 35:  return "yellow-orange"
    if 35 <= h < 50:  return "yellow"
    if 50 <= h < 85:  return "yellow-green"
    if 85 <= h < 105: return "green"
    if 105 <= h < 125: return "teal"
    if 125 <= h < 145: return "blue"
    if 145 <= h < 160: return "indigo"
    if 160 <= h <= 170: return "purple"
    return "unknown"


def color_name_to_family(name):
    earth_tones = {"beige","tan","olive-gray","taupe","greige","concrete","earthy",
                   "stone","muted","softly","khaki","buff","cream","light-brown",
                   "sandy-beige","yellow-orange","warm"}
    grays   = {"gray","grey","dark-gray","light-gray","charcoal","slate","ashen"}
    whites  = {"white","pale","bright","silver"}
    browns  = {"brown","bronze","ochre","sienna","rust","tan"}
    if name in earth_tones: return "EARTH_TONE"
    if name in grays:       return "GRAY"
    if name in whites:      return "WHITE"
    if name in browns:      return "BROWN"
    return "OTHER"


def extract_dominant_color(frame, roi=(0.55, 0.28, 1.0, 0.62)):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(roi[0]*w), int(roi[1]*h), int(roi[2]*w), int(roi[3]*h)
    region = frame[y1:y2, x1:x2]
    if region.size == 0: return "unknown"
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:,:,0].flatten(), hsv[:,:,1].flatten(), hsv[:,:,2].flatten()
    mask = (V <= 220) & ~((H>85)&(H<135)&(S>70))
    if mask.sum() < 50: return "overexposed"
    mh, ms, mv = H[mask].mean(), S[mask].mean(), V[mask].mean()
    return hsv_to_color_name(mh, ms, mv)


def classify_road_marking(frame):
    h, w = frame.shape[:2]
    roi = frame[int(0.55*h):int(0.85*h), int(0.3*w):int(0.7*w)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_pct = thresh.mean() / 255
    if white_pct < 0.02: return "none"
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return "none"
    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)
    aspect = cw / ch if ch > 0 else 0
    area = cv2.contourArea(largest)
    if area > (roi.shape[0]*roi.shape[1]*0.05) and 0.8 < aspect < 1.2:
        return "keep-clear box"
    elif area > 200:
        return "white marking"
    return "none"


# ═══════════════════════════ YOLO Detection ════════════════════════════════

_yolo_model = None

def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            logger.info("Loading YOLOv8n...")
            _yolo_model = YOLO("yolov8n.pt")  # nano — fast, downloads ~6MB
        except ImportError:
            logger.warning("ultralytics not installed. Run: pip install ultralytics")
            _yolo_model = None
    return _yolo_model


def detect_objects_in_frame(frame):
    """Returns dict of detected objects."""
    yolo = get_yolo()
    if yolo is None:
        return {}

    results = yolo(frame, verbose=False, conf=0.3)[0]
    detections = {"vehicles": [], "pedestrians": 0, "traffic_light": False, "signs": []}

    for box in results.boxes:
        cls_id = int(box.cls.item())
        if cls_id in VEHICLE_IDS:
            detections["vehicles"].append(VEHICLE_IDS[cls_id])
        elif cls_id == PEDESTRIAN_ID:
            detections["pedestrians"] += 1
        elif cls_id == TRAFFIC_LIGHT:
            detections["traffic_light"] = True
        elif cls_id == STOP_SIGN:
            detections["signs"].append("stop sign")

    return detections


# ═══════════════════════════ Frame Extraction ══════════════════════════════

def find_video_path(set_name, clip_name):
    for p in DATA_DIR.rglob(clip_name.replace(".mp4", "_blurred.mp4")):
        if set_name in str(p):
            return str(p)
    for p in DATA_DIR.rglob(clip_name):
        if set_name in str(p):
            return str(p)
    return None


def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0: return []
    indices = np.linspace(0, total-1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret: frames.append(frame)
    cap.release()
    return frames


# ═══════════════════════════ KG Builder ════════════════════════════════════

def build_clip_kg_v3(video_path, set_name="default"):
    """Build expanded KG: color + road + vehicles + pedestrians + signals."""
    G = nx.Graph()
    frames = extract_frames(video_path, num_frames=8)
    if not frames: return G

    building_roi = SET_BUILDING_ROI.get(set_name, SET_BUILDING_ROI["default"])

    # ── Pixel-based features (from v2) ────────────────────────────────────
    building_colors, road_markings = [], []
    for frame in frames:
        color = extract_dominant_color(frame, roi=building_roi)
        if color not in ("unknown", "overexposed"):
            building_colors.append(color)
        marking = classify_road_marking(frame)
        if marking != "none":
            road_markings.append(marking)

    if building_colors:
        dom_color = Counter(building_colors).most_common(1)[0][0]
        G.add_node("building", type="building", color=dom_color,
                   color_family=color_name_to_family(dom_color),
                   glare_coverage=round(1 - len(building_colors)/len(frames), 2))

    if road_markings:
        G.add_node("road_marking", type="marking",
                   description=Counter(road_markings).most_common(1)[0][0])

    # ── YOLO-based features ────────────────────────────────────────────────
    all_vehicle_types, pedestrian_counts, has_signal, sign_types = [], [], [], []

    for frame in frames:
        det = detect_objects_in_frame(frame)
        if det:
            all_vehicle_types.extend(det.get("vehicles", []))
            pedestrian_counts.append(det.get("pedestrians", 0))
            if det.get("traffic_light"): has_signal.append(True)
            sign_types.extend(det.get("signs", []))

    # Vehicles node
    if all_vehicle_types:
        type_counts = Counter(all_vehicle_types)
        median_count = int(np.median([
            len([v for v in det_frame if det_frame])/len(frames)
            for det_frame in [all_vehicle_types]
        ])) if all_vehicle_types else 0
        G.add_node("vehicles", type="vehicles",
                   count=len(set(all_vehicle_types)),
                   types=list(type_counts.keys()),
                   most_common=type_counts.most_common(1)[0][0],
                   type_counts=dict(type_counts))

    # Pedestrians node
    if pedestrian_counts:
        median_peds = int(np.median(pedestrian_counts))
        G.add_node("pedestrians", type="pedestrians",
                   present=median_peds > 0,
                   count=median_peds,
                   max_count=max(pedestrian_counts))

    # Traffic light node
    if any(has_signal):
        G.add_node("traffic_light", type="signal",
                   present=True,
                   detected_in_frames=sum(has_signal))

    # Signs node
    if sign_types:
        G.add_node("signs", type="signs",
                   present=True,
                   types=list(set(sign_types)))

    return G


# ═══════════════════════════ KG Cache ══════════════════════════════════════

def kg_to_dict(G):
    return {"nodes": {n: d for n, d in G.nodes(data=True)},
            "edges": list(G.edges())}

def dict_to_kg(d):
    G = nx.Graph()
    for n, data in d["nodes"].items():
        G.add_node(n, **data)
    G.add_edges_from(d["edges"])
    return G

def save_kg(G, clip_key):
    path = CACHE_DIR / f"{clip_key.replace('/', '_')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(kg_to_dict(G), f)

def load_kg(clip_key):
    path = CACHE_DIR / f"{clip_key.replace('/', '_')}.json"
    if path.exists():
        return dict_to_kg(json.load(open(path)))
    return None


# ═══════════════════════════ Prompt Builder ════════════════════════════════

def build_kg_prompt(G, question):
    nodes = dict(G.nodes(data=True))
    q_lower = question.lower()
    facts = []

    # Building color
    if any(w in q_lower for w in ["colour","color","hue","appear","look","shade","building","structure"]):
        if "building" in nodes:
            n = nodes["building"]
            facts.append(f"The building color is {n['color']} (color family: {n['color_family']}, measured from video frames).")

    # Road marking
    if any(w in q_lower for w in ["marking","painted","road","lane","centre","center","middle"]):
        if "road_marking" in nodes:
            facts.append(f"The road marking is: {nodes['road_marking']['description']}.")

    # Vehicles
    if any(w in q_lower for w in ["vehicle","car","bus","truck","motorcycle","bicycle","how many","count","type"]):
        if "vehicles" in nodes:
            n = nodes["vehicles"]
            facts.append(f"Vehicles detected: {n['type_counts']}. Most common: {n['most_common']}.")

    # Pedestrians
    if any(w in q_lower for w in ["pedestrian","person","people","walking","crossing"]):
        if "pedestrians" in nodes:
            n = nodes["pedestrians"]
            facts.append(f"Pedestrians: {'present' if n['present'] else 'none detected'} (max count: {n['max_count']}).")

    # Traffic light
    if any(w in q_lower for w in ["signal","light","traffic light","green","red","amber"]):
        if "traffic_light" in nodes:
            facts.append("A traffic light/signal was detected in this scene.")

    # Signs
    if any(w in q_lower for w in ["sign","board","signage"]):
        if "signs" in nodes:
            facts.append(f"Signs detected: {nodes['signs']['types']}.")

    if not facts:
        return None
    return "Scene knowledge from video analysis:\n" + "\n".join(f"- {f}" for f in facts)


# ═══════════════════════════ Main ══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set",        required=True)
    parser.add_argument("--max_clips",  type=int, default=200)
    parser.add_argument("--kg_only",    action="store_true")
    parser.add_argument("--use_ft",     action="store_true", default=True)
    args = parser.parse_args()
    set_name = args.set

    # Load dataset
    import csv
    data_csv = BASE_DIR / "data" / "morning_attribution.csv"
    all_rows = list(csv.DictReader(open(data_csv)))
    rows = [r for r in all_rows if r.get("set") == set_name]
    logger.info(f"Set: {set_name} | Questions: {len(rows)}")

    # Unique clips
    clip_col = next(c for c in rows[0].keys() if "video" in c.lower() or "clip" in c.lower())
    unique_clips = list(dict.fromkeys(
        (set_name, Path(r[clip_col]).name) for r in rows
    ))[:args.max_clips]

    # Build KGs
    logger.info("Building KGs (pixel + YOLO)...")
    kg_cache = {}
    for (sname, clip_name) in tqdm(unique_clips, desc="Building KGs"):
        clip_key = f"{sname}/{clip_name}"
        G = load_kg(clip_key)
        if G is None:
            vp = find_video_path(sname, clip_name)
            if vp:
                G = build_clip_kg_v3(vp, set_name=sname)
                save_kg(G, clip_key)
            else:
                G = nx.Graph()
        kg_cache[clip_key] = G
        logger.debug(f"{clip_key}: {dict(G.nodes(data=True))}")

    if args.kg_only:
        logger.info("KG-only mode done. Inspect kg_cache_v3/")
        return

    # Load model
    logger.info("Loading model...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if args.use_ft:
        model = PeftModel.from_pretrained(model, FT_ADAPTER)
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True,
        min_pixels=256*28*28, max_pixels=1280*28*28)
    model.eval()

    from qwen_vl_utils import process_vision_info

    # Evaluate
    out_path = EVAL_DIR / f"E006_GraphRAGv3_{set_name}.csv"
    import csv as csv_module
    fieldnames = ["question_id","video_id","set","question","answer_ft","answer_graphrag","actual_answer","kg_used"]
    writer = csv_module.DictWriter(open(out_path,"w",newline=""), fieldnames=fieldnames)
    writer.writeheader()

    for row in tqdm(rows, desc="Evaluating"):
        clip_name = Path(row[clip_col]).name
        clip_key  = f"{set_name}/{clip_name}"
        G = kg_cache.get(clip_key, nx.Graph())

        question = row["question"]
        actual   = row.get("actual_answer","")

        # FT-only answer (no KG)
        messages_ft = [{"role":"user","content":[
            {"type":"video","video":find_video_path(set_name, clip_name) or "","fps":1.0},
            {"type":"text","text":question}
        ]}]
        text_ft = processor.apply_chat_template(messages_ft, tokenize=False, add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(messages_ft)
        inputs_ft = processor(text=[text_ft], images=img_inputs, videos=vid_inputs, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs_ft, max_new_tokens=64)
        answer_ft = processor.decode(out[0][inputs_ft.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # KG-augmented answer
        kg_hint = build_kg_prompt(G, question)
        kg_used = kg_hint is not None
        augmented_q = f"{kg_hint}\n\nQuestion: {question}" if kg_hint else question

        messages_kg = [{"role":"user","content":[
            {"type":"video","video":find_video_path(set_name, clip_name) or "","fps":1.0},
            {"type":"text","text":augmented_q}
        ]}]
        text_kg = processor.apply_chat_template(messages_kg, tokenize=False, add_generation_prompt=True)
        img_inputs2, vid_inputs2 = process_vision_info(messages_kg)
        inputs_kg = processor(text=[text_kg], images=img_inputs2, videos=vid_inputs2, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out2 = model.generate(**inputs_kg, max_new_tokens=64)
        answer_kg = processor.decode(out2[0][inputs_kg.input_ids.shape[1]:], skip_special_tokens=True).strip()

        logger.info(f"Q:    {question[:60]}")
        logger.info(f"FT:   {answer_ft[:60]}")
        logger.info(f"KGv3: {answer_kg[:60]}")
        logger.info(f"Act:  {actual[:60]}\n")

        writer.writerow({
            "question_id":   row.get("question_id",""),
            "video_id":      row[clip_col],
            "set":           set_name,
            "question":      question,
            "answer_ft":     answer_ft,
            "answer_graphrag": answer_kg,
            "actual_answer": actual,
            "kg_used":       kg_used,
        })

    logger.info(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    main()
