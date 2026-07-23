"""
GraphRAG-KG-v3 with LLaVA-NeXT-Video-7B backbone.

Identical KG pipeline to graphrag_eval_v3.py (pixel color + YOLO),
but replaces Qwen with LLaVA-NeXT-Video-7B as the answering VLM.

This proves the KG technique is model-agnostic.

Output: results/eval/E010_GraphRAGLLaVA_all.csv
  → column: generated_answer (same format as llava_eval.py)
  → model:  "LLaVA-NeXT-Video-7B-KGv3"

Usage:
    python src/kg/graphrag_eval_llava.py --set Set_34 --max_clips 3   # test
    python src/kg/graphrag_eval_llava.py                               # full 403 Qs
"""

import argparse, csv, json, os, sys, torch, cv2
import numpy as np
import networkx as nx
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from loguru import logger
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/path/to/your/data")
DATA_DIR   = BASE_DIR / "data" / "videos"
CACHE_DIR  = BASE_DIR / "results" / "kg_cache_v3"      # reuse existing v3 cache
EVAL_DIR   = BASE_DIR / "results" / "eval"
DATA_CSV   = BASE_DIR / "data" / "morning_attribution.csv"
LLAVA_PATH = str(BASE_DIR / "models" / "llava_next_video_7b")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ── YOLO class IDs (COCO) ──────────────────────────────────────────────────
VEHICLE_IDS   = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PEDESTRIAN_ID = 0
TRAFFIC_LIGHT = 9
STOP_SIGN     = 11

# ── Per-set building ROI (from v3) ─────────────────────────────────────────
SET_BUILDING_ROI = {
    "Set_33": (0.75, 0.50, 1.0, 0.75),
    "default": (0.55, 0.28, 1.0, 0.62),
}


# ═══════════════════════════ Color Extraction (identical to v3) ════════════

def hsv_to_color_name(h, s, v):
    if v < 50: return "dark"
    if v > 220 and s < 30: return "overexposed"
    if s < 30:
        if v > 200: return "white"
        if v > 130: return "light-gray"
        if v > 80:  return "gray"
        return "dark-gray"
    if 60 <= h <= 120 and s < 55 and v > 80: return "olive-gray"
    if 40 <= s < 150 and 15 <= h <= 70 and v > 80: return "sandy-beige"
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
    grays  = {"gray","grey","dark-gray","light-gray","charcoal","slate","ashen"}
    whites = {"white","pale","bright","silver"}
    browns = {"brown","bronze","ochre","sienna","rust","tan"}
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


# ═══════════════════════════ YOLO (identical to v3) ════════════════════════

_yolo_model = None

def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            logger.info("Loading YOLOv8n...")
            _yolo_model = YOLO("yolov8n.pt")
        except ImportError:
            logger.warning("ultralytics not installed — YOLO features disabled")
    return _yolo_model


def detect_objects_in_frame(frame):
    yolo = get_yolo()
    if yolo is None: return {}
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


# ═══════════════════════════ Frame & Video ═════════════════════════════════

def find_video_path(set_name, clip_name):
    for p in DATA_DIR.rglob(clip_name.replace(".mp4", "_blurred.mp4")):
        if set_name in str(p): return str(p)
    for p in DATA_DIR.rglob(clip_name):
        if set_name in str(p): return str(p)
    return None


def extract_frames_bgr(video_path, num_frames=8):
    """Extract frames as BGR numpy arrays (for KG/YOLO)."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0: return []
    indices = np.linspace(0, total-1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret: frames.append(frame)
    cap.release()
    return frames


def extract_frames_rgb_array(video_path, num_frames=8):
    """Extract frames as (T, H, W, C) RGB numpy array (for LLaVA)."""
    frames_bgr = extract_frames_bgr(video_path, num_frames)
    if not frames_bgr: return None
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    return np.stack(rgb, axis=0)   # (T, H, W, 3)


# ═══════════════════════════ KG Builder (identical to v3) ══════════════════

def build_clip_kg_v3(video_path, set_name="default"):
    G = nx.Graph()
    frames = extract_frames_bgr(video_path, num_frames=8)
    if not frames: return G

    building_roi = SET_BUILDING_ROI.get(set_name, SET_BUILDING_ROI["default"])
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

    all_vehicle_types, pedestrian_counts, has_signal, sign_types = [], [], [], []
    for frame in frames:
        det = detect_objects_in_frame(frame)
        if det:
            all_vehicle_types.extend(det.get("vehicles", []))
            pedestrian_counts.append(det.get("pedestrians", 0))
            if det.get("traffic_light"): has_signal.append(True)
            sign_types.extend(det.get("signs", []))

    if all_vehicle_types:
        type_counts = Counter(all_vehicle_types)
        G.add_node("vehicles", type="vehicles",
                   count=len(set(all_vehicle_types)),
                   types=list(type_counts.keys()),
                   most_common=type_counts.most_common(1)[0][0],
                   type_counts=dict(type_counts))

    if pedestrian_counts:
        median_peds = int(np.median(pedestrian_counts))
        G.add_node("pedestrians", type="pedestrians",
                   present=median_peds > 0,
                   count=median_peds,
                   max_count=max(pedestrian_counts))

    if any(has_signal):
        G.add_node("traffic_light", type="signal",
                   present=True, detected_in_frames=sum(has_signal))

    if sign_types:
        G.add_node("signs", type="signs", present=True, types=list(set(sign_types)))

    return G


# ═══════════════════════════ KG Cache ══════════════════════════════════════

def kg_to_dict(G):
    return {"nodes": {n: d for n, d in G.nodes(data=True)}, "edges": list(G.edges())}

def dict_to_kg(d):
    G = nx.Graph()
    for n, data in d["nodes"].items(): G.add_node(n, **data)
    G.add_edges_from(d["edges"])
    return G

def save_kg(G, clip_key):
    path = CACHE_DIR / f"{clip_key.replace('/', '_')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump(kg_to_dict(G), f)

def load_kg(clip_key):
    path = CACHE_DIR / f"{clip_key.replace('/', '_')}.json"
    if path.exists(): return dict_to_kg(json.load(open(path)))
    return None


# ═══════════════════════════ Prompt Builder (identical to v3) ══════════════

def build_kg_context(G, question):
    nodes = dict(G.nodes(data=True))
    q_lower = question.lower()
    facts = []

    if any(w in q_lower for w in ["colour","color","hue","appear","look","shade","building","structure","tower"]):
        if "building" in nodes:
            n = nodes["building"]
            facts.append(f"The building color is {n['color']} (color family: {n['color_family']}, measured from video frames).")

    if any(w in q_lower for w in ["marking","painted","road","lane","centre","center","middle","graphic","shape"]):
        if "road_marking" in nodes:
            facts.append(f"The road marking is: {nodes['road_marking']['description']}.")

    if any(w in q_lower for w in ["vehicle","car","bus","truck","motorcycle","bicycle","how many","count","type"]):
        if "vehicles" in nodes:
            n = nodes["vehicles"]
            facts.append(f"Vehicles detected: {n['type_counts']}. Most common: {n['most_common']}.")

    if any(w in q_lower for w in ["pedestrian","person","people","walking","crossing"]):
        if "pedestrians" in nodes:
            n = nodes["pedestrians"]
            facts.append(f"Pedestrians: {'present' if n['present'] else 'none detected'} (max count: {n['max_count']}).")

    if any(w in q_lower for w in ["signal","light","traffic light","green","red","amber"]):
        if "traffic_light" in nodes:
            facts.append("A traffic light/signal was detected in this scene.")

    if any(w in q_lower for w in ["sign","board","signage"]):
        if "signs" in nodes:
            facts.append(f"Signs detected: {nodes['signs']['types']}.")

    if not facts:
        return None
    return "Scene knowledge from video analysis:\n" + "\n".join(f"- {f}" for f in facts)


# ═══════════════════════════ LLaVA Inference ═══════════════════════════════

def load_llava():
    logger.info(f"Loading LLaVA-NeXT-Video-7B from {LLAVA_PATH}...")
    from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

    processor = LlavaNextVideoProcessor.from_pretrained(LLAVA_PATH)
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        LLAVA_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    logger.info("LLaVA loaded.")
    return model, processor


def run_llava_inference(model, processor, video_path, question, kg_context=None):
    """Run LLaVA inference with optional KG context prepended to question."""
    video_array = extract_frames_rgb_array(video_path, num_frames=8)
    if video_array is None:
        return "Video could not be loaded."

    # Prepend KG context to question if available
    augmented_q = f"{kg_context}\n\nQuestion: {question}" if kg_context else question

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": augmented_q},
            ],
        }
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(
        text=prompt,
        videos=video_array,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = processor.decode(generated, skip_special_tokens=True).strip()
    return answer


# ═══════════════════════════ Main ══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GraphRAG-KG-v3 with LLaVA backbone")
    parser.add_argument("--set",       default=None, help="Evaluate one set only (e.g. Set_34)")
    parser.add_argument("--max_clips", type=int,     default=None, help="Limit to N clips (testing)")
    parser.add_argument("--output",    default=None, help="Override output CSV path")
    args = parser.parse_args()

    # ── Load dataset ─────────────────────────────────────────────────────────
    rows = list(csv.DictReader(open(DATA_CSV)))
    if args.set:
        rows = [r for r in rows if r["set"] == args.set]

    if args.max_clips:
        seen, filtered = set(), []
        for r in rows:
            key = (r["set"], r["video_file_path"])
            if key not in seen:
                seen.add(key)
                if len(seen) > args.max_clips: break
            if key in seen: filtered.append(r)
        rows = filtered

    logger.info(f"Evaluating {len(rows)} questions with LLaVA+KG-v3...")

    # ── Output path ───────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
    elif args.set:
        out_path = EVAL_DIR / f"E010_GraphRAGLLaVA_{args.set}.csv"
    else:
        out_path = EVAL_DIR / "E010_GraphRAGLLaVA_all.csv"

    # ── Pre-build KG cache for all clips ─────────────────────────────────────
    logger.info("Building/loading KG cache (pixel + YOLO)...")
    unique_clips = list(dict.fromkeys(
        (r["set"], Path(r["video_file_path"]).name) for r in rows
    ))

    kg_cache = {}
    for (set_name, clip_name) in tqdm(unique_clips, desc="Building KGs"):
        clip_key = f"{set_name}/{clip_name}"
        G = load_kg(clip_key)
        if G is None:
            vp = find_video_path(set_name, clip_name)
            if vp:
                G = build_clip_kg_v3(vp, set_name=set_name)
                save_kg(G, clip_key)
            else:
                logger.warning(f"Video not found: {clip_key}")
                G = nx.Graph()
        kg_cache[clip_key] = G

    logger.info(f"KG cache ready: {len(kg_cache)} clips")

    # ── Load LLaVA ────────────────────────────────────────────────────────────
    model, processor = load_llava()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    fieldnames = ["question_id", "video_id", "set", "question",
                  "model", "generated_answer", "actual_answer", "kg_used"]
    results, errors = [], 0
    kg_used_count = 0

    for row in tqdm(rows, desc="LLaVA+KG Inference"):
        set_name  = row["set"]
        clip_name = row["video_file_path"]
        question  = row["question"]
        actual    = row["actual_answer"]

        clip_key   = f"{set_name}/{Path(clip_name).name}"
        G          = kg_cache.get(clip_key, nx.Graph())
        kg_context = build_kg_context(G, question)
        kg_used    = kg_context is not None
        if kg_used: kg_used_count += 1

        video_path = find_video_path(set_name, Path(clip_name).name)
        if not video_path:
            logger.warning(f"Video not found: {set_name}/{clip_name}")
            answer = "Video not found."
            errors += 1
        else:
            try:
                answer = run_llava_inference(model, processor, video_path, question, kg_context)
            except Exception as e:
                logger.error(f"Inference error: {e}")
                answer = "Error during inference."
                errors += 1

        logger.info(f"Q:      {question[:60]}")
        logger.info(f"LLaVA+KG: {answer[:60]}")
        logger.info(f"Act:    {actual[:60]}  [KG={'YES' if kg_used else 'NO'}]\n")

        results.append({
            "question_id":      row["question_id"],
            "video_id":         clip_name,
            "set":              set_name,
            "question":         question,
            "model":            "LLaVA-NeXT-Video-7B-KGv3",
            "generated_answer": answer,
            "actual_answer":    actual,
            "kg_used":          kg_used,
        })

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"\nDone! {len(results)} questions, {errors} errors.")
    logger.info(f"KG context used in {kg_used_count}/{len(results)} questions ({kg_used_count/len(results)*100:.1f}%)")
    logger.info(f"Saved to: {out_path}")
    logger.info("Next: python src/eval/complete_eval.py --include_llava_kg")


if __name__ == "__main__":
    main()
