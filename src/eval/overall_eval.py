"""
Overall Evaluation — All 5 Question Types (Morning, 5 Sets)
============================================================
Runs Full KG v3 (pixel + YOLO) on all 5 UDVideoQA question types.

Question Types evaluated:
  Atr  — Attribution           (building colour, road markings)
  BU   — Basic Understanding   (scene comprehension)
  ER   — Event Reasoning       (cause-and-effect)
  RR   — Reverse Reasoning     (temporal ordering)
  CI   — Counterfactual Inference (hallucination robustness)

Attribution results are loaded from existing E006_GraphRAGv3_*.csv files
(already computed). BU / ER / RR / CI are run fresh.

Usage:
    # Smoke test — 5 questions per type, no Groq:
    CUDA_VISIBLE_DEVICES=6 python src/eval/overall_eval.py --max_q 5 --skip_judge

    # Full inference only (~2 hrs):
    CUDA_VISIBLE_DEVICES=6 python src/eval/overall_eval.py --skip_judge

    # Judge only (after inference CSVs are ready):
    python src/eval/overall_eval.py --judge_only

    # Full run (inference + judge, ~3 hrs total):
    CUDA_VISIBLE_DEVICES=6 python src/eval/overall_eval.py
"""

import argparse, csv, glob, json, os, sys, time
import cv2
import networkx as nx
import numpy as np
import torch
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/path/to/your/data")
DATA_DIR    = BASE_DIR / "data" / "videos"
ANN_DIR     = BASE_DIR / "data" / "raw" / "capstone_annotations"
EVAL_DIR    = BASE_DIR / "results" / "eval"
OVERALL_DIR = BASE_DIR / "results" / "overall"
BASE_MODEL  = str(BASE_DIR / "models" / "base_model")
FT_ADAPTER  = str(BASE_DIR / "models" / "baseline_adapter")

OVERALL_DIR.mkdir(parents=True, exist_ok=True)

# The 5 morning sets used throughout this project
MORNING_SETS = ["Set_26", "Set_30", "Set_33", "Set_34", "Set_35"]

# Per-set building ROI (calibrated earlier)
SET_BUILDING_ROI = {
    "Set_33": (0.75, 0.50, 1.0, 0.75),
    "default": (0.55, 0.28, 1.0, 0.62),
}

# ── Category normalisation ─────────────────────────────────────────────────────
# The raw JSONL files have messy/inconsistent category names — normalise all.
CATEGORY_MAP = {
    "attribution":                   "Atr",
    "basic understanding":           "BU",
    "event reasoning":               "ER",
    "wevent reasoning":              "ER",   # typo in annotations
    "reverse reasoning":             "RR",
    "temporal ordering":             "RR",   # sub-type, merge into RR
    "counterfactual inference":      "CI",
    "counterfactual":                "CI",
    "counterfact":                   "CI",
    "counterfact inference":         "CI",
    "acounterfactual inference":     "CI",   # typo
}

QTYPE_LABELS = {
    "Atr": "Attribution",
    "BU":  "Basic Understanding",
    "ER":  "Event Reasoning",
    "RR":  "Reverse Reasoning",
    "CI":  "Counterfactual Inference",
}

QTYPE_ORDER = ["Atr", "BU", "ER", "RR", "CI"]

# Weights from UDVideoQA paper (for weighted overall score)
QTYPE_WEIGHTS = {"BU": 1.0, "Atr": 1.2, "ER": 1.3, "RR": 1.3, "CI": 1.5}

# YOLO COCO class IDs
VEHICLE_IDS   = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PEDESTRIAN_ID = 0
TRAFFIC_LIGHT = 9
STOP_SIGN     = 11


# ══════════════════════════════════════════════════════════════════════════════
# ANNOTATION LOADING
# ══════════════════════════════════════════════════════════════════════════════

def normalise_category(raw):
    """Map raw annotation category to canonical 2-letter code. Returns None if unknown."""
    key = raw.strip().lower()
    return CATEGORY_MAP.get(key, None)


def load_morning_questions(target_qtypes=None):
    """
    Load questions from all 5 morning set JSONL files.

    Returns a dict: {qtype: [row_dict, ...]}
    Each row_dict: question_id, video_file_path, set, question, actual_answer, qtype
    Only rows where the video file is found on disk are included.
    """
    if target_qtypes is None:
        target_qtypes = QTYPE_ORDER

    questions = defaultdict(list)
    skipped_no_video = 0
    skipped_bad_cat  = defaultdict(int)

    for set_name in MORNING_SETS:
        ann_folder = ANN_DIR / set_name
        if not ann_folder.exists():
            logger.warning(f"Annotation folder not found: {ann_folder}")
            continue

        jsonl_files = list(ann_folder.glob("*.jsonl"))
        if not jsonl_files:
            logger.warning(f"No JSONL files in {ann_folder}")
            continue

        logger.info(f"Loading {set_name}: {[f.name for f in jsonl_files]}")

        seen_ids = set()
        for jfile in jsonl_files:
            for line_no, line in enumerate(open(jfile, encoding="utf-8")):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                raw_cat = d.get("category", d.get("type", ""))
                qtype   = normalise_category(raw_cat)
                if qtype is None:
                    skipped_bad_cat[raw_cat] += 1
                    continue
                if qtype not in target_qtypes:
                    continue

                vid_path  = d.get("video_file_path", "")
                clip_name = Path(vid_path).name
                qid       = str(d.get("index", f"{set_name}_{line_no}"))

                # Deduplicate by question_id
                key = f"{set_name}_{qid}"
                if key in seen_ids:
                    continue
                seen_ids.add(key)

                # Only keep if video file exists on disk
                found = find_video_path(set_name, clip_name)
                if not found:
                    skipped_no_video += 1
                    continue

                questions[qtype].append({
                    "question_id":     qid,
                    "video_file_path": vid_path,
                    "set":             set_name,
                    "question":        d.get("question", ""),
                    "actual_answer":   d.get("answer", ""),
                    "qtype":           qtype,
                })

    logger.info(f"Loaded questions per type: { {k: len(v) for k,v in questions.items()} }")
    if skipped_no_video:
        logger.info(f"Skipped {skipped_no_video} rows — video not found on disk.")
    if skipped_bad_cat:
        logger.info(f"Skipped unknown categories: {dict(skipped_bad_cat)}")
    return questions


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO / PATH UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

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
    if total <= 0:
        return []
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# PIXEL-BASED COLOUR EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def hsv_to_color_name(h, s, v):
    if v < 50:             return "dark"
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
    earth = {"beige","tan","olive-gray","taupe","greige","concrete","earthy",
             "stone","muted","softly","khaki","buff","cream","light-brown",
             "sandy-beige","yellow-orange","warm"}
    grays  = {"gray","grey","dark-gray","light-gray","charcoal","slate","ashen"}
    whites = {"white","pale","bright","silver"}
    browns = {"brown","bronze","ochre","sienna","rust","tan"}
    if name in earth:  return "EARTH_TONE"
    if name in grays:  return "GRAY"
    if name in whites: return "WHITE"
    if name in browns: return "BROWN"
    return "OTHER"


def extract_dominant_color(frame, roi=(0.55, 0.28, 1.0, 0.62)):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(roi[0]*w), int(roi[1]*h), int(roi[2]*w), int(roi[3]*h)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:,:,0].flatten(), hsv[:,:,1].flatten(), hsv[:,:,2].flatten()
    mask = (V <= 220) & ~((H > 85) & (H < 135) & (S > 70))
    if mask.sum() < 50:
        return "overexposed"
    return hsv_to_color_name(H[mask].mean(), S[mask].mean(), V[mask].mean())


def classify_road_marking(frame):
    h, w = frame.shape[:2]
    roi  = frame[int(0.55*h):int(0.85*h), int(0.3*w):int(0.7*w)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    if thresh.mean() / 255 < 0.02:
        return "none"
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "none"
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 200:
        return "none"
    rh, rw = roi.shape[:2]
    if rw > 0 and rh > 0:
        q_tl = thresh[:rh//2, :rw//2].mean()
        q_tr = thresh[:rh//2, rw//2:].mean()
        q_bl = thresh[rh//2:, :rw//2].mean()
        q_br = thresh[rh//2:, rw//2:].mean()
        filled = sum(1 for q in [q_tl, q_tr, q_bl, q_br] if q > 30)
        if filled >= 3: return "keep-clear box"
        if filled == 2:
            if (q_tl > 30 and q_br > 30) or (q_tr > 30 and q_bl > 30):
                return "white X marking"
            return "white line"
    x, y, cw, ch = cv2.boundingRect(largest)
    aspect = cw / ch if ch > 0 else 0
    if area > (roi.shape[0]*roi.shape[1]*0.05) and 0.8 < aspect < 1.2:
        return "keep-clear box"
    return "white marking"


# ══════════════════════════════════════════════════════════════════════════════
# YOLO DETECTION
# ══════════════════════════════════════════════════════════════════════════════

_yolo_model = None

def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            logger.info("Loading YOLOv8n...")
            _yolo_model = YOLO("yolov8n.pt")
        except ImportError:
            logger.warning("ultralytics not installed — YOLO disabled.")
    return _yolo_model


def detect_objects(frame):
    yolo = get_yolo()
    if yolo is None:
        return {}
    results = yolo(frame, verbose=False, conf=0.3)[0]
    det = {"vehicles": [], "pedestrians": 0, "traffic_light": False, "signs": []}
    for box in results.boxes:
        cls_id = int(box.cls.item())
        if cls_id in VEHICLE_IDS:     det["vehicles"].append(VEHICLE_IDS[cls_id])
        elif cls_id == PEDESTRIAN_ID: det["pedestrians"] += 1
        elif cls_id == TRAFFIC_LIGHT: det["traffic_light"] = True
        elif cls_id == STOP_SIGN:     det["signs"].append("stop sign")
    return det


# ══════════════════════════════════════════════════════════════════════════════
# KG BUILDER — Full KG v3 (all components)
# ══════════════════════════════════════════════════════════════════════════════

def build_clip_kg(video_path, set_name="default"):
    G = nx.Graph()
    frames = extract_frames(video_path, num_frames=8)
    if not frames:
        return G

    roi = SET_BUILDING_ROI.get(set_name, SET_BUILDING_ROI["default"])

    # Building colour
    colors = []
    for frame in frames:
        c = extract_dominant_color(frame, roi=roi)
        if c not in ("unknown", "overexposed"):
            colors.append(c)
    if colors:
        dom = Counter(colors).most_common(1)[0][0]
        G.add_node("building", type="building", color=dom,
                   color_family=color_name_to_family(dom))

    # Road marking
    markings = []
    for frame in frames:
        m = classify_road_marking(frame)
        if m != "none":
            markings.append(m)
    if markings:
        G.add_node("road_marking", type="marking",
                   description=Counter(markings).most_common(1)[0][0])

    # YOLO
    all_veh, ped_counts, has_sig, signs = [], [], [], []
    for frame in frames:
        det = detect_objects(frame)
        if det:
            all_veh.extend(det.get("vehicles", []))
            ped_counts.append(det.get("pedestrians", 0))
            if det.get("traffic_light"): has_sig.append(True)
            signs.extend(det.get("signs", []))

    if all_veh:
        tc = Counter(all_veh)
        G.add_node("vehicles", type="vehicles",
                   count=len(set(all_veh)), types=list(tc.keys()),
                   most_common=tc.most_common(1)[0][0], type_counts=dict(tc))

    if ped_counts:
        med = int(np.median(ped_counts))
        G.add_node("pedestrians", type="pedestrians",
                   present=med > 0, count=med, max_count=max(ped_counts))

    if any(has_sig):
        G.add_node("traffic_light", type="signal",
                   present=True, detected_in_frames=sum(has_sig))

    if signs:
        G.add_node("signs", type="signs",
                   present=True, types=list(set(signs)))

    return G


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER — question-aware injection
# ══════════════════════════════════════════════════════════════════════════════

def build_kg_prompt(G, question):
    nodes = dict(G.nodes(data=True))
    q     = question.lower()
    facts = []

    if "building" in nodes and any(w in q for w in
        ["colour","color","hue","appear","look","shade","building","structure",
         "facade","wall","brick","painted","surface"]):
        n = nodes["building"]
        facts.append(f"The building color is {n['color']} (family: {n['color_family']}).")

    if "road_marking" in nodes and any(w in q for w in
        ["marking","painted","road","lane","centre","center","middle",
         "shape","sign","cross","box","keep","arrow","line"]):
        facts.append(f"Road marking detected: {nodes['road_marking']['description']}.")

    if "vehicles" in nodes and any(w in q for w in
        ["vehicle","car","bus","truck","motorcycle","bicycle",
         "how many","count","type","traffic","moving","enter","exit",
         "turn","approach","speed","collide","brake","park"]):
        n = nodes["vehicles"]
        facts.append(f"Vehicles detected: {n['type_counts']}. Most common: {n['most_common']}.")

    if "pedestrians" in nodes and any(w in q for w in
        ["pedestrian","person","people","walking","crossing","man","woman",
         "crowd","jaywal","wait","cross","street","sidewalk"]):
        n = nodes["pedestrians"]
        facts.append(
            f"Pedestrians: {'present' if n['present'] else 'none'} "
            f"(max count: {n['max_count']})."
        )

    if "traffic_light" in nodes and any(w in q for w in
        ["signal","light","traffic light","green","red","amber","stop","go",
         "phase","cycle"]):
        facts.append("A traffic light/signal is visible in this scene.")

    if "signs" in nodes and any(w in q for w in
        ["sign","board","signage","notice","speed","limit","warning"]):
        facts.append(f"Signs detected: {nodes['signs']['types']}.")

    if not facts:
        return None
    return "Scene knowledge from video analysis:\n" + "\n".join(f"- {f}" for f in facts)


# ══════════════════════════════════════════════════════════════════════════════
# VLM INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model, processor, question, video_path, kg_hint=None):
    from qwen_vl_utils import process_vision_info
    text_prompt = f"{kg_hint}\n\nQuestion: {question}" if kg_hint else question
    content = []
    if video_path and os.path.exists(video_path):
        content.append({"type": "video", "video": video_path,
                        "max_pixels": 360*28*28, "fps": 1.0})
    content.append({"type": "text", "text": text_prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    img_in, vid_in = process_vision_info(messages)
    inputs = processor(text=[text], images=img_in, videos=vid_in,
                       padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return processor.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE RUNNER — one question type, with resume
# ══════════════════════════════════════════════════════════════════════════════

INFERENCE_FIELDS = [
    "question_id", "video_file_path", "set", "qtype",
    "question", "generated_answer", "actual_answer", "kg_hint_used",
]


def run_qtype(qtype, rows, model, processor, max_q=None):
    out_path  = OVERALL_DIR / f"{qtype}_answers.csv"
    questions = rows[:max_q] if max_q else rows

    # Resume support
    done_ids = set()
    if out_path.exists():
        for r in csv.DictReader(open(out_path, encoding="utf-8")):
            done_ids.add(r["question_id"])
        logger.info(f"[{qtype}] Resuming — {len(done_ids)} already done.")

    todo = [q for q in questions if q["question_id"] not in done_ids]

    if todo:
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=INFERENCE_FIELDS)
            if not done_ids:
                writer.writeheader()

            for row in tqdm(todo, desc=f"[{qtype}] {QTYPE_LABELS[qtype]}"):
                set_name   = row["set"]
                clip_name  = Path(row["video_file_path"]).name
                video_path = find_video_path(set_name, clip_name)

                G        = build_clip_kg(video_path, set_name=set_name) if video_path else nx.Graph()
                kg_hint  = build_kg_prompt(G, row["question"]) if G.number_of_nodes() > 0 else None
                answer   = run_inference(model, processor, row["question"],
                                         video_path or "", kg_hint=kg_hint)

                writer.writerow({
                    "question_id":      row["question_id"],
                    "video_file_path":  row["video_file_path"],
                    "set":              set_name,
                    "qtype":            qtype,
                    "question":         row["question"],
                    "generated_answer": answer,
                    "actual_answer":    row["actual_answer"],
                    "kg_hint_used":     kg_hint is not None,
                })
                f.flush()

    all_rows = list(csv.DictReader(open(out_path, encoding="utf-8")))
    if max_q:
        all_rows = all_rows[:max_q]
    logger.info(f"[{qtype}] {len(all_rows)} rows → {out_path.name}")
    return all_rows


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ATTRIBUTION FROM EXISTING E006 CSVs
# ══════════════════════════════════════════════════════════════════════════════

def load_attribution_rows(max_q=None):
    """Load Attribution from existing graphrag_eval_v3 CSVs (answer_graphrag column)."""
    rows = []
    for s in MORNING_SETS:
        p = EVAL_DIR / f"E006_GraphRAGv3_{s}.csv"
        if not p.exists():
            logger.warning(f"[Atr] {p.name} not found — skipping {s}.")
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            rows.append({
                "question_id":      r.get("question_id", ""),
                "video_file_path":  r.get("video_id", ""),
                "set":              r.get("set", s),
                "qtype":            "Atr",
                "question":         r.get("question", ""),
                "generated_answer": r.get("answer_graphrag", ""),
                "actual_answer":    r.get("actual_answer", ""),
                "kg_hint_used":     r.get("kg_used", True),
            })
    if max_q:
        rows = rows[:max_q]
    logger.info(f"[Atr] Loaded {len(rows)} rows from E006 CSVs.")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# GROQ JUDGE
# ══════════════════════════════════════════════════════════════════════════════

JUDGE_PROMPT = """You are evaluating a traffic video question answering system.

Question: {question}
Ground Truth: {ground_truth}
Model Answer: {model_answer}

Scoring rules:
1. Color synonyms are CORRECT:
   - olive-gray / taupe / beige-grey / sandy-beige / stone / warm gray = same earth-tone
   - gray / dark gray / charcoal / slate = same gray family
2. "white" when GT is earth-tone = WRONG
3. Road markings: "keep-clear" / "boxed keep-clear" = CORRECT
4. Non-committal ("cannot determine", "unclear", "don't know") = WRONG
5. Extra detail around a correct core fact = CORRECT
6. Yes/No questions: must match polarity

Reply with exactly one word: CORRECT or WRONG"""


def judge_one(client, question, model_answer, ground_truth):
    prompt = JUDGE_PROMPT.format(
        question=question.strip(),
        ground_truth=ground_truth.strip(),
        model_answer=model_answer.strip(),
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            verdict = resp.choices[0].message.content.strip().upper()
            return 1 if "CORRECT" in verdict else 0
        except Exception as e:
            logger.warning(f"Groq error (attempt {attempt+1}): {e}")
            time.sleep(15)
    return 0


def judge_all(rows, judged_path):
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        logger.error("Set GROQ_API_KEY: export GROQ_API_KEY=gsk_...")
        sys.exit(1)
    from groq import Groq
    client = Groq(api_key=api_key)

    # ── Resume: load already-judged question_ids ───────────────────────────────
    done_ids = set()
    if judged_path.exists():
        for r in csv.DictReader(open(judged_path, encoding="utf-8")):
            done_ids.add(r["question_id"])
        logger.info(f"Resuming judging — {len(done_ids)} rows already done.")

    todo = [r for r in rows if r["question_id"] not in done_ids]
    logger.info(f"Rows to judge: {len(todo)} (total: {len(rows)})")

    if not todo:
        logger.info("All rows already judged. Loading from file.")
        return list(csv.DictReader(open(judged_path, encoding="utf-8")))

    # ── Open file in append mode so each row is saved immediately ──────────────
    first_row  = {**todo[0], "judge_score": 0}
    fieldnames = list(first_row.keys())
    write_header = not judged_path.exists() or len(done_ids) == 0

    with open(judged_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, row in enumerate(tqdm(todo, desc="Groq judging")):
            score = judge_one(client, row.get("question", ""),
                              row.get("generated_answer", ""),
                              row.get("actual_answer", ""))
            out_row = {**row, "judge_score": score}
            writer.writerow(out_row)
            f.flush()   # write immediately — safe to Ctrl+C at any point
            time.sleep(3)   # ~20 req/min → stays under 500K tokens/day limit

    logger.info(f"Judged {len(todo)} new rows → {judged_path}")
    return list(csv.DictReader(open(judged_path, encoding="utf-8")))



# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════

# UDVideoQA paper Table 3 — Morning, best open-source (Qwen2.5-32B)
PAPER_BEST = {"Atr": 36.11, "BU": 75.66, "ER": 66.67, "RR": 25.00, "CI": 77.78}
PAPER_OVERALL = 56.24   # Qwen2.5-32B morning overall


def print_results_table(results):
    stats = defaultdict(lambda: {"c": 0, "t": 0})
    for r in results:
        qt = r.get("qtype", "?")
        stats[qt]["c"] += int(r.get("judge_score", 0))
        stats[qt]["t"] += 1

    print("\n" + "=" * 74)
    print("  OVERALL RESULTS — Full KG v3 (Morning, 5 Sets)")
    print("  Judge: LLaMA-3.1-8b-instant via Groq")
    print("=" * 74)
    print(f"  {'Type':<6}  {'Category':<28}  {'Correct':>7}  {'Total':>5}  {'Acc':>6}")
    print("  " + "-" * 68)

    total_correct   = 0
    total_questions = 0
    tw_score = 0.0
    tw_total = 0.0
    accs = {}

    for qt in QTYPE_ORDER:
        if qt not in stats:
            print(f"  {qt:<6}  {QTYPE_LABELS.get(qt, qt):<28}  {'N/A':>7}")
            continue
        s   = stats[qt]
        acc = s["c"] / s["t"] * 100 if s["t"] > 0 else 0
        accs[qt] = acc
        w   = QTYPE_WEIGHTS.get(qt, 1.0)
        bar = "█" * int(acc / 3)
        print(f"  {qt:<6}  {QTYPE_LABELS.get(qt, qt):<28}  {s['c']:>7}  {s['t']:>5}"
              f"  {acc:>5.1f}%  {bar}")
        total_correct   += s["c"]
        total_questions += s["t"]
        tw_score += acc * w * s["t"]
        tw_total += w * s["t"]

    print("  " + "-" * 68)
    simple   = total_correct / total_questions * 100 if total_questions else 0
    weighted = tw_score / tw_total if tw_total else 0
    print(f"  {'TOTAL':<6}  {'(unweighted average)':<28}  {total_correct:>7}  "
          f"{total_questions:>5}  {simple:>5.1f}%")
    print(f"  {'TOTAL':<6}  {'(paper-weighted)':<28}  {'':>13}  {weighted:>5.1f}%")
    print("=" * 74)

    # vs. paper
    print("\n  vs. UDVideoQA paper best open-source (Qwen2.5-32B, morning):")
    print(f"  {'Type':<6}  {'Paper':>8}  {'Ours':>8}  {'Delta':>9}")
    print("  " + "-" * 40)
    for qt in QTYPE_ORDER:
        if qt in accs and qt in PAPER_BEST:
            d = accs[qt] - PAPER_BEST[qt]
            print(f"  {qt:<6}  {PAPER_BEST[qt]:>7.2f}%  {accs[qt]:>7.1f}%  "
                  f"{'+' if d>=0 else ''}{d:>7.1f}pp")
    if simple > 0:
        d = simple - PAPER_OVERALL
        print(f"  {'OVRL':<6}  {PAPER_OVERALL:>7.2f}%  {simple:>7.1f}%  "
              f"{'+' if d>=0 else ''}{d:>7.1f}pp")
    print("=" * 74)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Overall eval — all 5 question types, Full KG v3."
    )
    parser.add_argument("--skip_judge", action="store_true",
                        help="Run inference only; skip Groq judging.")
    parser.add_argument("--judge_only", action="store_true",
                        help="Skip inference; judge existing CSV files.")
    parser.add_argument("--max_q",     type=int, default=None,
                        help="Limit to first N questions per type (smoke-test).")
    parser.add_argument("--qtypes",    nargs="+", default=None,
                        help="Only run specific types e.g. --qtypes BU ER RR CI")
    args = parser.parse_args()

    target_qtypes = args.qtypes if args.qtypes else QTYPE_ORDER
    all_rows = []

    # ── Attribution — load from existing E006 CSVs ──────────────────────────────
    if "Atr" in target_qtypes:
        logger.info("\n[Atr] Loading from existing E006_GraphRAGv3_*.csv files...")
        atr_rows = load_attribution_rows(max_q=args.max_q)
        all_rows.extend(atr_rows)

    # ── BU / ER / RR / CI — fresh inference ────────────────────────────────────
    non_atr = [qt for qt in target_qtypes if qt != "Atr"]

    if non_atr and not args.judge_only:
        logger.info(f"\nLoading JSONL questions for: {non_atr}")
        questions = load_morning_questions(target_qtypes=non_atr)

        if not any(questions.values()):
            logger.error("No questions found. Check ANN_DIR and MORNING_SETS.")
            sys.exit(1)

        logger.info("\nLoading Qwen2.5-VL-3B + LoRA FT adapter...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from peft import PeftModel

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="eager",
        )
        torch.cuda.empty_cache()   # clear fragmented memory before adapter load
        model = PeftModel.from_pretrained(model, FT_ADAPTER)
        model = model.merge_and_unload()
        model.eval()
        processor = AutoProcessor.from_pretrained(
            BASE_MODEL, trust_remote_code=True,
            min_pixels=128*28*28, max_pixels=360*28*28,
        )
        logger.info(f"Model ready. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

        for qt in non_atr:
            rows = questions.get(qt, [])
            if not rows:
                logger.warning(f"[{qt}] No questions found — skipping.")
                continue
            logger.info(f"\n{'='*60}\n[{qt}] {QTYPE_LABELS[qt]} — {len(rows)} questions")
            qt_rows = run_qtype(qt, rows, model, processor, max_q=args.max_q)
            all_rows.extend(qt_rows)

    elif non_atr and args.judge_only:
        for qt in non_atr:
            p = OVERALL_DIR / f"{qt}_answers.csv"
            if p.exists():
                rows = list(csv.DictReader(open(p, encoding="utf-8")))
                if args.max_q: rows = rows[:args.max_q]
                all_rows.extend(rows)
                logger.info(f"[{qt}] Loaded {len(rows)} rows from {p.name}.")
            else:
                logger.warning(f"[{qt}] {p.name} not found — skipping.")

    if not all_rows:
        logger.error("No rows collected. Check inference has been run first.")
        sys.exit(1)

    # ── Save merged CSV ─────────────────────────────────────────────────────────
    merged_path = OVERALL_DIR / "overall_all_answers.csv"
    with open(merged_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    logger.info(f"\nMerged CSV ({len(all_rows)} rows) → {merged_path}")
    counts = defaultdict(int)
    for r in all_rows: counts[r["qtype"]] += 1
    for qt in QTYPE_ORDER:
        logger.info(f"  {qt}: {counts.get(qt, 0)} rows")

    if args.skip_judge:
        logger.info("\n--skip_judge set. To judge:")
        logger.info("  python src/eval/overall_eval.py --judge_only")
        return

    # ── Groq judge ──────────────────────────────────────────────────────────────
    n = len(all_rows)
    logger.info(f"\nGroq judging {n} rows (~{n*2//60} min at 30 req/min)...")
    judged_path = OVERALL_DIR / "overall_judged.csv"
    results = judge_all(all_rows, judged_path)

    print_results_table(results)
    logger.info(f"Full results → {judged_path}")


if __name__ == "__main__":
    main()
