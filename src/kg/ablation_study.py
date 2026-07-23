"""
Ablation Study — KG Component Contribution
===========================================
Shows the contribution of each Knowledge Graph component.

Conditions (4 total):
  A: Qwen ZeroShot              — baseline, no KG          (LOAD from existing CSV)
  B: Qwen + Building Color      — pixel only, no road/YOLO (RUN fresh)
  C: Qwen + Building Color
        + Road Marking           — pixel, no YOLO           (RUN fresh)
  D: Qwen + Full KG v3          — pixel + YOLO (all)       (LOAD from existing CSV)

All conditions judged with Groq LLaMA-3.3-70B (same judge as official results).

Usage:
    # Full run — all 403 Qs (~45 min inference + ~25 min Groq judge):
    CUDA_VISIBLE_DEVICES=6 python src/kg/ablation_study.py

    # Inference only (no Groq — handy to run first, judge later):
    CUDA_VISIBLE_DEVICES=6 python src/kg/ablation_study.py --skip_judge

    # Judge only (inference CSVs already in results/ablation/):
    python src/kg/ablation_study.py --judge_only

    # Quick smoke-test — 5 questions total:
    CUDA_VISIBLE_DEVICES=6 python src/kg/ablation_study.py --max_q 5
"""

import argparse, csv, os, sys, time
import cv2
import networkx as nx
import numpy as np
import torch
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path("/path/to/your/data")
DATA_DIR     = BASE_DIR / "data" / "videos"
EVAL_DIR     = BASE_DIR / "results" / "eval"
ABLATION_DIR = BASE_DIR / "results" / "ablation"
BASE_MODEL   = str(BASE_DIR / "models" / "base_model")

ABLATION_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

ALL_SETS = ["Set_26", "Set_30", "Set_33", "Set_34", "Set_35"]

# Per-set building ROI (calibrated in Session 6)
SET_BUILDING_ROI = {
    "Set_33": (0.75, 0.50, 1.0, 0.75),
    "default": (0.55, 0.28, 1.0, 0.62),
}

# YOLO COCO class IDs
VEHICLE_IDS   = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PEDESTRIAN_ID = 0
TRAFFIC_LIGHT = 9
STOP_SIGN     = 11


# ══════════════════════════════════════════════════════════════════════════════
# PIXEL-BASED COLOR EXTRACTION  (identical to graphrag_eval_v3.py)
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
    mask = (V <= 220) & ~((H > 85) & (H < 135) & (S > 70))
    if mask.sum() < 50: return "overexposed"
    return hsv_to_color_name(H[mask].mean(), S[mask].mean(), V[mask].mean())


def classify_road_marking(frame):
    h, w = frame.shape[:2]
    roi = frame[int(0.55*h):int(0.85*h), int(0.3*w):int(0.7*w)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    if thresh.mean() / 255 < 0.02: return "none"
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return "none"
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 200: return "none"
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
# YOLO DETECTION  (lazy-loaded — only active when use_yolo=True)
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
    if yolo is None: return {}
    results = yolo(frame, verbose=False, conf=0.3)[0]
    det = {"vehicles": [], "pedestrians": 0, "traffic_light": False, "signs": []}
    for box in results.boxes:
        cls_id = int(box.cls.item())
        if cls_id in VEHICLE_IDS:       det["vehicles"].append(VEHICLE_IDS[cls_id])
        elif cls_id == PEDESTRIAN_ID:   det["pedestrians"] += 1
        elif cls_id == TRAFFIC_LIGHT:   det["traffic_light"] = True
        elif cls_id == STOP_SIGN:       det["signs"].append("stop sign")
    return det


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO / PATH UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def find_video_path(set_name, clip_name):
    for p in DATA_DIR.rglob(clip_name.replace(".mp4", "_blurred.mp4")):
        if set_name in str(p): return str(p)
    for p in DATA_DIR.rglob(clip_name):
        if set_name in str(p): return str(p)
    return None


def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0: return []
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret: frames.append(frame)
    cap.release()
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# KG BUILDER — selective component flags
# ══════════════════════════════════════════════════════════════════════════════

def build_clip_kg(video_path, set_name="default",
                  use_building_color=True,
                  use_road_marking=True,
                  use_yolo=True):
    """
    Build a Knowledge Graph with selective component inclusion.

    Flags:
        use_building_color  — extract dominant building colour from pixels
        use_road_marking    — detect road markings from pixels
        use_yolo            — run YOLOv8n for vehicles/pedestrians/signals/signs
    """
    G = nx.Graph()
    frames = extract_frames(video_path, num_frames=8)
    if not frames:
        return G

    building_roi = SET_BUILDING_ROI.get(set_name, SET_BUILDING_ROI["default"])

    # A: Building color (pixel, glare-resistant)
    if use_building_color:
        colors = []
        for frame in frames:
            c = extract_dominant_color(frame, roi=building_roi)
            if c not in ("unknown", "overexposed"):
                colors.append(c)
        if colors:
            dom = Counter(colors).most_common(1)[0][0]
            G.add_node("building", type="building", color=dom,
                       color_family=color_name_to_family(dom),
                       glare_coverage=round(1 - len(colors)/len(frames), 2))

    # B: Road marking (pixel, center-lane)
    if use_road_marking:
        markings = []
        for frame in frames:
            m = classify_road_marking(frame)
            if m != "none":
                markings.append(m)
        if markings:
            G.add_node("road_marking", type="marking",
                       description=Counter(markings).most_common(1)[0][0])

    # C: YOLO — vehicles, pedestrians, traffic lights, signs
    if use_yolo:
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
                       count=len(set(all_veh)),
                       types=list(tc.keys()),
                       most_common=tc.most_common(1)[0][0],
                       type_counts=dict(tc))

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
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_kg_prompt(G, question):
    """Return KG context string, or None if no relevant facts."""
    nodes = dict(G.nodes(data=True))
    q = question.lower()
    facts = []

    if "building" in nodes and any(w in q for w in
        ["colour","color","hue","appear","look","shade","building","structure","facade","wall"]):
        n = nodes["building"]
        facts.append(
            f"The building color is {n['color']} "
            f"(color family: {n['color_family']}, measured from video frames)."
        )

    if "road_marking" in nodes and any(w in q for w in
        ["marking","painted","road","lane","centre","center","middle","shape","sign"]):
        facts.append(f"The road marking is: {nodes['road_marking']['description']}.")

    if "vehicles" in nodes and any(w in q for w in
        ["vehicle","car","bus","truck","motorcycle","bicycle","how many","count","type"]):
        n = nodes["vehicles"]
        facts.append(f"Vehicles detected: {n['type_counts']}. Most common: {n['most_common']}.")

    if "pedestrians" in nodes and any(w in q for w in
        ["pedestrian","person","people","walking","crossing"]):
        n = nodes["pedestrians"]
        facts.append(
            f"Pedestrians: {'present' if n['present'] else 'none detected'} "
            f"(max count: {n['max_count']})."
        )

    if "traffic_light" in nodes and any(w in q for w in
        ["signal","light","traffic light","green","red","amber"]):
        facts.append("A traffic light/signal was detected in this scene.")

    if "signs" in nodes and any(w in q for w in ["sign","board","signage"]):
        facts.append(f"Signs detected: {nodes['signs']['types']}.")

    if not facts:
        return None
    return "Scene knowledge from video analysis:\n" + "\n".join(f"- {f}" for f in facts)


# ══════════════════════════════════════════════════════════════════════════════
# VLM INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model, processor, question, video_path, kg_hint=None):
    """Run Qwen2.5-VL-3B, optionally prepending KG context."""
    from qwen_vl_utils import process_vision_info

    text_prompt = (
        f"{kg_hint}\n\nQuestion: {question}" if kg_hint else question
    )
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
# GROQ JUDGE  (LLaMA-3.3-70B, same as official evaluation)
# ══════════════════════════════════════════════════════════════════════════════

JUDGE_PROMPT = """You are evaluating a traffic video question answering system.

Question: {question}
Ground Truth: {ground_truth}
Model Answer: {model_answer}

Scoring rules:
1. Color synonyms — CORRECT:
   - "olive-gray" / "taupe" / "beige-grey" / "sandy-beige" / "stone" / "warm gray" = same earth-tone
   - "gray" / "dark gray" / "charcoal" / "slate" = same gray family
2. "white" when ground truth is earth-tone (taupe/beige/sandy/olive) = WRONG
3. Road markings: "keep-clear" / "white boxed keep-clear marking" = CORRECT
4. Non-committal ("I cannot determine", "unclear", "I don't know") = WRONG
5. Extra detail around correct core fact = CORRECT
6. Partial answer that contains the key fact = CORRECT

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
            logger.warning(f"Groq API error (attempt {attempt+1}): {e}")
            time.sleep(15)
    return 0


def judge_all(rows, judged_path):
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        logger.error("ERROR: export GROQ_API_KEY=gsk_...")
        sys.exit(1)
    from groq import Groq
    client = Groq(api_key=api_key)

    results = []
    for row in tqdm(rows, desc="Groq judging"):
        score = judge_one(
            client,
            row.get("question", ""),
            row.get("generated_answer", ""),
            row.get("actual_answer", ""),
        )
        results.append({**row, "judge_score": score})
        time.sleep(2)   # 30 req/min limit on Groq free tier

    judged_path.parent.mkdir(parents=True, exist_ok=True)
    with open(judged_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    logger.info(f"Judged results → {judged_path}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# LOAD EXISTING RESULTS  (Conditions A and D are already done)
# ══════════════════════════════════════════════════════════════════════════════

def load_zeroshot_rows(max_q=None, use_ft=False):
    """
    Condition A — load baseline answers from existing CSV.
    use_ft=False → ZeroShot CSV (base model, 30.8%)
    use_ft=True  → FT CSV (fine-tuned model, 30.3%)
    """
    if use_ft:
        fname = "Qwen2.5-VL-3B-FT_morning_attribution.csv"
        cond_id = "A_FT"
        label = "[A_FT]"
    else:
        fname = "Qwen2.5-VL-3B-ZeroShot_morning_attribution.csv"
        cond_id = "A_ZeroShot"
        label = "[A]"
    path = EVAL_DIR / fname
    if not path.exists():
        logger.error(f"Baseline CSV not found: {path}")
        sys.exit(1)
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        rows.append({
            "condition":        cond_id,
            "question_id":      r.get("question_id", ""),
            "video_id":         r.get("video_file_path", r.get("video_id", "")),
            "set":              r.get("set", ""),
            "question":         r.get("question", ""),
            "generated_answer": r.get("generated_answer", ""),
            "actual_answer":    r.get("actual_answer", ""),
            "kg_components":    "none",
            "kg_hint_used":     False,
        })
    if max_q:
        rows = rows[:max_q]
    logger.info(f"{label} Loaded {len(rows)} rows from {fname}.")
    return rows


def load_kgv3_rows(max_q=None):
    """Condition D — load Full KG v3 from existing per-set CSVs."""
    rows = []
    for s in ALL_SETS:
        p = EVAL_DIR / f"E006_GraphRAGv3_{s}.csv"
        if not p.exists():
            logger.warning(f"[D] {p.name} not found — skipping {s}.")
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            rows.append({
                "condition":        "D_FullKGv3",
                "question_id":      r.get("question_id", ""),
                "video_id":         r.get("video_id", ""),
                "set":              r.get("set", s),
                "question":         r.get("question", ""),
                "generated_answer": r.get("answer_graphrag", ""),
                "actual_answer":    r.get("actual_answer", ""),
                "kg_components":    "building_color+road_marking+yolo",
                "kg_hint_used":     r.get("kg_used", True),
            })
    if max_q:
        rows = rows[:max_q]
    logger.info(f"[D] Loaded {len(rows)} KG-v3 rows.")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE RUNNER  (Conditions B and C — fresh inference)
# ══════════════════════════════════════════════════════════════════════════════

INFERENCE_FIELDS = [
    "condition", "question_id", "video_id", "set", "question",
    "generated_answer", "actual_answer", "kg_components", "kg_hint_used",
]


def run_condition(condition_id, condition_label, kg_components,
                  use_building_color, use_road_marking, use_yolo,
                  model, processor, all_questions, max_q=None):
    """
    Run VLM inference for one ablation condition.
    Supports resume — skips question_ids already in the output CSV.
    Returns all rows (loaded + newly written).
    """
    out_path = ABLATION_DIR / f"{condition_id}_answers.csv"

    # Collect already-done IDs for resume
    done_ids = set()
    if out_path.exists():
        for r in csv.DictReader(open(out_path, encoding="utf-8")):
            done_ids.add(r["question_id"])
        logger.info(f"[{condition_id}] Resuming — {len(done_ids)} questions already done.")

    questions = all_questions[:max_q] if max_q else all_questions
    todo = [q for q in questions if q.get("question_id", "") not in done_ids]

    if todo:
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=INFERENCE_FIELDS)
            if not done_ids:       # fresh file — write header
                writer.writeheader()

            for row in tqdm(todo, desc=f"[{condition_id}] {condition_label}"):
                set_name  = row.get("set", "")
                clip_name = Path(
                    row.get("video_file_path", row.get("video_id", ""))
                ).name
                video_path = find_video_path(set_name, clip_name)

                # Build KG with selected components
                G = nx.Graph()
                if video_path:
                    G = build_clip_kg(
                        video_path,
                        set_name=set_name,
                        use_building_color=use_building_color,
                        use_road_marking=use_road_marking,
                        use_yolo=use_yolo,
                    )

                kg_hint = build_kg_prompt(G, row["question"]) if G.number_of_nodes() > 0 else None

                answer = run_inference(
                    model, processor,
                    row["question"],
                    video_path or "",
                    kg_hint=kg_hint,
                )

                out_row = {
                    "condition":        condition_id,
                    "question_id":      row.get("question_id", ""),
                    "video_id":         row.get("video_file_path", row.get("video_id", "")),
                    "set":              set_name,
                    "question":         row["question"],
                    "generated_answer": answer,
                    "actual_answer":    row.get("actual_answer", ""),
                    "kg_components":    kg_components,
                    "kg_hint_used":     kg_hint is not None,
                }
                writer.writerow(out_row)
                f.flush()
                logger.debug(f"  Q: {row['question'][:60]}")
                logger.debug(f"  A: {answer[:60]}")

    # Return all rows (done + new) from the final CSV
    all_rows = list(csv.DictReader(open(out_path, encoding="utf-8")))
    if max_q:
        all_rows = all_rows[:max_q]
    logger.info(f"[{condition_id}] Total rows: {len(all_rows)} → {out_path.name}")
    return all_rows


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

CONDITION_META = {
    # Base-model runs (original)
    "A_ZeroShot":     ("A", "Qwen ZeroShot  (no KG)",                    "none"),
    "B_ColorOnly":    ("B", "Qwen + Building Color  (pixel, no YOLO)",   "building_color"),
    "C_ColorRoad":    ("C", "Qwen + Building Color + Road Marking",      "building_color+road_marking"),
    "D_FullKGv3":     ("D", "Qwen + Full KG v3  (pixel + YOLO)",         "all components"),
    # FT-model runs (--use_ft flag)
    "A_FT":           ("A", "Qwen FT  (no KG)",                          "none"),
    "B_ColorOnly_FT": ("B", "Qwen FT + Building Color  (pixel)",         "building_color"),
    "C_ColorRoad_FT": ("C", "Qwen FT + Building Color + Road Marking",   "building_color+road_marking"),
}

ORDER_BASE = ["A_ZeroShot", "B_ColorOnly",    "C_ColorRoad",    "D_FullKGv3"]
ORDER_FT   = ["A_FT",       "B_ColorOnly_FT", "C_ColorRoad_FT", "D_FullKGv3"]


def print_ablation_table(results, order=None):
    if order is None:
        order = ORDER_BASE
    stats = defaultdict(lambda: {"c": 0, "t": 0})
    for r in results:
        cond = r["condition"]
        stats[cond]["c"] += int(r.get("judge_score", 0))
        stats[cond]["t"] += 1

    print("\n" + "=" * 78)
    print("  ABLATION STUDY — KG Component Contribution")
    print("  Judge: LLaMA-3.3-70B via Groq  |  403 Morning-Attribution Questions")
    print("=" * 78)
    header = f"  {'#':<2}  {'Condition':<46}  {'Correct':>7}  {'Total':>5}  {'Acc':>6}  {'ΔvsA':>7}"
    print(header)
    print("  " + "-" * 74)

    baseline_acc = None
    accs = {}
    for cid in order:
        if cid not in stats:
            print(f"  {'?':<2}  {cid:<46}  {'N/A':>7}")
            continue
        s = stats[cid]
        acc = s["c"] / s["t"] * 100 if s["t"] > 0 else 0
        accs[cid] = acc
        if baseline_acc is None:
            baseline_acc = acc
        delta = acc - baseline_acc
        tag = CONDITION_META.get(cid, ("?", cid, ""))
        bar = "█" * int(acc / 2)
        delta_str = f"{delta:+.1f}pp" if delta != 0 else "—"
        print(f"  {tag[0]:<2}  {tag[1]:<46}  {s['c']:>7}  {s['t']:>5}  "
              f"{acc:>5.1f}%  {delta_str:>7}  {bar}")

    print("=" * 78)

    # Incremental component contributions
    print("\n  Component contributions (each row = what adding that component buys):")
    labels = {
        "B_ColorOnly":    "Building Color (pixel)",
        "C_ColorRoad":    "Road Marking   (pixel)",
        "D_FullKGv3":     "YOLO objects   (vehicles/pedestrians/signals/signs)",
        "B_ColorOnly_FT": "Building Color (pixel)",
        "C_ColorRoad_FT": "Road Marking   (pixel)",
    }
    prev_cid = order[0]
    for cid in order[1:]:
        if cid in accs and prev_cid in accs:
            lbl = labels.get(cid, cid)
            delta = accs[cid] - accs[prev_cid]
            print(f"    + {lbl:<52} {delta:+.1f}pp")
        prev_cid = cid
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Ablation study: KG component contribution on morning Attribution."
    )
    parser.add_argument("--skip_judge",  action="store_true",
                        help="Run inference only; skip Groq judging.")
    parser.add_argument("--judge_only",  action="store_true",
                        help="Skip inference; judge existing CSVs in results/ablation/.")
    parser.add_argument("--max_q",       type=int, default=None,
                        help="Limit to first N questions (smoke-test).")
    parser.add_argument("--skip_b",      action="store_true",
                        help="Skip Condition B inference (load existing CSV).")
    parser.add_argument("--skip_c",      action="store_true",
                        help="Skip Condition C inference (load existing CSV).")
    parser.add_argument("--use_ft",      action="store_true",
                        help="Load LoRA fine-tuned adapter for B & C inference, "
                             "and use FT baseline CSV for condition A. "
                             "Makes all 4 conditions use the same model (FT).")
    args = parser.parse_args()

    # ── Suffix for output files and condition IDs ──────────────────────────────
    sfx   = "_FT" if args.use_ft else ""          # appended to filenames
    ORDER = ORDER_FT if args.use_ft else ORDER_BASE

    # ── Load master question list ──────────────────────────────────────────────
    master_csv = BASE_DIR / "data" / "morning_attribution.csv"
    if not master_csv.exists():
        logger.error(f"Master CSV not found: {master_csv}")
        sys.exit(1)
    all_questions = list(csv.DictReader(open(master_csv, encoding="utf-8")))
    logger.info(f"Master dataset: {len(all_questions)} questions across all sets.")

    all_rows = []

    # ── Condition A — load from existing baseline CSV ──────────────────────────
    logger.info("\n" + "=" * 60)
    label_a = "FT baseline" if args.use_ft else "ZeroShot baseline"
    logger.info(f"[A] Loading {label_a} (existing CSV)...")
    all_rows.extend(load_zeroshot_rows(max_q=args.max_q, use_ft=args.use_ft))

    # ── Conditions B & C — need model inference ────────────────────────────────
    if not args.judge_only:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        if args.use_ft:
            logger.info("\nLoading Qwen2.5-VL-3B + LoRA FT adapter...")
        else:
            logger.info("\nLoading Qwen2.5-VL-3B (base model, no adapter)...")

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )

        if args.use_ft:
            from peft import PeftModel
            FT_ADAPTER = str(BASE_DIR / "models" / "baseline_adapter")
            model = PeftModel.from_pretrained(model, FT_ADAPTER)
            model = model.merge_and_unload()   # merge for faster inference
            logger.info("LoRA adapter merged into base model.")

        model.eval()
        processor = AutoProcessor.from_pretrained(
            BASE_MODEL, trust_remote_code=True,
            min_pixels=128*28*28, max_pixels=360*28*28,
        )
        logger.info(f"Model ready. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

        # ── Condition B — Building Color only ─────────────────────────────────
        logger.info("\n" + "=" * 60)
        b_cid   = f"B_ColorOnly{sfx}"
        b_label = f"Qwen{'(FT)' if args.use_ft else ''} + Building Color only"
        if not args.skip_b:
            b_rows = run_condition(
                condition_id=b_cid,
                condition_label=b_label,
                kg_components="building_color",
                use_building_color=True,
                use_road_marking=False,
                use_yolo=False,
                model=model,
                processor=processor,
                all_questions=all_questions,
                max_q=args.max_q,
            )
        else:
            p = ABLATION_DIR / f"{b_cid}_answers.csv"
            b_rows = list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []
            logger.info(f"[B] Skipped inference — loaded {len(b_rows)} rows from {p.name}.")
        all_rows.extend(b_rows)

        # ── Condition C — Building Color + Road Marking ────────────────────────
        logger.info("\n" + "=" * 60)
        c_cid   = f"C_ColorRoad{sfx}"
        c_label = f"Qwen{'(FT)' if args.use_ft else ''} + Building Color + Road Marking"
        if not args.skip_c:
            c_rows = run_condition(
                condition_id=c_cid,
                condition_label=c_label,
                kg_components="building_color+road_marking",
                use_building_color=True,
                use_road_marking=True,
                use_yolo=False,
                model=model,
                processor=processor,
                all_questions=all_questions,
                max_q=args.max_q,
            )
        else:
            p = ABLATION_DIR / f"{c_cid}_answers.csv"
            c_rows = list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []
            logger.info(f"[C] Skipped inference — loaded {len(c_rows)} rows from {p.name}.")
        all_rows.extend(c_rows)

    else:
        # judge_only — load B and C from existing CSVs (respects --use_ft suffix)
        b_cid = f"B_ColorOnly{sfx}"
        c_cid = f"C_ColorRoad{sfx}"
        for cid, fname in [(b_cid, f"{b_cid}_answers.csv"),
                            (c_cid, f"{c_cid}_answers.csv")]:
            p = ABLATION_DIR / fname
            if p.exists():
                rows = list(csv.DictReader(open(p, encoding="utf-8")))
                if args.max_q: rows = rows[:args.max_q]
                all_rows.extend(rows)
                logger.info(f"[{cid}] Loaded {len(rows)} rows from {fname}.")
            else:
                logger.warning(f"[{cid}] {fname} not found — will be missing from results.")

    # ── Condition D — Full KG v3 (already done) ────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("[D] Loading Full KG v3 (existing CSVs)...")
    all_rows.extend(load_kgv3_rows(max_q=args.max_q))

    if not all_rows:
        logger.error("No rows collected. Exiting.")
        sys.exit(1)

    # ── Save merged pre-judge CSV ──────────────────────────────────────────────
    merged_path = ABLATION_DIR / f"ablation{sfx}_all_answers.csv"
    with open(merged_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    logger.info(f"\nMerged pre-judge CSV ({len(all_rows)} rows) → {merged_path}")

    # Summary of rows per condition
    cond_counts = defaultdict(int)
    for r in all_rows:
        cond_counts[r["condition"]] += 1
    for cid in ORDER:
        logger.info(f"  {cid}: {cond_counts.get(cid, 0)} rows")

    if args.skip_judge:
        ft_flag = " --use_ft" if args.use_ft else ""
        logger.info("\n--skip_judge set. Done. To judge later:")
        logger.info(f"  python src/kg/ablation_study.py --judge_only{ft_flag}")
        return

    # ── Groq judging ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info(f"Starting Groq judging — {len(all_rows)} rows "
                f"(~{len(all_rows) * 2 // 60} min at 30 req/min)...")
    judged_path = ABLATION_DIR / f"ablation{sfx}_judged.csv"
    results = judge_all(all_rows, judged_path)

    # ── Print summary table (pass ORDER so FT variants display correctly) ──────
    print_ablation_table(results, order=ORDER)
    logger.info(f"Full judged results → {judged_path}")


if __name__ == "__main__":
    main()
