"""
E003 — GraphRAG: Knowledge Graph + Retrieval Augmented Generation
=================================================================
Approach:
  1. For each failing question, extract frames from the video
  2. Use the VLM itself to describe scene attributes (colors, objects, markings)
  3. Build a per-clip Knowledge Graph (NetworkX) from descriptions
  4. Query the KG for relevant facts based on the question
  5. Inject retrieved facts into the prompt → re-run model
  6. Compare: does KG-augmented answer beat original?

Target: Morning Attribution failures (Pattern 1 — color confusion from glare)
Expected: Building color questions especially should improve dramatically.

Usage:
    python src/kg/graphrag_eval.py --max_clips 10
    python src/kg/graphrag_eval.py --set Set_34 --max_clips 50
"""

import os
import re
import csv
import cv2
import json
import argparse
import networkx as nx
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm
from loguru import logger

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR    = "/path/to/your/data"
VIDEO_DIR   = f"{BASE_DIR}/data/videos"
BASE_MODEL  = f"{BASE_DIR}/models/base_model"
ADAPTER_DIR = f"{BASE_DIR}/models/baseline_adapter"
RESULTS_DIR = f"{BASE_DIR}/results"
KG_DIR      = f"{RESULTS_DIR}/kg_cache"   # Cache KGs to avoid recomputing
EVAL_DIR    = f"{RESULTS_DIR}/eval"
ANALYSIS_DIR= f"{RESULTS_DIR}/analysis"

os.makedirs(KG_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/eval", exist_ok=True)


# ── Video → Frames ────────────────────────────────────────────────
def extract_frames(video_path: str, num_frames: int = 5) -> list:
    """Extract N evenly-spaced frames from a video. Returns list of numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    indices = [int(i * total / num_frames) for i in range(num_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def frames_to_pil(frames: list) -> list:
    """Convert OpenCV BGR frames to PIL images."""
    from PIL import Image
    pil_frames = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(rgb))
    return pil_frames


# ── VLM Scene Description ─────────────────────────────────────────
DESCRIPTION_PROMPT = """You are analyzing a traffic scene image.
List the following attributes you can observe. Be specific and concise.
Format your response EXACTLY as:
BUILDING_COLOR: <color of prominent building if visible, else 'none'>
ROAD_MARKING: <type of road marking in center lane, e.g. 'white X', 'white line', 'keep clear box', 'none'>
VEHICLE_COLORS: <list of vehicle colors visible, e.g. 'red sedan, blue truck, silver car'>
TRAFFIC_SIGNAL: <state of traffic light if visible: 'red', 'green', 'amber', 'none'>
OTHER_MARKINGS: <any other notable markings or signs visible>"""


def describe_frame(model, processor, pil_image) -> dict:
    """Ask VLM to describe scene attributes in a single frame."""
    import tempfile
    from PIL import Image

    # Save to temp file (VLM needs file path or PIL image)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir="/home/anwar/tmp") as tmp:
        pil_image.save(tmp.name)
        tmp_path = tmp.name

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": tmp_path},
            {"type": "text",  "text": DESCRIPTION_PROMPT},
        ]
    }]

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=150, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

        os.unlink(tmp_path)
        return parse_description(response)

    except Exception as e:
        logger.warning(f"Frame description failed: {e}")
        os.unlink(tmp_path) if os.path.exists(tmp_path) else None
        return {}


def parse_description(text: str) -> dict:
    """Parse the structured VLM description into a dict."""
    fields = {
        "building_color":  r"BUILDING_COLOR:\s*(.+)",
        "road_marking":    r"ROAD_MARKING:\s*(.+)",
        "vehicle_colors":  r"VEHICLE_COLORS:\s*(.+)",
        "traffic_signal":  r"TRAFFIC_SIGNAL:\s*(.+)",
        "other_markings":  r"OTHER_MARKINGS:\s*(.+)",
    }
    result = {}
    for key, pattern in fields.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if val.lower() not in ("none", "n/a", "unknown", ""):
                result[key] = val
    return result


# ── Build Knowledge Graph ─────────────────────────────────────────
def build_clip_kg(descriptions: list) -> nx.Graph:
    """
    Build a Knowledge Graph from a list of per-frame descriptions.
    Aggregates consistent attributes across frames (majority vote).
    """
    G = nx.Graph()

    # Aggregate attributes across frames
    agg = defaultdict(list)
    for desc in descriptions:
        for key, val in desc.items():
            if val:
                agg[key].append(val.lower().strip())

    # Add nodes with most common value for each attribute
    if agg["building_color"]:
        # Pick most common (majority vote)
        color = max(set(agg["building_color"]), key=agg["building_color"].count)
        if "none" not in color:
            G.add_node("building", type="building", color=color,
                       confidence=agg["building_color"].count(color)/len(agg["building_color"]))

    if agg["road_marking"]:
        marking = max(set(agg["road_marking"]), key=agg["road_marking"].count)
        if "none" not in marking:
            G.add_node("road_marking", type="marking", description=marking,
                       confidence=agg["road_marking"].count(marking)/len(agg["road_marking"]))

    if agg["traffic_signal"]:
        state = max(set(agg["traffic_signal"]), key=agg["traffic_signal"].count)
        if "none" not in state:
            G.add_node("signal", type="traffic_signal", state=state)

    # Vehicle colors → individual vehicle nodes
    all_vehicles = []
    for vc_str in agg["vehicle_colors"]:
        # "red sedan, blue truck" → ["red sedan", "blue truck"]
        vehicles = [v.strip() for v in vc_str.split(",") if v.strip() and "none" not in v]
        all_vehicles.extend(vehicles)

    if all_vehicles:
        # Count unique vehicle descriptions
        vc_counts = defaultdict(int)
        for v in all_vehicles:
            vc_counts[v] += 1
        for i, (vdesc, count) in enumerate(sorted(vc_counts.items(), key=lambda x: -x[1])[:5]):
            parts = vdesc.split()
            color = parts[0] if parts else "unknown"
            vtype = parts[1] if len(parts) > 1 else "vehicle"
            G.add_node(f"vehicle_{i}", type="vehicle", color=color,
                       description=vdesc, confidence=count/len(descriptions))

    return G


def save_kg(G: nx.Graph, clip_id: str):
    """Save KG as JSON for caching."""
    data = nx.node_link_data(G)
    path = f"{KG_DIR}/{clip_id.replace('/', '_')}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_kg(clip_id: str) -> nx.Graph | None:
    """Load cached KG if it exists."""
    path = f"{KG_DIR}/{clip_id.replace('/', '_')}.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return nx.node_link_graph(data)
    return None


# ── KG Query → Prompt Injection ───────────────────────────────────
def query_kg_for_question(G: nx.Graph, question: str) -> str:
    """
    Given a question, retrieve relevant facts from KG.
    Returns a string of facts to inject into the prompt.
    """
    q = question.lower()
    facts = []

    # Building color questions
    if any(w in q for w in ["building", "structure", "facade"]) and \
       any(w in q for w in ["colour", "color", "appear"]):
        if G.has_node("building"):
            color = G.nodes["building"].get("color", "")
            conf  = G.nodes["building"].get("confidence", 0)
            if color and conf > 0.3:
                facts.append(f"The prominent building in this scene is {color} in color.")

    # Road marking questions
    if any(w in q for w in ["marking", "painted", "marking", "sign", "lane"]):
        if G.has_node("road_marking"):
            desc = G.nodes["road_marking"].get("description", "")
            if desc:
                facts.append(f"The road marking visible in this scene is: {desc}.")

    # Traffic signal questions
    if any(w in q for w in ["signal", "light", "traffic light", "red", "green"]):
        if G.has_node("signal"):
            state = G.nodes["signal"].get("state", "")
            if state:
                facts.append(f"The traffic signal in this scene shows: {state}.")

    # Vehicle color questions
    if any(w in q for w in ["vehicle", "car", "truck", "bus", "colour", "color"]):
        vehicles = [n for n, d in G.nodes(data=True) if d.get("type") == "vehicle"]
        if vehicles:
            descs = [G.nodes[v].get("description", "") for v in vehicles[:3]]
            descs = [d for d in descs if d]
            if descs:
                facts.append(f"Vehicles visible in this scene: {', '.join(descs)}.")

    return " ".join(facts) if facts else ""


# ── GraphRAG Inference ────────────────────────────────────────────
def run_graphrag_inference(model, processor, question: str,
                           video_path: str, kg: nx.Graph) -> tuple:
    """
    Run inference with KG facts injected into prompt.
    Returns (answer_without_kg, answer_with_kg)
    """
    from qwen_vl_utils import process_vision_info

    kg_facts = query_kg_for_question(kg, question)

    def build_prompt(with_kg: bool) -> str:
        if with_kg and kg_facts:
            return (
                f"You are analyzing a traffic video clip.\n"
                f"Background facts observed from this scene:\n{kg_facts}\n\n"
                f"Using both the video and these facts, answer concisely:\n"
                f"Question: {question}\n\nAnswer:"
            )
        else:
            return (
                f"Watch the traffic video carefully and answer the following question concisely.\n\n"
                f"Question: {question}\n\nAnswer:"
            )

    results = {}
    for use_kg in [False, True]:
        content = []
        if video_path and os.path.exists(video_path):
            content.append({
                "type": "video",
                "video": video_path,
                "max_pixels": 360 * 28 * 28,
                "fps": 1.0,
            })
        content.append({"type": "text", "text": build_prompt(use_kg)})

        messages = [{"role": "user", "content": content}]
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)

            in_len = inputs["input_ids"].shape[1]
            answer = processor.decode(output_ids[0][in_len:], skip_special_tokens=True).strip()
            results["with_kg" if use_kg else "without_kg"] = answer

        except Exception as e:
            results["with_kg" if use_kg else "without_kg"] = f"ERROR: {e}"

    return results.get("without_kg", ""), results.get("with_kg", ""), kg_facts


# ── Find video path ───────────────────────────────────────────────
def find_video_path(set_name: str, clip_name: str) -> str | None:
    clip_stem = clip_name.replace(".mp4", "")
    blurred_name = f"{clip_stem}_blurred.mp4"
    set_path = os.path.join(VIDEO_DIR, set_name)
    if not os.path.exists(set_path):
        return None
    for root, dirs, files in os.walk(set_path):
        if blurred_name in files:
            return os.path.join(root, blurred_name)
        if clip_name in files:
            return os.path.join(root, clip_name)
    return None


# ── Simple score ──────────────────────────────────────────────────
def simple_score(pred: str, gt: str) -> float:
    pred, gt = pred.lower().strip(), gt.lower().strip()
    if pred == gt or gt in pred or pred in gt:
        return 1.0
    stop = {"the","a","an","is","are","it","in","on","of","and","or","there","no","not"}
    pw = set(pred.split()) - stop
    gw = set(gt.split()) - stop
    if not gw:
        return 0.0
    return 1.0 if len(pw & gw) / len(gw) >= 0.5 else 0.0


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=str, default="Set_34",
                        help="Which set to focus on (default: Set_34)")
    parser.add_argument("--max_clips", type=int, default=20,
                        help="Max number of unique clips to process")
    parser.add_argument("--failures_only", action="store_true", default=True,
                        help="Only run on FT model failures (default: True)")
    args = parser.parse_args()

    # Load FT failures
    failures_csv = f"{ANALYSIS_DIR}/Qwen2.5-VL-3B-FT_failures.csv"
    all_questions_csv = f"{BASE_DIR}/data/morning_attribution.csv"

    # Load questions for target set
    questions = []
    with open(all_questions_csv) as f:
        for row in csv.DictReader(f):
            if row["set"] == args.set:
                questions.append(row)

    logger.info(f"Target set: {args.set} — {len(questions)} questions")

    # Load FT results to find failures
    ft_csv = f"{EVAL_DIR}/Qwen2.5-VL-3B-FT_morning_attribution.csv"
    ft_results = {}
    with open(ft_csv) as f:
        for row in csv.DictReader(f):
            if row["set"] == args.set:
                score = simple_score(row["generated_answer"], row["actual_answer"])
                ft_results[row["question_id"]] = {**row, "score": score}

    # Filter to failures only
    if args.failures_only:
        questions = [q for q in questions
                     if ft_results.get(q["question_id"], {}).get("score", 0) < 0.5]
        logger.info(f"Failures only: {len(questions)} questions")

    # Limit clips
    seen_clips = {}
    filtered = []
    for q in questions:
        clip_key = (q["set"], q["video_file_path"])
        if clip_key not in seen_clips:
            if len(seen_clips) >= args.max_clips:
                continue
            seen_clips[clip_key] = True
        filtered.append(q)
    questions = filtered
    logger.info(f"After clip limit ({args.max_clips} clips): {len(questions)} questions")

    # Load model
    logger.info("Loading model...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager"
    )
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model = model.merge_and_unload()
    model.eval()

    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True,
                                              min_pixels=128*28*28, max_pixels=360*28*28)
    logger.info(f"Model ready. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Run GraphRAG evaluation
    out_csv = f"{EVAL_DIR}/E003_GraphRAG_{args.set}.csv"
    fieldnames = ["question_id","set","video_id","question",
                  "answer_ft","answer_graphrag","actual_answer",
                  "kg_facts_used","score_ft","score_graphrag","improved"]

    results = []
    kg_cache = {}  # clip_id → nx.Graph

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(questions, desc="GraphRAG Eval"):
            set_name  = row["set"]
            clip_name = row["video_file_path"]
            question  = row["question"]
            actual    = row["actual_answer"]
            q_id      = row["question_id"]
            clip_key  = f"{set_name}/{clip_name}"

            video_path = find_video_path(set_name, clip_name)
            if not video_path:
                logger.warning(f"Video not found: {clip_key}")
                continue

            # Build/load KG for this clip
            if clip_key not in kg_cache:
                # Try cache first
                G = load_kg(clip_key)
                if G is None:
                    logger.info(f"Building KG for {clip_key}...")
                    frames = extract_frames(video_path, num_frames=5)
                    pil_frames = frames_to_pil(frames)
                    descriptions = [describe_frame(model, processor, img) for img in pil_frames]
                    descriptions = [d for d in descriptions if d]
                    G = build_clip_kg(descriptions)
                    save_kg(G, clip_key)
                    logger.debug(f"KG built: {G.number_of_nodes()} nodes")
                kg_cache[clip_key] = G
            else:
                G = kg_cache[clip_key]

            # Get FT answer (from cached results)
            ft_answer = ft_results.get(q_id, {}).get("generated_answer", "")

            # Run GraphRAG inference
            _, graphrag_answer, kg_facts = run_graphrag_inference(
                model, processor, question, video_path, G
            )

            score_ft = simple_score(ft_answer, actual)
            score_kg = simple_score(graphrag_answer, actual)
            improved = score_kg > score_ft

            result = {
                "question_id":   q_id,
                "set":           set_name,
                "video_id":      clip_name,
                "question":      question,
                "answer_ft":     ft_answer,
                "answer_graphrag": graphrag_answer,
                "actual_answer": actual,
                "kg_facts_used": kg_facts,
                "score_ft":      score_ft,
                "score_graphrag": score_kg,
                "improved":      improved,
            }
            results.append(result)
            writer.writerow(result)
            f.flush()

    # Final summary
    total     = len(results)
    ft_correct = sum(1 for r in results if r["score_ft"] >= 0.5)
    kg_correct = sum(1 for r in results if r["score_graphrag"] >= 0.5)
    improved   = sum(1 for r in results if r["improved"])
    regressed  = sum(1 for r in results if r["score_ft"] >= 0.5 and r["score_graphrag"] < 0.5)

    print(f"\n{'='*60}")
    print(f"  E003 GraphRAG Results — {args.set}")
    print(f"{'='*60}")
    print(f"  Total questions:    {total}")
    print(f"  FT correct:         {ft_correct} ({ft_correct/total:.1%})")
    print(f"  GraphRAG correct:   {kg_correct} ({kg_correct/total:.1%})")
    print(f"  Improved by KG:     {improved}")
    print(f"  Regressed by KG:    {regressed}")
    print(f"  Net gain:           +{improved - regressed}")
    print(f"  Results saved:      {out_csv}")
    print(f"{'='*60}\n")

    # Show examples where KG helped
    helps = [r for r in results if r["improved"]]
    if helps:
        print("  === KG HELPED THESE ===")
        for r in helps[:5]:
            print(f"  Q: {r['question'][:65]}")
            print(f"  KG facts: {r['kg_facts_used'][:80]}")
            print(f"  FT said:  {r['answer_ft'][:60]}")
            print(f"  KG said:  {r['answer_graphrag'][:60]}")
            print(f"  Actual:   {r['actual_answer'][:60]}")
            print()


if __name__ == "__main__":
    main()
