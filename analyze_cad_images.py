#!/usr/bin/env python3
import argparse
import base64
import io
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

DEFAULT_DATASET = "dataset_eval_500.json"
DEFAULT_ORG_DIR = "benchmark/cad_imgs/org_imgs"
DEFAULT_EDIT_DIR = "benchmark/cad_imgs/edit_imgs"

MODEL_PRESETS = {
    "qwen": {
        "model": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        "base_url": "http://147.46.242.103:7121/v1",
        "extra_body": None,
        "max_images": 8,
        "max_tokens": 1024,
    },
}

_CAD_SEQUENCE_NOTATION = """
================================================
CAD SEQUENCE NOTATION
================================================
A sequence describes one or more sketch-then-extrude operations.

-- Sketch primitives --
  line,x,y
	  Line segment endpoint at (x, y) in sketch-plane coordinates.
	  Points are connected in order; the final point closes back to the first.

  arc,x,y,endX,endY
	  Arc from current point, bulging through (x,y), ending at (endX,endY).

  circle,centerX,centerY,centerZ,radiusX,radiusY,radiusZ
	  Full circle. The radius values describe the ellipse axes in 3D space;
	  equal radiusX/radiusY/radiusZ means a true circle.

-- Sketch structure --
  <curve_end>   ends one curve primitive
  <loop_end>    ends one closed contour (profile boundary or hole)
  <face_end>    ends one sketch face (can contain multiple loops: outer + inner holes)
  <sketch_end>  ends the full sketch; the next token is the extrude operation

-- Extrude operations --
  add,cx,cy,cz, nx,ny,nz, ux,uy,uz, vx,vy,vz, depth,scaleX,scaleY
	  Extrudes the sketch outward - ADDS material (boss/pad).

  cut,cx,cy,cz, nx,ny,nz, ux,uy,uz, vx,vy,vz, depth,scaleX,scaleY
	  Extrudes the sketch inward - REMOVES material (pocket/hole).

  Parameters:
	cx,cy,cz      Sketch plane center in world coordinates
	nx,ny,nz      Sketch plane normal vector (face orientation)
	ux,uy,uz      Sketch U-axis (horizontal direction on the sketch plane)
	vx,vy,vz      Sketch V-axis (vertical direction on the sketch plane)
	depth         Extrusion depth (how far the feature extends into/out of the body)
	scaleX,scaleY Sketch scaling factors in U and V directions

  Normal vector interpretation:
	(0,0,1)  or  (0,0,-1)  -> top or bottom face
	(0,1,0)  or  (0,-1,0)  -> front or back face
	(1,0,0)  or  (-1,0,0)  -> right or left face
	Mixed values            -> angled/chamfered face

  <extrude_end>  ends one full sketch+extrude block

-- Reading the diff --
  The EDITED sequence contains everything in ORIGINAL plus new tokens appended at the end.
  Focus on what was ADDED to the sequence - that is the edit.
  A new <sketch_end> ... <extrude_end> block = one new feature.
  Whether it is add or cut tells you if material was added or removed.
"""

_RENDERING_ARTIFACT_RULES = """
================================================
RENDERING ARTIFACT RULES
================================================
IGNORE:
  - Triangulated mesh lines and diagonal wireframe patterns
  - Texture-like surface lines that appear on flat faces
  - Shading differences that don't correspond to geometry changes

FOCUS ON:
  - New or removed edges and silhouette lines
  - New or removed faces
  - Changes in profile outline across views
"""

_OUTPUT_FORMAT = """
================================================
OUTPUT FORMAT
================================================
Return exactly 1 simple sentence with no label or prefix.
"""


# ---------------------------------------------------------------------------
# Prompt: visual-only
# 8 images, no sequences. Pure image comparison.
# ---------------------------------------------------------------------------
PROMPT_VISUAL_ONLY = (
    "You are an expert CAD analyst with deep knowledge of parametric 3D modeling.\n\n"
    "================================================\n"
    "IMAGE INPUT MAPPING\n"
    "================================================\n"
    "You receive 8 images in this exact order:\n"
    "  image[0] = ORIGINAL object - ISOMETRIC view\n"
    "  image[1] = ORIGINAL object - FRONT view\n"
    "  image[2] = ORIGINAL object - RIGHT view\n"
    "  image[3] = ORIGINAL object - TOP view\n"
    "  image[4] = EDITED object   - ISOMETRIC view\n"
    "  image[5] = EDITED object   - FRONT view\n"
    "  image[6] = EDITED object   - RIGHT view\n"
    "  image[7] = EDITED object   - TOP view\n\n"
    "Compare [0<->4] for isometric, [1<->5] for front, [2<->6] for right, [3<->7] for top.\n"
    "Use all 4 view pairs together. Each view reveals different spatial information:\n"
    "  - ISOMETRIC: overall shape, depth, and 3D form\n"
    "  - FRONT:     height and width, Y/Z-axis features\n"
    "  - RIGHT:     depth and height, X/Z-axis features\n"
    "  - TOP:       width and depth, X/Y-axis footprint\n"
    + _RENDERING_ARTIFACT_RULES
    + "\n================================================\n"
    "TASK\n"
    "================================================\n"
    "Compare all 4 view pairs and identify the single geometric change made to the object.\n"
    "Focus on new or removed edges, faces, and silhouette changes across all views.\n"
    + _OUTPUT_FORMAT
)

# ---------------------------------------------------------------------------
# Prompt: program-only
# Sequences only, no images. Text-only API call.
# ---------------------------------------------------------------------------
PROMPT_PROGRAM_ONLY = (
    "You are an expert CAD analyst with deep knowledge of parametric 3D modeling.\n"
    + _CAD_SEQUENCE_NOTATION
    + "\n================================================\n"
    "TASK\n"
    "================================================\n"
    "Find tokens present in EDITED but not in ORIGINAL.\n"
    "Identify: add or cut? Which face (from normal vector)? What shape was sketched?\n"
    "Describe the single geometric change in one sentence.\n"
    + _OUTPUT_FORMAT
    + "\nORIGINAL SEQUENCE:\n{original_seq}\n\nEDITED SEQUENCE:\n{edited_seq}\n"
)

# ---------------------------------------------------------------------------
# Prompt: joint
# Original object (4 images + sequence) vs edited object (4 images + sequence).
# Images sent in same order as text-bridge: orig[iso,front,right,top] then edit[iso,front,right,top].
# ---------------------------------------------------------------------------
PROMPT_JOINT = (
    "You are an expert CAD analyst with deep knowledge of parametric 3D modeling.\n\n"
    + _CAD_SEQUENCE_NOTATION
    + _RENDERING_ARTIFACT_RULES
    + "\n================================================\n"
    "ORIGINAL OBJECT\n"
    "================================================\n"
    "Images: image[0]=ISOMETRIC  image[1]=FRONT  image[2]=RIGHT  image[3]=TOP\n\n"
    "Sequence:\n{original_seq}\n\n"
    "================================================\n"
    "EDITED OBJECT\n"
    "================================================\n"
    "Images: image[4]=ISOMETRIC  image[5]=FRONT  image[6]=RIGHT  image[7]=TOP\n\n"
    "Sequence:\n{edited_seq}\n\n"
    "================================================\n"
    "TASK\n"
    "================================================\n"
    "Compare the original and edited objects using both images and sequences.\n"
    "Identify what was added or removed, its shape, and where it is located.\n"
    "Write: 'FINAL ANSWER: <one precise sentence>'\n"
)

# ---------------------------------------------------------------------------
# Prompt: text-bridge (original / default)
# 8 images + both sequences. Sequence leads; images localize.
# ---------------------------------------------------------------------------
PROMPT_TEXT_BRIDGE = (
    "You are an expert CAD analyst with deep knowledge of parametric 3D modeling.\n\n"
    + _CAD_SEQUENCE_NOTATION
    + _RENDERING_ARTIFACT_RULES
    + "\n================================================\n"
    "IMAGE INPUT MAPPING\n"
    "================================================\n"
    "You receive 8 images in this exact order:\n"
    "  image[0] = ORIGINAL object - ISOMETRIC view\n"
    "  image[1] = ORIGINAL object - FRONT view\n"
    "  image[2] = ORIGINAL object - RIGHT view\n"
    "  image[3] = ORIGINAL object - TOP view\n"
    "  image[4] = EDITED object   - ISOMETRIC view\n"
    "  image[5] = EDITED object   - FRONT view\n"
    "  image[6] = EDITED object   - RIGHT view\n"
    "  image[7] = EDITED object   - TOP view\n\n"
    "================================================\n"
    "TASK\n"
    "================================================\n"
    "STEP 1 — SEQUENCE DIFFERENCE\n"
    "   Compare ORIGINAL SEQUENCE and EDITED SEQUENCE.\n"
    "   Identify what changed: operation (add/cut), shape, face, depth.\n"
    "   Write: 'Sequence difference: <description of what changed>'\n\n"
    "STEP 2 — IMAGE DIFFERENCE\n"
    "   Compare image pairs: [0<->4] isometric, [1<->5] front, [2<->6] right, [3<->7] top.\n"
    "   Identify what visually changed: new or removed geometry, its shape, and its location.\n"
    "   Write: 'Image difference: <description of what changed>'\n\n"
    "STEP 3 — COMBINE\n"
    "   Compare the sequence difference and image difference.\n"
    "   Where they agree, state it confidently. Where one fills a gap in the other, use it.\n"
    "   Write: 'FINAL ANSWER: <one precise sentence combining both>'\n"
    + "\nORIGINAL SEQUENCE:\n{original_seq}\n\nEDITED SEQUENCE:\n{edited_seq}\n"
)

PROMPT_TEMPLATE = PROMPT_TEXT_BRIDGE  # backward-compat alias

PROMPT_MODES = {
    "visual-only":  PROMPT_VISUAL_ONLY,
    "program-only": PROMPT_PROGRAM_ONLY,
    "joint":        PROMPT_JOINT,
    "text-bridge":  PROMPT_TEXT_BRIDGE,
}


def _clean_llm_output(text):
	out = (text or "").strip().replace("\\\"", "\"")
	if out.startswith("```"):
		out = re.sub(r"^```[a-zA-Z]*\\n?", "", out).strip()
		out = re.sub(r"\\n?```$", "", out).strip()
	if len(out) >= 2 and out[0] == '"' and out[-1] == '"':
		out = out[1:-1].strip()
	out = re.sub(r"\\s+", " ", out).strip()
	return out


def _extract_field(raw_text, label):
	pattern = rf"(?im)^\\s*{re.escape(label)}\\s*:\\s*(.+)$"
	m = re.search(pattern, raw_text or "")
	return _clean_llm_output(m.group(1)) if m else ""


def _extract_final_answer(raw_text):
	final = _extract_field(raw_text, "FINAL ANSWER (Step 4)")
	if final:
		return final
	final = _extract_field(raw_text, "FINAL ANSWER")
	if final:
		return final

	text = _clean_llm_output(raw_text)
	for prefix in (
		"IMAGE DIFFERENCE:",
		"IMAGE DIFFERENCE :",
		"SEQUENCE DIFFERENCE:",
		"SEQUENCE DIFFERENCE :",
		"FINAL ANSWER:",
		"FINAL ANSWER :",
	):
		if text.upper().startswith(prefix):
			text = _clean_llm_output(text[len(prefix):])
			break

	if text:
		return text

	lines = [ln.strip() for ln in (raw_text or "").splitlines() if ln.strip()]
	return _clean_llm_output(lines[-1]) if lines else ""


def _image_to_data_url(path, max_size=512):
	img = Image.open(path)
	if max(img.size) > max_size:
		img.thumbnail((max_size, max_size), Image.LANCZOS)
	buf = io.BytesIO()
	img.save(buf, format="PNG")
	b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
	return f"data:image/png;base64,{b64}"


def _select_images(image_paths, max_images):
    """image_paths = [orig_iso, orig_front, orig_right, orig_top, edit_iso, edit_front, edit_right, edit_top]
    When max_images=4: pick orig_iso, orig_front, edit_iso, edit_front."""
    if max_images >= 8 or len(image_paths) <= max_images:
        return image_paths
    # Take the iso+front pairs: indices 0,1 (original) and 4,5 (edited)
    return [image_paths[0], image_paths[1], image_paths[4], image_paths[5]]


def _invoke_multiview(client, model, prompt_text, image_paths, max_images=8, max_tokens=512, temperature=0.0, extra_body=None, retries=3):
	candidates = [max_images, 4] if max_images > 4 else [max_images]
	for n_img in candidates:
		selected = _select_images(image_paths, n_img)
		content = [{"type": "text", "text": prompt_text}]
		for path in selected:
			content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(path)}})
		kwargs = dict(
			model=model,
			messages=[{"role": "user", "content": content}],
			max_tokens=max_tokens,
			temperature=temperature,
		)
		if extra_body:
			kwargs["extra_body"] = extra_body
		for attempt in range(1, retries + 1):
			try:
				response = client.chat.completions.create(**kwargs)
				if n_img < max_images:
					print(f"  [fallback] used {n_img} images instead of {max_images}")
				return response.choices[0].message.content or ""
			except Exception as e:
				print(f"  [attempt {attempt}/{retries}, {n_img} imgs] {e}")
	raise RuntimeError(f"Failed after {retries} retries with {max_images} and 4 images")


def _invoke_text_only(client, model, prompt_text, max_tokens=512, temperature=0.0, extra_body=None):
	kwargs = dict(
		model=model,
		messages=[{"role": "user", "content": prompt_text}],
		max_tokens=max_tokens,
		temperature=temperature,
	)
	if extra_body:
		kwargs["extra_body"] = extra_body
	response = client.chat.completions.create(**kwargs)
	return response.choices[0].message.content or ""


def _create_client(base_url, api_key):
	return OpenAI(base_url=base_url, api_key=api_key)


def _candidate_paths(index_value, original_name, edited_name, base_dir, view):
	prefix = f"{index_value:05d}_{original_name}_{edited_name}_"
	for suffix in ("002", "001", "003"):
		yield base_dir / f"{prefix}{suffix}_final_{view}.png"


def _find_view_path(index_value, original_name, edited_name, base_dir, view):
	for path in _candidate_paths(index_value, original_name, edited_name, base_dir, view):
		if path.exists():
			return path

	pattern = f"*_{original_name}_{edited_name}_*_final_{view}.png"
	matches = sorted(base_dir.glob(pattern))
	return matches[0] if matches else None


def _find_view_paths(index_value, original_name, edited_name, base_dir):
	views = ["iso", "front", "right", "top"]
	paths = []
	missing = []
	for view in views:
		path = _find_view_path(index_value, original_name, edited_name, base_dir, view)
		if path is None:
			missing.append(view)
		paths.append(path)
	return paths, missing


def _is_int(value):
	return isinstance(value, int) and not isinstance(value, bool)


def process_dataset(
	dataset_path=DEFAULT_DATASET,
	org_dir=DEFAULT_ORG_DIR,
	edit_dir=DEFAULT_EDIT_DIR,
	num_items=None,
	model="Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
	base_url="http://147.46.242.103:7121/v1",
	api_key="EMPTY",
	output_path="cad_image_eval_results.json",
	debug_output_path=None,
	temperature=0.0,
	max_tokens=512,
	extra_body=None,
	max_images=8,
	prompt_mode="text-bridge",
):
	load_dotenv()

	if not api_key:
		api_key = os.environ.get("CAD_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"

	org_dir = Path(org_dir)
	edit_dir = Path(edit_dir)

	print("=" * 100)
	print("LangChain Vision-Based CAD Eval (Late Fusion)")
	print(f"Endpoint: {base_url}")
	print(f"Model: {model}")
	print("=" * 100)
	print()

	client = _create_client(base_url=base_url, api_key=api_key)

	print(f"Loading dataset from {dataset_path}...")
	with open(dataset_path, "r") as f:
		dataset = json.load(f)

	items_to_process = dataset[:num_items] if num_items else dataset
	print(f"Processing {len(items_to_process)} items")
	print(f"Org images:  {org_dir}")
	print(f"Edit images: {edit_dir}")
	print()

	results = []
	debug_results = []
	success_count = 0

	for idx, item in enumerate(items_to_process, 1):
		index_value = item.get("index")
		original_name = item.get("original_pic_name")
		edited_name = item.get("edited_pic_name")
		original_seq = item.get("original_sequence", "")
		edited_seq = item.get("edited_sequence", "")
		debug_info = {}

		print(f"[{idx}/{len(items_to_process)}] {original_name} -> {edited_name} (index={index_value})")
		print("-" * 100)

		original_paths = [None, None, None, None]
		edited_paths = [None, None, None, None]

		if not (_is_int(index_value) and original_name and edited_name):
			print("  Warning: Missing index or image names - skipping")
			answer = ""
			debug_info["skipped"] = "missing index or pic name"
		else:
			original_paths, missing_org = _find_view_paths(index_value, original_name, edited_name, org_dir)
			edited_paths, missing_edit = _find_view_paths(index_value, original_name, edited_name, edit_dir)
			missing_views = {"original": missing_org, "edited": missing_edit}

			needs_images = prompt_mode != "program-only"
			if needs_images and any(p is None for p in original_paths + edited_paths):
				print(f"  Warning: Missing images - {missing_views}")
				answer = ""
				debug_info["skipped"] = "missing images"
				debug_info["missing_views"] = missing_views
			else:
				template = PROMPT_MODES.get(prompt_mode, PROMPT_TEXT_BRIDGE)
				prompt_text = template.format(
					original_seq=original_seq or "",
					edited_seq=edited_seq or "",
				)
				print("  Analyzing with model...")
				try:
					if prompt_mode == "program-only":
						raw = _invoke_text_only(client, model, prompt_text, max_tokens=max_tokens, temperature=temperature, extra_body=extra_body)
					else:
						raw = _invoke_multiview(client, model, prompt_text, original_paths + edited_paths, max_images=max_images, max_tokens=max_tokens, temperature=temperature, extra_body=extra_body)
					final_answer = _extract_final_answer(raw)
					answer = final_answer
					debug_info["raw_response"] = raw
					debug_info["image_difference"] = _extract_field(raw, "IMAGE DIFFERENCE")
					debug_info["sequence_difference"] = _extract_field(raw, "SEQUENCE DIFFERENCE")
					debug_info["final_answer"] = final_answer
					debug_info["original_images"] = [str(p) for p in original_paths]
					debug_info["edited_images"] = [str(p) for p in edited_paths]
				except Exception as e:
					print(f"  Skipping (all retries failed): {e}")
					answer = ""
					debug_info["skipped"] = str(e)

		if answer:
			success_count += 1

		result = {
			"index": index_value,
			"original_pic_name": original_name,
			"edited_pic_name": edited_name,
			"original_sequence": original_seq,
			"edited_sequence": edited_seq,
			"original_images": [str(p) for p in original_paths if p],
			"edited_images": [str(p) for p in edited_paths if p],
			"question": "What is the geometric difference between the original and edited CAD objects?",
			"answer": answer,
		}
		results.append(result)

		if debug_output_path is not None:
			debug_results.append(
				{
					"index": index_value,
					"original_pic_name": original_name,
					"edited_pic_name": edited_name,
					"steps": debug_info,
				}
			)

		print(f"  Answer: {answer if answer else '(blank)'}")
		print()

	with open(output_path, "w") as f:
		json.dump(results, f, indent=2)

	if debug_output_path is not None:
		with open(debug_output_path, "w") as f:
			json.dump(debug_results, f, indent=2)
		print(f"Saved debug output: {debug_output_path}")

	print(f"Saved results: {output_path}")
	print(f"Processed {success_count}/{len(items_to_process)} items successfully")


def main():
	parser = argparse.ArgumentParser(description="Late-fusion CAD eval using 8-view images and sequences")
	parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to dataset_eval_50.json")
	parser.add_argument("--org-dir", default=DEFAULT_ORG_DIR, help="Directory containing original images")
	parser.add_argument("--edit-dir", default=DEFAULT_EDIT_DIR, help="Directory containing edited images")
	parser.add_argument("--num-items", type=int, default=None, help="Number of items to process (default: all)")
	parser.add_argument(
		"--preset",
		choices=list(MODEL_PRESETS.keys()),
		default=None,
		help=f"Model preset shortcut: {', '.join(MODEL_PRESETS.keys())} (overrides --model and --base-url defaults)",
	)
	parser.add_argument("--model", default=None, help="Model name exposed by /v1/models (overrides preset)")
	parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (overrides preset)")
	parser.add_argument("--api-key", default=os.environ.get("CAD_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY", help="API key (if required by server)")
	parser.add_argument("--output", default=None, help="Output JSON file path (default: benchmark/cad_eval_<prompt-mode>_500.json)")
	parser.add_argument("--debug-output", default=None, help="Optional debug JSON path for intermediate outputs")
	parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
	parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens in the response (default: preset value)")
	parser.add_argument(
		"--prompt-mode",
		choices=list(PROMPT_MODES.keys()),
		default="text-bridge",
		help="Prompt strategy: visual-only (images only), program-only (sequences only), joint (original vs edited), text-bridge (images+sequences with cross-check, default)",
	)

	args = parser.parse_args()

	output_path = args.output or f"benchmark/cad_eval_{args.prompt_mode.replace('-', '_')}_500.json"

	# Resolve model/base_url/extra_body: preset < explicit args < env var
	preset = MODEL_PRESETS.get(args.preset, MODEL_PRESETS["qwen"]) if args.preset else MODEL_PRESETS["qwen"]
	model = args.model or preset["model"]
	base_url = args.base_url or os.environ.get("CAD_LLM_BASE_URL") or preset["base_url"]
	extra_body = preset["extra_body"]
	max_images = preset["max_images"]
	# --max-tokens explicitly passed overrides preset default
	max_tokens = args.max_tokens if args.max_tokens is not None else preset["max_tokens"]

	process_dataset(
		dataset_path=args.dataset,
		org_dir=args.org_dir,
		edit_dir=args.edit_dir,
		num_items=args.num_items,
		model=model,
		base_url=base_url,
		api_key=args.api_key,
		output_path=output_path,
		debug_output_path=args.debug_output,
		temperature=args.temperature,
		max_tokens=max_tokens,
		extra_body=extra_body,
		max_images=max_images,
		prompt_mode=args.prompt_mode,
	)


if __name__ == "__main__":
	main()
