import os
import json
from typing import List

from helpers import (
    call_openai,
    call_openai_image,
    call_instruct_pix2pix,
    download_image,
)


def parse_prompt_list(text: str) -> List[str]:
    """Parse a model response into a list of prompt strings.

    Tries JSON first, then falls back to parsing bullet/numbered lists.
    """
    text = text.strip()
    # Try JSON array
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass

    # Fallback: split lines/bullets like "1. foo", "- bar", "* baz"
    prompts: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # remove leading bullet/numbering
        if line[0] in ("-", "*", "•"):
            line = line[1:].strip()
        if ". " in line[:4]:  # handles "1. foo", "12. foo"
            parts = line.split(". ", 1)
            if parts[0].isdigit():
                line = parts[1].strip()
        # remove wrapping quotes
        if (line.startswith('"') and line.endswith('"')) or (
            line.startswith("'") and line.endswith("'")
        ):
            line = line[1:-1].strip()
        if line:
            prompts.append(line)
    return prompts


def run_img_pipeline(
    benign_prompt_file: str = "benign_img_prompt.txt",
    mal_prompt_file: str = "mal_img.txt",
    out_dir_benign: str = "benign_img",
    out_dir_mal: str = "mal_img",
    image_model: str = "gpt-image-1",
    chat_model_primary: str = "gpt-5",
    chat_model_fallback: str = "gpt-4-0125-preview",
) -> None:
    """Full pipeline:
    1) Ask a chat model for a list of image prompts using the prompt in benign_prompt_file.
    2) Generate an image for each prompt using OpenAI Image model and download to out_dir_benign.
    3) Edit each image with InstructPix2Pix using the prompt in mal_prompt_file and save to out_dir_mal.
    """

    # Ensure output directories exist
    os.makedirs(out_dir_benign, exist_ok=True)
    os.makedirs(out_dir_mal, exist_ok=True)

    # 1) Read benign prompt and get list of prompts via chat model
    with open(benign_prompt_file, "r", encoding="utf-8") as f:
        benign_prompt = f.read().strip()

    try:
        chat_resp = call_openai(benign_prompt, model=chat_model_primary)
    except Exception:
        # Fallback to a widely-available model if gpt-5 is unavailable
        chat_resp = call_openai(benign_prompt, model=chat_model_fallback)

    prompts = parse_prompt_list(chat_resp)
    if not prompts:
        raise ValueError("Failed to parse any prompts from the chat response.")

    # 2) Generate image for each prompt and download
    benign_paths: List[str] = []
    for idx, p in enumerate(prompts, start=1):
        urls = call_openai_image(p, model=image_model, n=1, response_format="url")
        if not urls:
            continue
        url = urls[0]
        file_path = os.path.join(out_dir_benign, f"benign_{idx:03}.png")
        try:
            download_image(url, file_path)
            benign_paths.append(file_path)
        except Exception as e:
            # Skip failed downloads, continue pipeline
            print(f"[warn] failed to download image for prompt {idx}: {e}")

    if not benign_paths:
        raise RuntimeError("No benign images were generated/downloaded.")

    # 3) Read malicious editing prompt
    with open(mal_prompt_file, "r", encoding="utf-8") as f:
        mal_prompt = f.read().strip()

    # 4) Edit each benign image with instruct-pix2pix
    for idx, img_path in enumerate(benign_paths, start=1):
        out_path = os.path.join(out_dir_mal, f"mal_{idx:03}.png")
        try:
            call_instruct_pix2pix(
                image_path=img_path,
                prompt=mal_prompt,
                output_path=out_path,
            )
        except Exception as e:
            print(f"[warn] failed to edit image {img_path}: {e}")

    print(f"Done. Saved {len(benign_paths)} benign images to '{out_dir_benign}' and edited results to '{out_dir_mal}'.")


if __name__ == "__main__":
    run_img_pipeline()
