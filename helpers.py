import os
import time
import argparse
from openai import OpenAI
from anthropic import Anthropic
from huggingface_hub import InferenceClient

def call_openai(prompt: str, model: str = "gpt-4-0125-preview") -> str:
    """Call an OpenAI model with a prompt and return the response."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content

def call_openai_image(
    prompt: str,
    model: str = "gpt-image-1",
    size: str = "1024x1024",
    quality: str = "standard",
    n: int = 1,
    response_format: str = "url",
) -> list[str]:
    """
    Generate images using OpenAI's Images API (DALL·E 3 via gpt-image-1).
    Args:
        quality: "standard" or "hd" (if available for your org).
        n: Number of images to generate.
        response_format: "url" to return hosted URLs, or "b64_json" to return base64-encoded image strings.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=n,
        response_format=response_format,
    )

    if response_format == "b64_json":
        return [d.b64_json for d in response.data]
    else:
        return [d.url for d in response.data]

def call_claude(prompt: str, model: str = "claude-3-opus-20240229") -> str:
    """Call an Anthropic Claude model via its API."""
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def call_llama(prompt: str, model_name: str = "meta-llama/Llama-2-70b-chat-hf") -> str:
    """
    Call an open-source model hosted on Hugging Face's inference API.
    This example uses the text-generation-inference endpoint.
    """
    client = InferenceClient(
        model=model_name,
        token=os.environ["HF_API_TOKEN"]
    )
    
    response = client.text_generation(
        prompt=prompt,
        max_new_tokens=512,
        temperature=0.2,
        return_full_text=False
    )
    return response


def call_instruct_pix2pix(
    image_path: str,
    prompt: str,
    output_path: str,
    model_name: str = "timbrooks/instruct-pix2pix",
    strength: float = 0.8,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    seed: int | None = None,
) -> bytes:
    """
    Call the InstructPix2Pix (Stable Diffusion v1.5) model via Hugging Face Inference API.
    Args:
        model_name: HF model repo id (default: "timbrooks/instruct-pix2pix").
        strength: How strongly to apply the edit (0-1, higher = more change).
        guidance_scale: Classifier-free guidance scale (typical 5.0-12.0).
        num_inference_steps: Number of denoising steps.
        seed: Optional random seed for reproducibility.
        output_path: The result is also saved to this path.

    Returns:
        The edited image as raw bytes (PNG by default). If output_path is provided, the bytes are also written to disk.
    """
    client = InferenceClient(
        model=model_name,
        token=os.environ["HF_API_TOKEN"],
    )

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Perform image-to-image editing
    result = client.image_to_image(
        image=image_bytes,
        prompt=prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )

    # Normalize to bytes (the client may return bytes or a PIL.Image)
    if isinstance(result, bytes):
        out_bytes = result
    else:
        # Attempt to handle PIL.Image without importing PIL directly
        try:
            import io
            buf = io.BytesIO()
            result.save(buf, format="PNG")  # type: ignore[attr-defined]
            out_bytes = buf.getvalue()
        except Exception:
            raise RuntimeError("Unexpected return type from image_to_image; install Pillow or set client to return bytes.")

    # Optionally write to disk
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(out_bytes)

    return out_bytes


def two_model_workflow(initial_task: str) -> str:
    # Step 1: Solve a hard benign sub-task with strong models.
    hard_task_template = ""
    with open(f"benign_{initial_task}.txt", "r") as f:
        hard_task_template = f.read()
        f.close()
    hard_task_prompt = hard_task_template.format(task=initial_task)
    solution_p1 = call_claude(hard_task_prompt)  # or call_openai if desired

    # Step 2: Solve a easy malicious sub-task with weak models.
    easy_task_template = ""
    with open(f"malicious_{initial_task}.txt", "r") as f:
        easy_task_template = f.read()
        f.close()
    easy_task_prompt = easy_task_template.format(task=initial_task, benign_solution=solution_p1)
    solution_p2 = call_llama(easy_task_prompt)

    return solution_p2


def auto_decompose_and_solve(initial_task: str) -> str:
    """
    Automatically decomposes a complex task and solves it using a multi-model workflow.
    """
    # Step 1: Decompose the problem using an open-source model (weak).
    decomposition_template = ""
    with open("decomposition_prompt.txt", "r") as f:
        decomposition_template = f.read()
    decomposition_prompt = decomposition_template.format(NEFARIOUS_TASK=initial_task)
    related_tasks = call_llama(decomposition_prompt)

    # Step 2: Solve hard benign sub-tasks with strong models.
    subtask_solutions = []
    for subtask in related_tasks.split("\n"):
        if not subtask.strip():
            continue
        solution = call_claude(subtask)  # or call_openai if desired
        subtask_solutions.append((subtask, solution))
        # Sleep between API calls to respect rate limits.
        time.sleep(1)

    # Step 3: Solve easy malicious sub-tasks with weak models.
    for subtask in related_tasks.split("\n"):
        if not subtask.strip():
            continue
        solution = call_llama(subtask)
        subtask_solutions.append((subtask, solution))
        time.sleep(1)

def download_image(url: str, output_path: str, timeout: float = 30.0) -> str:
    """
    Download an image from a URL and save it to disk.
    """
    import urllib.request
    from urllib.error import URLError, HTTPError

    # Ensure parent directory exists if provided
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Some servers require a user agent header
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()

    with open(output_path, "wb") as f:
        f.write(data)

    return output_path


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-Model Workflow Example")
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()
    task = args.task
    result = two_model_workflow(task)
    print("Final result:\n", result)
