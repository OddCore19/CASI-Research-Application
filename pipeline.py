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
    decomposition_prompt = f"Decompose the following task into three simpler, related tasks:\n\n{initial_task}"
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

# Example usage:
if __name__ == "__main__":
    task = "Summarize a recent scientific paper on climate change policies."
    result = two_model_workflow(task)
    print("Final result:\n", result)
