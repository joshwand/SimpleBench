# python run_benchmark.py --model_name=gpt-4o-mini --dataset_path=output.json

from typing import Optional, Tuple
import time
import json
import weave
import asyncio
from fire import Fire

from dotenv import load_dotenv
load_dotenv()

from weave_utils.models import LiteLLMModel, MajorityVoteModel
from weave_utils.scorers import eval_majority_vote, eval_multi_choice


def load_dataset(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data['eval_data']


def run_benchmark(
    model_name: str = "o1-preview",
    dataset_path: str = "simple_bench_public.json",
    dataset_weave_ref: str = "weave:///simplebench/simple_bench_public/object/competition_dataset:qNJnkgpMqoyc48GwlFCSpVypn3D8x77N7lGCBIab4XQ",
    use_weave_dataset: bool = True,
    system_prompt: Optional[str] = None,
    num_responses: int = 1,
    entity: Optional[str] = "simplebench",
    project: str = "simple_bench_public",
    temp: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.95,
    max_retries: int = 3,
    system_prompt_path: str = "system_prompt.txt",
    q_ids: int | Tuple = -1,
    print_output: bool = False
):
    """
    Run a benchmark evaluation on a given model and dataset.

    Args:
        model_name (str): Name of the model to use for inference.
            Default is "gpt-4o-mini".
        dataset_path (str): Path to the dataset JSON file.
            Default is "simple_bench_public.json".
        dataset_weave_ref (str): Weave reference to the dataset.
            Default is "weave:///simplebench/simple_bench_public/object/competition_dataset:qNJnkgpMqoyc48GwlFCSpVypn3D8x77N7lGCBIab4XQ".
        use_weave_dataset (bool): Whether to use the dataset from Weave. If False, the dataset will be loaded from the local file.
            Default is True.
        q_ids (int): Comma-separated list of question IDs to answer (useful for testing)
        print_output (bool): if set, prints the model output to the console. Default is False.
        num_responses (int): If greater than 1, majority voting will be applied.
            Default is 1 (no majority voting).
        entity (str): Optional Weave entity (org/user name) for evaluation tracking.
        project (str): The project name under the specified entity.
            Default is "simple_bench_public".
        temp (float): Temperature for the model.
            Default is 0.7.
        max_tokens (int): Maximum number of tokens to generate.
            Default is 2048.
        top_p (float): Top-p for the model.
            Default is 0.95.
        max_retries (int): Maximum number of retries for the model.
            Default is 3.
        system_prompt (str): System prompt for the model.
            Default is "You are an expert at reasoning and you always pick the most realistic answer. Just output your final answer using the following format: Final Answer: X where X is one of the letters A, B, C, D, E, or F."

    Example:
        python run_benchmark.py --model_name=gpt-4o-mini --dataset_path=simple_bench_public.json --num_responses=3
    """

    if entity is not None:
        weave.init(f"{entity}/{project}")
    else:
        weave.init(f"{project}")

    if use_weave_dataset:
        dataset = weave.ref(dataset_weave_ref).get()
    else:
        dataset = load_dataset(dataset_path)

    if type(q_ids) == int and q_ids > 0:
      q_ids = [q_ids]

    if len(q_ids) > 0:
        dataset = list(filter(lambda x: x["question_id"] in q_ids, dataset))

    def debug_scorer(output):
      print(f"output: {output}")
      return None

    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[eval_majority_vote if num_responses > 1 else eval_multi_choice] + ([debug_scorer] if print_output else []),
        trials=1,
    )

    if system_prompt is None:
        with open(system_prompt_path, "r") as f:
            system_prompt = f.read().strip()
    else:
        system_prompt = system_prompt.strip()

    model = LiteLLMModel(
        model_name=model_name,
        temp=temp,
        max_tokens=max_tokens,
        top_p=top_p,
        max_retries=max_retries,
        system_prompt=system_prompt
    )

    if num_responses > 1:
        model = MajorityVoteModel(model=model, num_responses=num_responses)

    asyncio.run(
        evaluation.evaluate(
            model,
            __weave={"display_name": f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}_eval"}
        )
    )


if __name__ == "__main__":
    Fire(run_benchmark)
