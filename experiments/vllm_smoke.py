from huggingface_hub import HfApi
from vllm import LLM, SamplingParams


def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    revision = HfApi().model_info(model_id).sha
    print("model", model_id, "revision", revision, flush=True)
    llm = LLM(
        model=model_id, revision=revision, tokenizer_revision=revision,
        dtype="half", max_model_len=512, max_num_seqs=1,
        gpu_memory_utilization=0.5, enforce_eager=True,
        tensor_parallel_size=1,
    )
    params = SamplingParams(temperature=0, max_tokens=32)
    outputs = llm.chat(
        [{"role": "user", "content": "Explain what a GPU does in one sentence."}],
        params,
    )
    for output in outputs:
        generated = output.outputs[0]
        print("generated_tokens", len(generated.token_ids))
        print("text", generated.text)

if __name__ == "__main__":
    main()
