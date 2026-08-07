conda activate gemma4-finetuning
python export_sft_to_gguf.py



Modelfile:
FROM ./gemma-4-31B-it.Q4_K_M.gguf
FROM ./gemma-4-31B-it.BF16-mmproj.gguf

PARAMETER num_ctx 9216
PARAMETER num_predict 8192
PARAMETER temperature 0


then:
ollama create gemma4-31B-it-tikz-sft-normal -f Modelfile
ollama show gemma4-31B-it-tikz-sft-normal