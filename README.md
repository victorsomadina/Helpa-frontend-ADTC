# Helpa

First-aid guidance in Yorùbá and English. Gemma 4 E2B, LoRA fine-tuned, GGUF Q4_K_M, llama.cpp.

## Run it — one command

**Mac / Linux:**
```bash
curl -sL https://raw.githubusercontent.com/Yusasif-A/Helpa/main/run.sh | bash
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/Yusasif-A/Helpa/main/run.ps1 | iex
```

Downloads the model + vision projector + llama.cpp once, then runs entirely
on your own CPU. Accepts text, photos, and video (video is sampled as one
frame — llama.cpp has no true video input).

Weights: https://huggingface.co/yusasif/Helpa-ADTC-challenge
