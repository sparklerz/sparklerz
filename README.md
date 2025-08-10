# Sarat Kannan
*Distributed training for LLMs — Stack: PyTorch DDP/FSDP · DeepSpeed ZeRO · Ray (Train/Tune) · MosaicML LLM Foundry · W&B/MLflow · HF Hub · Hivemind · Petals*

I build scalable LLM training/inference systems across **multi-GPU** (PyTorch DDP/FSDP, DeepSpeed ZeRO, Ray, MosaicML LLM Foundry) and **distributed swarms** (Hivemind, Petals). Writing experiments up so others can reproduce them.

## Selected work
- **Multi-GPU LLM Fine-Tuning (monorepo)** — DDP/FSDP, ZeRO, Ray Train/Tune, MosaicML; W&B/MLflow + HF Hub for fully traceable runs.  
    - Repo: https://github.com/sparklerz/multigpu-llm-finetuning
- **Hivemind fine-tuning (Qwen2-0.5B)** — Internet-scale data parallelism with DHT + fault tolerance; measured val-loss reductions.
    - Overview: (meta repo) https://github.com/sparklerz/hivemind-qwen2-0.5b  
    - Article: https://medium.com/@kannansarat9/finetuning-qwen-0-5b-using-hivemind-data-parallelism-over-the-internet-e20af1b15c05
- **Petals (LLaMA-2-70B)** — Distributed inference + Deep prompt-tuning via swarm model-parallelism.
    - Overview: (meta repo) https://github.com/sparklerz/petals-llama2-70b 
    - Part 1: https://medium.com/@kannansarat9/part-1-inferencing-llama-2-70b-using-petals-swarm-model-parallelism-over-the-internet-a29de8f8aef3
    - Part 2: https://medium.com/@kannansarat9/part-2-prompt-tuning-llama-2-70b-using-petals-model-parallelism-over-the-internet-89cdee667840

## Writing
Medium: https://medium.com/@kannansarat9

## Contact
DMs open: https://x.com/saratkannan
