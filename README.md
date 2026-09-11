# Sarat Kannan

I build end-to-end AI/ML systems spanning **RAG**, **agentic AI**, **LLM fine-tuning**, and **applied deep learning**, with a production software engineering background in building scalable APIs, data pipelines, and backend systems.

## Selected work
- **Hybrid RAG Suite**
    - Built a RAG app that ingests PDFs, chunks text, stores embeddings in Chroma, and supports chat over each uploaded document set.
    - Implemented hybrid retrieval that combines dense embeddings with BM25, merges ranks with Reciprocal Rank Fusion, and supports optional cross-encoder reranking.
    - Built a web Q&A mode that queries Tavily search API, Wikipedia, and arXiv in parallel with timeouts, then generates answers grounded in retrieved sources.
    - Created an evaluation module (Hit@K, MRR) with checks via expected-text matches or gold chunk id labels.
    - Demo: https://huggingface.co/spaces/ash001/Hybrid_RAG_Suite
    - Repo: https://github.com/sparklerz/Hybrid-RAG-Suite
- **LangGraph Multi-Agent Orchestrator**
    - Built a Streamlit multi-agent chat app that routes each query to SQL, Neo4j, web tools, or a general assistant using LangGraph state graphs.
    - Implemented a read-only SQLite SQL agent with LangChain tool-calling, capped at 30 steps and 60 sec per run.
    - Built a Neo4j agent that generates Cypher from the graph schema, executes it, then answers from query results.
    - Integrated web tools and a safe calculator, and logged each tool call.
    - Deployed to Hugging Face Spaces using GitHub Actions CI/CD.
    - Demo: https://huggingface.co/spaces/ash001/LangGraph_Multi-Agent_Orchestrator
    - Repo: https://github.com/sparklerz/LangGraph-Multi-Agent-Orchestrator
- **Gemma-2B QLoRA Adapter Fine-Tuning**
    - Fine-tuned `google/gemma-2b` with QLoRA on Databricks Dolly-15k, using 2,000 train and 200 eval samples.
    - Trained with 4-bit NF4 (double quant), gradient checkpointing, and 4-step accumulation.
    - Trained LoRA adapters with r=8, alpha=16, dropout=0.05 on attention and MLP projection layers.
    - Logged metrics to W&B, saved adapter weights with a `run_metadata.json`, and published a base vs tuned comparison report.
    - Adapter: https://huggingface.co/ash001/gemma-2b-dolly-qlora-adapter
    - W&B report: https://wandb.ai/kannansarat9/gemma-qlora
    - Repo: https://github.com/sparklerz/Gemma-2B-QLoRA-Adapter-Fine-Tuning
- **Deep Learning Fundamentals Suite**
    - Built a multi-page Streamlit app with 6 deep learning demos across tabular data, NLP, time series, and image classification.
    - Packaged each model for repeatable inference by saving weights and preprocessing files, then hosting them on Hugging Face Hub for app downloads and caching.
    - Trained models in TensorFlow/Keras and PyTorch, including ANNs, SimpleRNNs, LSTMs, and ResNet-18 transfer learning, reaching 97.9% validation accuracy on Cats vs Dogs and 2.86 RMSE on NYC taxi fare.
    - Demo: https://deep-learning-fundamentals-suite.streamlit.app/
    - Repo: https://github.com/sparklerz/Deep-Learning-Fundamentals-Suite

## Earlier LLM systems work
Before focusing more broadly on AI/ML engineering, I explored distributed and decentralised LLM training and inference.

- **Hivemind fine-tuning (Qwen2-0.5B)** — Internet-scale data parallelism with DHT + fault tolerance; measured val-loss reductions.
    - Overview: (meta repo) https://github.com/sparklerz/hivemind-qwen2-0.5b  
    - Article: https://medium.com/@kannansarat9/finetuning-qwen-0-5b-using-hivemind-data-parallelism-over-the-internet-e20af1b15c05
- **Petals (LLaMA-2-70B)** — Decentralised inference + Deep prompt-tuning via swarm model-parallelism.
    - Overview: (meta repo) https://github.com/sparklerz/petals-llama2-70b 
    - Part 1: https://medium.com/@kannansarat9/part-1-inferencing-llama-2-70b-using-petals-swarm-model-parallelism-over-the-internet-a29de8f8aef3
    - Part 2: https://medium.com/@kannansarat9/part-2-prompt-tuning-llama-2-70b-using-petals-model-parallelism-over-the-internet-89cdee667840
- **Distributed Multi-GPU LLM Fine-Tuning (monorepo)** — PyTorch (DDP, FSDP), DeepSpeed (ZeRO Offload, Pipeline Parallelism), Ray (Train, Tune), MosaicML; W&B/MLflow + HF Hub for fully traceable runs.  
    - Repo: https://github.com/sparklerz/multigpu-llm-finetuning

## Writing
Medium: https://medium.com/@kannansarat9

## Contact
DMs open: https://x.com/saratkannan
