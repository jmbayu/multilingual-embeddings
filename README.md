# OpenAI vs open-source multilingual embeddings models

<a href="https://colab.research.google.com/github/Yannael/multilingual-embeddings/blob/main/Multilingual_Embeddings_OpenAI_vs_OpenSource.ipynb" target="_blank"><img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" /></a>

This noteboook provides example code to assess which embedding model works best for your data. The example task is a retrieval task (as in RAG - retrieval augmented generation), on multilingual data. See associated Medium article [here](https://medium.com/p/e5ccb7c90f05).

The data source is based on the European AI Act, and models cover some of the latest OpenAI and open-source embeddings models (as of 02/2024) to deal with multilingual data:

OpenAI released [two models](https://openai.com/blog/new-embedding-models-and-api-updates) in January 2024:

- text-embedding-3-small (released 25/01/2024)
- text-embedding-3-large (released 25/01/2024)

We compare with the following open-source models

- [e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) (released 04/01/2024)
- [multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) (released 08/02/2024)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) (released 29/01/2024)
- [nomic-embed-text-v1](https://huggingface.co/nomic-ai/nomic-embed-text-v1) (released 10/02/2024)



## Setup


Specific hardware requirement:

ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA T1200 Laptop GPU, compute capability 7.5, VMM: yes

This project requires PyTorch with CUDA 11.7 support. To set up the project, first create a Poetry environment and then install the required packages with `pip`:



+-----------------------------------------------------------------------------+
| NVIDIA-SMI 517.66       Driver Version: 517.66       CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA T1200 La... WDDM  | 00000000:01:00.0 Off |                  N/A |
| N/A   59C    P8     2W /  N/A |      0MiB /  4096MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

```bash
poetry shell
pip install torch==2.0.1+cu117 torchaudio==2.0.2+cu117 torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html