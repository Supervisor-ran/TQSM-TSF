# MuTSP-Tourism

Code for Web text collection, **TQSM-TSF** text selection, and **MMMA-NLinear** tourism forecasting. Baselines are not included.

Run the commands below **from the repository root** on Windows, Linux, or macOS (with [uv](https://docs.astral.sh/uv/) installed).

## 1. Install

```sh
uv venv --python 3.10
uv pip install -r requirements.txt
uv run --no-project python -m src.run --help
```

The default PyTorch installation is platform-dependent. For NVIDIA GPU support, use the appropriate PyTorch/CUDA wheel for your system instead.

## 2. API keys and pretrained models

Create a **`.env`** file in the repository root:

```dotenv
XAI_API_KEY=your_xai_api_key
OPENAI_API_KEY=placeholder
HF_HOME=./pretrained_models/huggingface
HF_HUB_CACHE=./pretrained_models/huggingface/hub
```

`OPENAI_API_KEY` is needed by the current client initializer, even for Grok-only collection. Never commit `.env` to Git.

Place an existing **complete Hugging Face cache** in `pretrained_models/huggingface/hub/`, or let the models download there on first use:

- TQSM: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Japanese forecasting: `sonoisa/sentence-bert-base-ja-mean-tokens-v2`
- Yelp/HotelRec forecasting: `sentence-transformers/all-mpnet-base-v2`
- Collection sentiment: `LoneWolfgang/bert-for-japanese-twitter-sentiment`

For the usual Hugging Face user cache instead, remove both `HF_...` entries from `.env`.

## 3. Collect and select texts

- Set the Grok model name in `src/model_util/llm_caller.py` (`ask_grok`, default `grok-3`).
- Set locations, keywords, and collection dates in `src/data_collector/run_text_collector.py` (`CONFIG`).
- Prepare monthly numerical data under `data/num_collection/domestic/raw/`.

```sh
uv run --no-project --env-file .env python -m src.data_collector.run_text_collector
uv run --no-project --env-file .env python -m src.data_collector.run_topk_generator
```

Outputs are under `data/text_collection/` (`raw/`, `scored/`, `topk/`). **Skip collection if you already have prepared Top-k data.** The Web collection step uses a paid xAI API.

## 4. Train and evaluate

Use prepared datasets in `data/` (Kamakura/Yokohama), `data_Yelp/` (New Orleans), and `hotelrec_data/` (Amsterdam).

```sh
# Kamakura; change --x_num_loc to yokohama for Yokohama
uv run --no-project --env-file .env python -m src.run --x_num_type domestic --x_num_loc kamakura --seq_len 6 --pred_len 1 --lr 0.001 --train_epochs 10 --use_cross_modality --use_movement_alignment --use_data_ablation --ablation_variants full --ablation_modes balanced

# New Orleans; for Amsterdam use --x_num_type hotelrec --x_num_loc amsterdam
uv run --no-project --env-file .env python -m src.run --x_num_type yelp --x_num_loc new_orleans --seq_len 6 --pred_len 1 --lr 0.001 --train_epochs 10 --use_cross_modality --use_movement_alignment
```

These are quick-start commands, not the final hyperparameters for every paper experiment. Run `uv run --no-project python -m src.run --help` for all options.

## Citation

If you use this code or the **TQSM-TSF / MMMA-NLinear** methods, please cite:

> Ran Liu and Tsunenori Mine. 2027. **Quality-Aware Web Signal Selection and Movement-Aligned Multimodal Learning for Tourism Demand Forecasting.** In *Proceedings of the 20th ACM International Conference on Web Search and Data Mining (WSDM '27)*, **Findings Track**, Hong Kong, China. ACM. (To appear.)

```bibtex
@inproceedings{liu2027qualityaware,
  author    = {Liu, Ran and Mine, Tsunenori},
  title     = {Quality-Aware Web Signal Selection and Movement-Aligned Multimodal Learning for Tourism Demand Forecasting},
  booktitle = {Proceedings of the 20th ACM International Conference on Web Search and Data Mining (WSDM '27)},
  year      = {2027},
  publisher = {Association for Computing Machinery},
  note      = {Findings Track, Hong Kong, China. To appear}
}
```

The citation will be updated with the official DOI, page numbers, and final ACM proceedings metadata after publication.
