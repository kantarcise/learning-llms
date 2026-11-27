# Chapter 7 on the RTX 3090

From the repository root on Linux, run:

```bash
bash experiments/instruction-tuning/run-3090.sh
```

The launcher uses the repository’s locked CUDA environment, downloads and verifies the book helpers, dataset and GPT-2 Medium checkpoint, then trains for **two epochs (232 updates)**. Allow room for the 1.73 GB download, the trained checkpoint, and the CUDA environment if it is new. The Mac run stopped after its last logged evaluation at update 21. This run starts from the original pretrained weights.

The launcher prints its output directory under `artifacts/chapter-07/rtx3090-*/`. Save `metrics.json`, `loss.svg` and `training.log` for the chapter notes. Keep `model.pth` locally for §7.7. Load it into the same GPTModel config using `torch.load(..., map_location="cpu", weights_only=True)` followed by `load_state_dict`. The checkpoint contains inference weights; optimizer state is not saved. The larger Stanford Alpaca dataset in Exercise 7.3 is a separate experiment.
