# [ICML26] Localization then Neutralization: Gradient-guided Token Suppression against Visual Prompt Injection Attack

## Environment Setup

Install the required Python packages first:

```
python -m pip install -r requirements.txt
```

If you want to download the model locally, please refer to [here](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).

## Evaluation

To test the defense effectiveness of our method, please run:

```
cd ./code/eval
python inference.py --model-path /Qwen/Qwen2-VL-7B-Instruct --device cude:0 --defense our
python inference.py --model-path /Qwen/Qwen2-VL-7B-Instruct --device cude:0 --defense wo
```
