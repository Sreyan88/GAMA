# GAMA: A Large Audio-Language Model with Advanced Audio Understanding and Complex Reasoning Abilities  
<p align="center"><img src="https://github.com/Sreyan88/GAMA/blob/main/assets/GAMA.png?raw=true" alt="GAMA Logo." width="300"/></p>

This is the official implementation of our paper [GAMA: A Large Audio-Language Model with Advanced Audio Understanding and Complex Reasoning Abilities](https://arxiv.org/abs/2406.11768).

### Demo  
We have hosted 2 HF spaces, generously supported by HuggingFace for GAMA and GAMA-IT. Feel free to play around with our models here:  

<div align="center">

[![GAMA](https://img.shields.io/badge/%F0%9F%A4%97%20GAMA-Online_Demo-orange)](https://huggingface.co/spaces/sonalkum/GAMA)&nbsp;&nbsp;&nbsp;&nbsp;
[![GAMA](https://img.shields.io/badge/%F0%9F%A4%97%20GAMA%20IT-Online_Demo-black)](https://huggingface.co/spaces/sonalkum/GAMA-IT)&nbsp;

</div>

### Resources  

All resources required for GAMA and GAMA-IT can be found in [this drive](https://drive.google.com/drive/u/0/folders/1W8ZtlhXNZ2IdVcKWsQpLD4jVw98brYDM). Information about the files is provided below in respective sections. We also share some additional CLAP Checkpoints (to be used with [this repository](https://github.com/LAION-AI/CLAP)) to promote research in this space. These CLAP checkpoints are trained on 2M+ audio-caption pairs with large batch sizes on H100s.  

### Setup 🏋️
```shell
conda create -n gama python=3.10
conda activate gama
pip install -r requirements.txt
pip install -e hf-dev-train/transformers-main
pip install -e peft-main
```
----
### Training 🏃‍♂️

**When preparing audio files, please make sure all audio files use the same sampling rate of 16kHz.**

The format of the dataset is a JSON file of a list of dicts, in the following format:

```json
[
 {
  "audio_id": "path_to_audio_file",
  "instruction": "Question",
  "dataset": "dataset_name", % (optional)
  "task": "type_of_task", % question type (optional)
  "output": "corect_answer"
 },
  ...
]
```
- Download the Llama-2-7b-chat-hf-qformer from [here](https://drive.google.com/drive/u/0/folders/1W8ZtlhXNZ2IdVcKWsQpLD4jVw98brYDM).
- Update the path of the dowloaded Llama-2-7b-chat-hf-qformer in [finetune.py](./finetune.py) on line 93 and 98.

Use the following commands to train the model:
```shell
conda activate gama
cd train_script
# run finetuning on the data to train GAMA
./stage1.sh # need to specify the path of Llama-2-7b-chat-hf-qformer in for the `--base_model` arg.
./stage2.sh # need to specify the checkpoint in stage 1 training
./stage3.sh # need to specify the checkpoint in stage 2 training
./stage4.sh # need to specify the checkpoint in stage 3 training
# to instruction tune GAMA
./stage5.sh # need to specify the checkpoint in stage 4 training
```
**To infer or instruction tune GAMA on your own dataset, we have provided the checkpoints for stage 4 and stage 5 [here](https://drive.google.com/drive/u/0/folders/1W8ZtlhXNZ2IdVcKWsQpLD4jVw98brYDM).**

----
### Inference of GAMA 🔖
To infer GAMA/GAMA-IT on [CompA-R benchmark](https://drive.google.com/drive/u/0/folders/1W8ZtlhXNZ2IdVcKWsQpLD4jVw98brYDM), change the path to model in [gama_inf.py](/gama_inf.py) on line 215, and run:
```shell
python gama_inf.py
```
- CompA-R audios can be downloaded from [here](https://drive.google.com/drive/u/0/folders/1W8ZtlhXNZ2IdVcKWsQpLD4jVw98brYDM).
  
### Evaluation
To evaluate GAMA we use the evaluation scheme employed by [LTU](https://github.com/YuanGongND/ltu/tree/main), the evaluation scripts can be found [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu/eval).

----
**Note:** The current code of GAMA does not include the implementation of `soft-prompt`. The code for `soft-prompt` (and its related checkpoints) will be released after the paper is accepted. However, the stage 5 checkpoint released currently performs almost as well as with `soft-prompt`.

----

### Acknowledgement 🌻
We would like to thank the authors of [LTU](https://github.com/YuanGongND/ltu/tree/main) for open-sourcing their code, which inspired our work.

### Citation 🔏
```bib
@misc{ghosh2024gamalargeaudiolanguagemodel,
      title={GAMA: A Large Audio-Language Model with Advanced Audio Understanding and Complex Reasoning Abilities}, 
      author={Sreyan Ghosh and Sonal Kumar and Ashish Seth and Chandra Kiran Reddy Evuru and Utkarsh Tyagi and S Sakshi and Oriol Nieto and Ramani Duraiswami and Dinesh Manocha},
      year={2024},
      eprint={2406.11768},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2406.11768}, 
}
```
