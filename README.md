# GAMA: A Large Audio-Language Model with Advanced Audio Understanding and Complex Reasoning Abilities  
<p align="center"><img src="https://github.com/Sreyan88/GAMA/blob/main/assets/GAMA.png?raw=true" alt="GAMA Logo." width="300"/></p>

This is the official implementation of our paper [GAMA: A Large Audio-Language Model with Advanced Audio Understanding and Complex Reasoning Abilities](https://arxiv.org/abs/2406.11768).

### Updates 🚨    

- 🎉 GAMA achieves the highest F1 score amongst all LALMs on [Deductive Reasoning benchmark by Microsoft](https://arxiv.org/abs/2407.18062) (Table 4; ACE F1 and NACC)!    
- 🎉 GAMA achieves the highest F1/Accuracy score amongst all LALMs on [Audio Hallucination benchmark by NTU](https://arxiv.org/abs/2406.08402)! A staggering 81.7% on POPE for Random and w/ Sampling. 

### Demo  
We have hosted 2 HF spaces, generously supported by HuggingFace🤗 for GAMA and GAMA-IT. Feel free to play around with our models here:  

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
@inproceedings{ghosh-etal-2024-gama,
    title = "{GAMA}: A Large Audio-Language Model with Advanced Audio Understanding and Complex Reasoning Abilities",
    author = "Ghosh, Sreyan  and
      Kumar, Sonal  and
      Seth, Ashish  and
      Evuru, Chandra Kiran Reddy  and
      Tyagi, Utkarsh  and
      Sakshi, S  and
      Nieto, Oriol  and
      Duraiswami, Ramani  and
      Manocha, Dinesh",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.361",
    pages = "6288--6313",
    abstract = "Perceiving and understanding non-speech sounds and non-verbal speech is essential to making decisions that help us interact with our surroundings. In this paper, we propose GAMA, a novel General-purpose Large Audio-Language Model (LALM) with Advanced Audio Understanding and Complex Reasoning Abilities. We build GAMA by integrating an LLM with multiple types of audio representations, including features from a custom Audio Q-Former, a multi-layer aggregator that aggregates features from multiple layers of an audio encoder. We fine-tune GAMA on a large-scale audio-language dataset, which augments it with audio understanding capabilities. Next, we propose CompA-R (Instruction-Tuning for Complex Audio Reasoning), a synthetically generated instruction-tuning (IT) dataset with instructions that require the model to perform complex reasoning on the input audio. We instruction-tune GAMA with CompA-R to endow it with complex reasoning abilities, where we further add a soft prompt as input with high-level semantic evidence by leveraging event tags of the input audio. Finally, we also propose CompA-R-test, a human-labeled evaluation dataset for evaluating the capabilities of LALMs on open-ended audio question-answering that requires complex reasoning. Through automated and expert human evaluations, we show that GAMA outperforms all other LALMs in literature on diverse audio understanding tasks by margins of 1{\%}-84{\%} and demonstrates state-of-the-art performance on deductive reasoning and hallucination evaluation benchmarks. Further, GAMA IT-ed on CompA-R proves to be superior in its complex reasoning capabilities.",
}
```
