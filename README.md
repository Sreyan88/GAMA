# [GAMA: A Large Audio-Language Model with Advanced Audio Understanding and Complex Reasoning Abilities](https://arxiv.org/abs/2406.11768)
<p align="center"><img src="https://github.com/Sreyan88/GAMA/blob/main/assets/GAMA.png?raw=true" alt="GAMA Logo." width="300"/></p>

This is the official implementation of our paper [GAMA: A Large Audio-Language Model with Advanced Audio Understanding and Complex Reasoning Abilities](https://arxiv.org/abs/2406.11768).



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
Download the LLaMa-2-chat checkpoint from Huggingface and Audio-Qformer safetensor config and index jsons from [here](https://drive.google.com/drive/u/0/folders/1W8ZtlhXNZ2IdVcKWsQpLD4jVw98brYDM). Move the downloaded safetensors and jsons to downloaded LLaMa's folder.

Use the following commands to train the model:
```shell
conda activate gama
cd train_script
# run finetuning on the data to train GAMA
./stage1.sh
./stage2.sh # need to specify the checkpoint in stage 1 training
./stage3.sh # need to specify the checkpoint in stage 2 training
./stage4.sh # need to specify the checkpoint in stage 3 training
# to instruction tune GAMA
./stage5.sh # need to specify the checkpoint in stage 4 training
```
**To infer or instruction tune GAMA on your own dataset, we have provided the checkpoints for stage 4 and stage 5 [here](https://drive.google.com/drive/u/0/folders/1W8ZtlhXNZ2IdVcKWsQpLD4jVw98brYDM).**

----
### Inference of GAMA 🔖
To infer GAMA/GAMA-IT on [CompA-R benchmark](https://drive.google.com/drive/u/0/folders/1W8ZtlhXNZ2IdVcKWsQpLD4jVw98brYDM), chaneg the path to model in [gama_inf.py](/gama_inf.py) on line 215, and run:
```shell
python gama_inf.py
```

### Evaluation
To evaluate GAMA we use the evaluation scheme employed by [LTU](https://github.com/YuanGongND/ltu/tree/main), the evaluation scripts can be found [here](https://github.com/YuanGongND/ltu/tree/main/src/ltu/eval).

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
