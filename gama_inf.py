import sys
import os

default_cuda_devices = "0"
if len(sys.argv) > 1:
    argument = sys.argv[1]
    if argument == '4':
        argument = default_cuda_devices
else:
    argument = default_cuda_devices
os.environ["CUDA_VISIBLE_DEVICES"] = argument
import numpy as np
import os
import torchaudio
import fire
import json
import torch
from tqdm import tqdm
import time
import torchvision
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaConfig

from utils.prompter import Prompter

device = "cuda" if torch.cuda.is_available() else "cpu"

def int16_to_float32_torch(x):
    return (x / 32767.0).type(torch.float32)

def float32_to_int16_torch(x):
    x = torch.clamp(x, min=-1., max=1.)
    return (x * 32767.).type(torch.int16)

def get_mel(audio_data):
        # mel shape: (n_mels, T)
    mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=48000,
            n_fft=1024,
            win_length=1024,
            hop_length=480,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm=None,
            onesided=True,
            n_mels=64,
            f_min=50,
            f_max=14000
    ).to(audio_data.device)
        
    mel = mel_tf(audio_data)

        # we use log mel spectrogram as input
    mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)

    return mel.T  # (T, n_mels)


def get_audio_features(sample, audio_data, max_len, data_truncating, data_filling, require_grad=False):
        grad_fn = suppress if require_grad else torch.no_grad
        with grad_fn():
            if len(audio_data) > max_len:
                if data_truncating == "rand_trunc":
                    longer = torch.tensor([True])
                elif data_truncating == "fusion":
                    # fusion
                    mel = get_mel(audio_data)
                    # split to three parts
                    chunk_frames = max_len // 480 + 1  # the +1 related to how the spectrogram is computed
                    total_frames = mel.shape[0]
                    if chunk_frames == total_frames:
                        # there is a corner case where the audio length is
                        # larger than max_len but smaller than max_len+hop_size.
                        # In this case, we just use the whole audio.
                        mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                        sample["mel_fusion"] = mel_fusion
                        longer = torch.tensor([False])
                    else:
                        ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
                        # print('total_frames-chunk_frames:', total_frames-chunk_frames,
                        #       'len(audio_data):', len(audio_data),
                        #       'chunk_frames:', chunk_frames,
                        #       'total_frames:', total_frames)
                        if len(ranges[1]) == 0:
                            # if the audio is too short, we just use the first chunk
                            ranges[1] = [0]
                        if len(ranges[2]) == 0:
                            # if the audio is too short, we just use the first chunk
                            ranges[2] = [0]
                        # randomly choose index for each part
                        idx_front = np.random.choice(ranges[0])
                        idx_middle = np.random.choice(ranges[1])
                        idx_back = np.random.choice(ranges[2])
                        # select mel
                        mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
                        mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
                        mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]

                        # shrink the mel
                        mel_shrink = torchvision.transforms.Resize(size=[chunk_frames, 64])(mel[None])[0]
                        # logging.info(f"mel_shrink.shape: {mel_shrink.shape}")

                        # stack
                        mel_fusion = torch.stack([mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
                        sample["mel_fusion"] = mel_fusion #.unsqueeze(0)
                        longer = torch.tensor([True])
                else:
                    raise NotImplementedError(
                        f"data_truncating {data_truncating} not implemented"
                    )
                # random crop to max_len (for compatibility)
                overflow = len(audio_data) - max_len
                idx = np.random.randint(0, overflow + 1)
                audio_data = audio_data[idx: idx + max_len]

            else:  # padding if too short
                if len(audio_data) < max_len:  # do nothing if equal
                    if data_filling == "repeatpad":
                        n_repeat = int(max_len / len(audio_data))
                        audio_data = audio_data.repeat(n_repeat)
                        # audio_data = audio_data.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                        # audio_data = F.interpolate(audio_data,size=max_len,mode="bicubic")[0,0,0]
                        audio_data = F.pad(
                            audio_data,
                            (0, max_len - len(audio_data)),
                            mode="constant",
                            value=0,
                        )
                    elif data_filling == "pad":
                        audio_data = F.pad(
                            audio_data,
                            (0, max_len - len(audio_data)),
                            mode="constant",
                            value=0,
                        )
                    elif data_filling == "repeat":
                        n_repeat = int(max_len / len(audio_data))
                        audio_data = audio_data.repeat(n_repeat + 1)[:max_len]
                    else:
                        raise NotImplementedError(
                            f"data_filling {data_filling} not implemented"
                        )
                if data_truncating == 'fusion':
                    mel = get_mel(audio_data)
                    mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                    sample["mel_fusion"] = mel_fusion
                longer = torch.tensor([False])

        sample["longer"] = longer
        sample["waveform"] = audio_data
        sample["mel_fusion"] = sample["mel_fusion"].unsqueeze(0)
        # print(sample["mel_fusion"].shape)
        # print("---------------------")
        return sample


    
def load_audio(filename):
    waveform, sr = torchaudio.load(filename)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr,
                                              use_energy=False, window_type='hanning',
                                              num_mel_bins=128, dither=0.0, frame_shift=10)
    target_length = 1024
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    # normalize the fbank
    fbank = (fbank + 5.081) / 4.4849
    return fbank



def main(
    base_model: str = "/fs/nexus-projects/brain_project/Llama-2-7b-chat-hf-qformer",
    prompt_template: str = "alpaca_short",  # The prompt template to use, will default to alpaca.
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    # model = LlamaForCausalLM.from_pretrained(base_model, device_map="auto")
    model = LlamaForCausalLM.from_pretrained(base_model, device_map="auto") #, torch_dtype=torch.bfloat16


    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    temp, top_p, top_k = 0.1, 0.95, 500
    # change it to your model path
    eval_mdl_path = '/fs/gamma-projects/audio/gama/new_data/stage5/checkpoint-2500/pytorch_model.bin'
    state_dict = torch.load(eval_mdl_path, map_location='cpu')
    msg = model.load_state_dict(state_dict, strict=False)

    model.is_parallelizable = True
    model.model_parallel = True

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()
    file = open('/fs/nexus-projects/brain_project/acl_sk_24/GAMA_Benchmark_new.json','r')
    file = json.load(file)
    res = []
    for i in tqdm(file):
        tmp = {}
        for j in i['instruction_output']:
            audio_path = i['audio_id']
            instruction = j['instruction']
            prompt = prompter.generate_prompt(instruction, None)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            if audio_path != 'empty':
                cur_audio_input = load_audio(audio_path).unsqueeze(0)
                if torch.cuda.is_available() == False:
                    pass
                else:
                    cur_audio_input = cur_audio_input.to(device)
            else:
                cur_audio_input = None

            generation_config = GenerationConfig(
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=1.1,
                max_new_tokens=400,
                bos_token_id=model.config.bos_token_id,
                eos_token_id=model.config.eos_token_id,
                pad_token_id=model.config.pad_token_id,
                num_return_sequences=1
            )

            # Without streaming

            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids.to(device),
                    audio_input=cur_audio_input,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=400,
                )
            s = generation_output.sequences[0]
            output = tokenizer.decode(s)[6:-4]
            output = output[len(prompt):]
            # print('----------------------')
            # print(output)
            tmp['audio_id'] = audio_path
            tmp['instruction'] = instruction
            tmp['scene_caption'] = i['caption']
            tmp['prediction'] = output
            tmp['timestamp_events'] = i['timestamp_events']
            tmp['ref'] = j["output"]
            res.append(tmp)
    with open("stage5_answers_qformer_all.json", "w") as res_file:
        json.dump(res, res_file, indent=4)

if __name__ == "__main__":
    fire.Fire(main)
