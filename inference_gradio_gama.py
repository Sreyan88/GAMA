import os
import gradio as gr
import torch
import torchaudio
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from utils.prompter import Prompter
import datetime
import time,json

device = "cuda" if torch.cuda.is_available() else "cpu"

base_model = "/fs/nexus-projects/brain_project/acl_sk_24/GAMA//train_script/Llama-2-7b-chat-hf-qformer/"

prompter = Prompter('alpaca_short')
tokenizer = LlamaTokenizer.from_pretrained(base_model)

model = LlamaForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float32)

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

### Stage 4 ckpt
eval_mdl_path = '/fs/gamma-projects/audio/gama/new_data/stage4_all_mix_new/checkpoint-46800/pytorch_model.bin'

### Stage 5 ckpt
# eval_mdl_path = '/fs/gamma-projects/audio/gama/new_data/stage5_all_mix_all/checkpoint-900/pytorch_model.bin'

state_dict = torch.load(eval_mdl_path, map_location='cpu')
msg = model.load_state_dict(state_dict, strict=False)

model.is_parallelizable = True
model.model_parallel = True

# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model.eval()
eval_log = []
cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_save_path = './inference_log/'
if os.path.exists(log_save_path) == False:
    os.mkdir(log_save_path)
log_save_path = log_save_path + cur_time + '.json'

SAMPLE_RATE = 16000
AUDIO_LEN = 1.0

def load_audio(filename):
    waveform, sr = torchaudio.load(filename)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    audio_info = 'Original input audio length {:.2f} seconds, number of channels: {:d}, sampling rate: {:d}.'.format(waveform.shape[1]/sr, waveform.shape[0], sr)
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
    return fbank, audio_info

def predict(audio_path, question):
    print('audio path, ', audio_path)
    begin_time = time.time()

    instruction = question
    prompt = prompter.generate_prompt(instruction, None)
    print('Input prompt: ', prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    if audio_path != 'empty':
        cur_audio_input, audio_info = load_audio(audio_path)
        cur_audio_input = cur_audio_input.unsqueeze(0)
        if torch.cuda.is_available() == False:
            pass
        else:
            # cur_audio_input = cur_audio_input.half().to(device)
            cur_audio_input = cur_audio_input.to(device)
    else:
        cur_audio_input = None
        audio_info = 'Audio is not provided, answer pure language question.'

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
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
    output = tokenizer.decode(s)[len(prompt)+6:-4] # trim <s> and </s>
    end_time = time.time()
    print(output)
    cur_res = {'audio_id': audio_path, 'input': instruction, 'output': output}
    eval_log.append(cur_res)
    with open(log_save_path, 'w') as outfile:
        json.dump(eval_log, outfile, indent=1)
    print('eclipse time: ', end_time - begin_time, ' seconds.')
    return audio_info, output

link = "https://github.com/Sreyan88/GAMA"
text = "[Github]"
paper_link = "https://sreyan88.github.io/gamaaudio/"
paper_text = "[Paper]"
demo = gr.Interface(fn=predict,
                    inputs=[gr.Audio(type="filepath"), gr.Textbox(value='Describe the audio.', label='Edit the textbox to ask your own questions!')],
                    outputs=[gr.Textbox(label="Audio Meta Information"), gr.Textbox(label="GAMA Output")],
                    cache_examples=True,
                    title="Quick Demo of GAMA",
                    description="GAMA is a novel Large Large Audio-Language Model that is capable of understanding audio inputs and answer any open-ended question about it." + f"<a href='{paper_link}'>{paper_text}</a> " + f"<a href='{link}'>{text}</a> <br>" +
                    "GAMA is authored by members of the GAMMA Lab at the University of Maryland, College Park and Adobe, USA. <br>" +
                    "**GAMA is not an ASR model and has limited ability to recognize the speech content. It primarily focuses on perception and understanding of non-speech sounds.**<br>" +
                    "Input an audio and ask quesions! Audio will be converted to 16kHz and padded or trim to 10 seconds.")
demo.launch(debug=True, share=True)
