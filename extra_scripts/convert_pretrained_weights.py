import os
import torch

def process_file(file_path):
    state_dict = torch.load(file_path)

    filtered_state_dict = {name: param for name, param in state_dict.items() if 'lora' in name or 'audio' in name}

    print(f"Parameters in file {file_path}:")
    for name in filtered_state_dict.keys():
        print(name)

    torch.save(filtered_state_dict, file_path[:-4] + '_trainable.bin')
    print(file_path[:-4] + '_trainable.bin')
    print('----------------------------------')

# Walk through the current directory and its subdirectories
count = 0
for dirpath, dirnames, filenames in os.walk('/fs/nexus-projects/brain_project/acl_sk_24/GAMA//llm/alpaca-lora-main/'):
    for file in filenames:
        if file == "pytorch_model.bin":
            cur_target = os.path.join(dirpath, file)
            if os.path.exists(cur_target[:-4] + '_trainable.bin') == False:
                print(os.path.join(dirpath, file))
                process_file(os.path.join(dirpath, file))
                count +=1
print(count)