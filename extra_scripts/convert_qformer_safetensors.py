import argparse
import json
import os
import shutil
from collections import defaultdict
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Set, Tuple

import torch

from huggingface_hub import CommitInfo, CommitOperationAdd, Discussion, HfApi, hf_hub_download
from huggingface_hub.file_download import repo_folder_name
from safetensors.torch import _find_shared_tensors, _is_complete, load_file, save_file


COMMIT_DESCRIPTION = """
This is an automated PR created with https://huggingface.co/spaces/safetensors/convert

This new file is equivalent to `pytorch_model.bin` but safe in the sense that
no arbitrary code can be put into it.

These files also happen to load much faster than their pytorch counterpart:
https://colab.research.google.com/github/huggingface/notebooks/blob/main/safetensors_doc/en/speed.ipynb

The widgets on your model page will run using this model even if this is not merged
making sure the file actually works.

If you find any issues: please report here: https://huggingface.co/spaces/safetensors/convert/discussions

Feel free to ignore this PR.
"""

ConversionResult = Tuple[List["CommitOperationAdd"], List[Tuple[str, "Exception"]]]


def _remove_duplicate_names(
    state_dict: Dict[str, torch.Tensor],
    *,
    preferred_names: List[str] = None,
    discard_names: List[str] = None,
) -> Dict[str, List[str]]:
    if preferred_names is None:
        preferred_names = []
    preferred_names = set(preferred_names)
    if discard_names is None:
        discard_names = []
    discard_names = set(discard_names)

    shareds = _find_shared_tensors(state_dict)
    to_remove = defaultdict(list)
    for shared in shareds:
        complete_names = set([name for name in shared if _is_complete(state_dict[name])])
        if not complete_names:
            if len(shared) == 1:
                # Force contiguous
                name = list(shared)[0]
                state_dict[name] = state_dict[name].clone()
                complete_names = {name}
            else:
                raise RuntimeError(
                    f"Error while trying to find names to remove to save state dict, but found no suitable name to keep for saving amongst: {shared}. None is covering the entire storage.Refusing to save/load the model since you could be storing much more memory than needed. Please refer to https://huggingface.co/docs/safetensors/torch_shared_tensors for more information. Or open an issue."
                )

        keep_name = sorted(list(complete_names))[0]

        # Mecanism to preferentially select keys to keep
        # coming from the on-disk file to allow
        # loading models saved with a different choice
        # of keep_name
        preferred = complete_names.difference(discard_names)
        if preferred:
            keep_name = sorted(list(preferred))[0]

        if preferred_names:
            preferred = preferred_names.intersection(complete_names)
            if preferred:
                keep_name = sorted(list(preferred))[0]
        for name in sorted(shared):
            if name != keep_name:
                to_remove[keep_name].append(name)
    return to_remove


def get_discard_names(model_id: str, revision: Optional[str], folder: str, token: Optional[str]) -> List[str]:
    try:
        import json

        import transformers

        config_filename = hf_hub_download(
            model_id, revision=revision, filename="config.json", token=token, cache_dir=folder
        )
        with open(config_filename, "r") as f:
            config = json.load(f)
        architecture = config["architectures"][0]

        class_ = getattr(transformers, architecture)

        # Name for this varible depends on transformers version.
        discard_names = getattr(class_, "_tied_weights_keys", [])

    except Exception:
        discard_names = []
    return discard_names


class AlreadyExists(Exception):
    pass


def check_file_size(sf_filename: str, pt_filename: str):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size

    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """
        )


def rename(pt_filename: str) -> str:
    filename, ext = os.path.splitext(pt_filename)
    local = f"{filename}.safetensors"
    local = local.replace("pytorch_model", "model")
    return local


def convert_multi(
    model_id: str, *, revision=Optional[str], folder: str, token: Optional[str], discard_names: List[str]
) -> ConversionResult:
    filename = hf_hub_download(
        repo_id=model_id, revision=revision, filename="pytorch_model.bin.index.json", token=token, cache_dir=folder
    )
    with open(filename, "r") as f:
        data = json.load(f)

    filenames = set(data["weight_map"].values())
    local_filenames = []
    for filename in filenames:
        pt_filename = hf_hub_download(repo_id=model_id, filename=filename, token=token, cache_dir=folder)

        sf_filename = rename(pt_filename)
        sf_filename = os.path.join(folder, sf_filename)
        convert_file(pt_filename, sf_filename, discard_names=discard_names)
        local_filenames.append(sf_filename)

    index = os.path.join(folder, "model.safetensors.index.json")
    with open(index, "w") as f:
        newdata = {k: v for k, v in data.items()}
        newmap = {k: rename(v) for k, v in data["weight_map"].items()}
        newdata["weight_map"] = newmap
        json.dump(newdata, f, indent=4)
    local_filenames.append(index)

    operations = [
        CommitOperationAdd(path_in_repo=os.path.basename(local), path_or_fileobj=local) for local in local_filenames
    ]
    errors: List[Tuple[str, "Exception"]] = []

    return operations, errors


def convert_single(
    model_id: str, *, revision: Optional[str], folder: str, token: Optional[str], discard_names: List[str]
) -> ConversionResult:
    pt_filename = hf_hub_download(
        repo_id=model_id, revision=revision, filename="pytorch_model.bin", token=token, cache_dir=folder
    )

    sf_name = "model.safetensors"
    sf_filename = os.path.join(folder, sf_name)
    convert_file(pt_filename, sf_filename, discard_names)
    operations = [CommitOperationAdd(path_in_repo=sf_name, path_or_fileobj=sf_filename)]
    errors: List[Tuple[str, "Exception"]] = []
    return operations, errors


def convert_file(
    pt_filename: str,
    sf_filename: str,
    discard_names: List[str],
):
    loaded = torch.load(pt_filename, map_location="cpu")

    wts = loaded['model'].copy()

    for key in list(wts.keys()):
        if 'Qformer.' in key:
            wts[key.replace('Qformer.','model.Qformer.')] = wts[key]
            print(wts[key])
            del wts[key]
        elif 'query_tokens' in key:
            wts[key.replace('query_tokens','model.query_tokens')] = wts[key]
            print(wts[key])
            del wts[key]
        elif 'ln_vision' in key:
            wts[key.replace('ln_vision.','model.qformer_proj_norm.')] = wts[key]
            print(wts[key])
            del wts[key]
        else:
            del wts[key]

    if "model" in loaded:
        loaded = wts
    
    to_removes = _remove_duplicate_names(loaded, discard_names=discard_names)

    metadata = {"format": "pt"}
    for kept_name, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            if to_remove not in metadata:
                metadata[to_remove] = kept_name
            del loaded[to_remove]
    # Force tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = "/fs/nexus-projects/brain_project/acl_sk_24/GAMA//train_script"
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata=metadata)
    check_file_size(sf_filename, pt_filename)
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


def create_diff(pt_infos: Dict[str, List[str]], sf_infos: Dict[str, List[str]]) -> str:
    errors = []
    for key in ["missing_keys", "mismatched_keys", "unexpected_keys"]:
        pt_set = set(pt_infos[key])
        sf_set = set(sf_infos[key])

        pt_only = pt_set - sf_set
        sf_only = sf_set - pt_set

        if pt_only:
            errors.append(f"{key} : PT warnings contain {pt_only} which are not present in SF warnings")
        if sf_only:
            errors.append(f"{key} : SF warnings contain {sf_only} which are not present in PT warnings")
    return "\n".join(errors)


def convert_generic(
    model_id: str, *, revision=Optional[str], folder: str, filenames: Set[str], token: Optional[str]
) -> ConversionResult:
    operations = []
    errors = []

    extensions = set([".bin", ".ckpt"])
    for filename in filenames:
        prefix, ext = os.path.splitext(filename)
        if ext in extensions:
            pt_filename = hf_hub_download(
                model_id, revision=revision, filename=filename, token=token, cache_dir=folder
            )
            dirname, raw_filename = os.path.split(filename)
            if raw_filename == "pytorch_model.bin":
                # XXX: This is a special case to handle `transformers` and the
                # `transformers` part of the model which is actually loaded by `transformers`.
                sf_in_repo = os.path.join(dirname, "model.safetensors")
            else:
                sf_in_repo = f"{prefix}.safetensors"
            sf_filename = os.path.join(folder, sf_in_repo)
            try:
                convert_file(pt_filename, sf_filename, discard_names=[])
                operations.append(CommitOperationAdd(path_in_repo=sf_in_repo, path_or_fileobj=sf_filename))
            except Exception as e:
                errors.append((pt_filename, e))
    return operations, errors


convert_file("/fs/nexus-projects/brain_project/acl_sk_24/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20240601232/checkpoint_57.pth", "/fs/nexus-projects/brain_project/acl_sk_24/GAMA//train_script/Llama-2-7b-chat-hf-qformer/model-00004-of-00002.safetensors", None)