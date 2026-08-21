#Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
from transformers import TrainingArguments
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence

import torch

import transformers
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava import LlavaLlamaForCausalLM

from PIL import Image
import torch.nn as nn
import math


# TODO: import and use code from ../data/dataset.py

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_vision_select_feature: Optional[str] = field(default='patch')
    mm_patch_merge_type: Optional[str] = field(default='flat')


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    eval_data_path: Optional[str] = field(default=None,
                                          metadata={"help": "Path to the evaluation data."})

    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_token_len: int = 0
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    group_by_modality_length: bool = field(default=False)
    mm_projector_lr: Optional[float] = field(default=None)
    report_to: Optional[str] = field(default="none")

def filter_samples_with_existing_images(samples, image_folder):
    filtered = []
    missing = 0
    for sample in samples:
        image_path = os.path.join(image_folder, sample.get('image', ''))
        if image_path and os.path.exists(image_path):
            filtered.append(sample)
        else:
            missing += 1
    print(f"Number of missing images: {missing}")
    return filtered, missing


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=2048,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _mask_targets(target, tokenized_lens, speakers):
    """
    Masks out tokens corresponding to the human turns (set to IGNORE_INDEX).
    Only assistant (gpt) tokens are left for loss calculation.
    """
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX  # Mask the system prompt/header

    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker.lower() == "human":
            target[cur_idx:cur_idx + tokenized_len] = IGNORE_INDEX
        # else: keep as is (assistant/gpt tokens)
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

# old preprocess
# def preprocess_multimodal(
#     sources: Sequence[str],
#     multimodal_cfg: dict,
#     cur_token_len: int,
# ) -> Dict:
#     is_multimodal = multimodal_cfg['is_multimodal']
#     image_token_len = cur_token_len
#     if not is_multimodal:
#         return sources

#     for source in sources:
#         for sentence in source:
#             # keep image as it is instead of expanding to image patch tokens
#             # replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
#             replace_token = DEFAULT_IMAGE_PATCH_TOKEN
#             if multimodal_cfg['use_im_start_end']:
#                 replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN

#             if isinstance(sentence["value"], int):
#                 sentence["value"] = str(sentence["value"])
#             sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

#     return sources

# new preprocess

def preprocess_multimodal(
    sources: Sequence[str],
    multimodal_cfg: dict,
    cur_token_len: int,
) -> Dict:
    is_multimodal = multimodal_cfg['is_multimodal']
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if isinstance(sentence["value"], int):
                sentence["value"] = str(sentence["value"])
    return sources

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX

    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply the conversation template
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    input_ids = torch.stack([
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        for prompt in conversations
    ], dim=0)

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Iterate per (source, target) pair - fixes the sources[0] bug
    for source, target in zip(sources, targets):
        seq_len = target.size(0)
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        # Cap total_len to actual tensor size for slicing safety
        total_len = min(total_len, seq_len)

        system_msg = conv.system
        system_tokens = tokenizer(system_msg, add_special_tokens=False).input_ids
        system_len = len(system_tokens)

        cur_len = 1
        target[:min(cur_len, seq_len)] = IGNORE_INDEX

        cur_len += system_len
        target[1:min(cur_len, seq_len)] = IGNORE_INDEX

        for j in range(0, len(source), 2):
            if cur_len >= seq_len:
                break

            if j < len(source):
                human_msg = source[j]["value"]
                human_header = conv.sep + conv.roles[0] + ": "
                human_full = human_header + human_msg
                human_tokens = tokenizer(human_full, add_special_tokens=False).input_ids
                human_len = len(human_tokens)

                end = min(cur_len + human_len, seq_len)
                target[cur_len:end] = IGNORE_INDEX
                cur_len = end

            if j + 1 < len(source) and cur_len < seq_len:
                gpt_msg = source[j + 1]["value"]
                gpt_header = conv.sep + conv.roles[1] + ": "
                gpt_full = gpt_header + gpt_msg + conv.sep2
                gpt_tokens = tokenizer(gpt_full, add_special_tokens=False).input_ids
                gpt_len = len(gpt_tokens)

                gpt_header_tokens = tokenizer(gpt_header, add_special_tokens=False).input_ids
                gpt_header_len = len(gpt_header_tokens)

                # Mask the header only; the response body is kept trainable
                end = min(cur_len + gpt_header_len, seq_len)
                target[cur_len:end] = IGNORE_INDEX
                cur_len = end

                response_len = gpt_len - gpt_header_len
                cur_len = min(cur_len + response_len, seq_len)

        # Mask everything past the actual content
        target[cur_len:] = IGNORE_INDEX

    return dict(
        input_ids=input_ids,
        labels=targets,
    )
# def preprocess_v1(
#     sources,
#     tokenizer: transformers.PreTrainedTokenizer,
# ) -> Dict:
#     from llava.mm_utils import tokenizer_image_token
#     from llava.constants import IMAGE_TOKEN_INDEX
    
#     conv = conversation_lib.default_conversation.copy()
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

#     # Apply the conversation template
#     conversations = []
#     for i, source in enumerate(sources):
#         if roles[source[0]["from"]] != conv.roles[0]:
#             source = source[1:]

#         conv.messages = []
#         for j, sentence in enumerate(source):
#             role = roles[sentence["from"]]
#             assert role == conv.roles[j % 2], f"{i}"
#             conv.append_message(role, sentence["value"])
#         conversations.append(conv.get_prompt())

#     # Tokenize conversations using tokenizer_image_token so <image> -> -200
#     input_ids = torch.stack([
#         tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
#         for prompt in conversations
#     ], dim=0)
    
#     targets = input_ids.clone()

#     assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

#     sep = conv.sep + conv.roles[1] + ": "
    
#     for conversation, target in zip(conversations, targets):
#         total_len = int(target.ne(tokenizer.pad_token_id).sum())

#         system_msg = conv.system
#         system_tokens = tokenizer(system_msg, add_special_tokens=False).input_ids
#         system_len = len(system_tokens)
        
#         cur_len = 1
#         target[:cur_len] = IGNORE_INDEX
        
#         cur_len += system_len
#         target[1:cur_len] = IGNORE_INDEX
        
#         for j in range(0, len(sources[0]), 2):
#             if j < len(sources[0]):
#                 human_msg = sources[0][j]["value"]
#                 human_header = conv.sep + conv.roles[0] + ": "
#                 human_full = human_header + human_msg
#                 human_tokens = tokenizer(human_full, add_special_tokens=False).input_ids
#                 human_len = len(human_tokens)
                
#                 if cur_len + human_len <= total_len:
#                     target[cur_len:cur_len + human_len] = IGNORE_INDEX
#                     cur_len += human_len
            
#             if j + 1 < len(sources[0]):
#                 gpt_msg = sources[0][j + 1]["value"]
#                 gpt_header = conv.sep + conv.roles[1] + ": "
#                 gpt_full = gpt_header + gpt_msg + conv.sep2
#                 gpt_tokens = tokenizer(gpt_full, add_special_tokens=False).input_ids
#                 gpt_len = len(gpt_tokens)
                
#                 gpt_header_tokens = tokenizer(gpt_header, add_special_tokens=False).input_ids
#                 gpt_header_len = len(gpt_header_tokens)
                
#                 if cur_len + gpt_header_len <= total_len:
#                     target[cur_len:cur_len + gpt_header_len] = IGNORE_INDEX
#                     cur_len += gpt_header_len
                
#                 response_len = gpt_len - gpt_header_len
#                 cur_len += response_len
        
#         target[cur_len:] = IGNORE_INDEX
        
#         non_ignore = (target != IGNORE_INDEX).sum().item()
#         if non_ignore == 0:
#             print(f"WARNING: All tokens masked!")
#         else:
#             print(f"✓ Sample has {non_ignore}/{total_len} trainable tokens")

#     return dict(
#         input_ids=input_ids,
#         labels=targets,
#     )

# def preprocess_v1(
#     sources,
#     tokenizer: transformers.PreTrainedTokenizer,
# ) -> Dict:
#     conv = conversation_lib.default_conversation.copy()
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

#     # Apply the conversation template
#     conversations = []
#     for i, source in enumerate(sources):
#         if roles[source[0]["from"]] != conv.roles[0]:
#             # Skip the first one if it is not from human
#             source = source[1:]

#         conv.messages = []
#         for j, sentence in enumerate(source):
#             role = roles[sentence["from"]]
#             assert role == conv.roles[j % 2], f"{i}"
#             conv.append_message(role, sentence["value"])
#         conversations.append(conv.get_prompt())

#     # Tokenize conversations
#     input_ids = tokenizer(
#         conversations,
#         return_tensors="pt",
#         padding="longest",
#         max_length=tokenizer.model_max_length,
#         truncation=True,
#     ).input_ids
    
#     targets = input_ids.clone()

#     assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

#     # Mask targets - FIXED VERSION
#     sep = conv.sep + conv.roles[1] + ": "  # " ASSISTANT: "
    
#     for conversation, target in zip(conversations, targets):
#         total_len = int(target.ne(tokenizer.pad_token_id).sum())

#         # Split by sep2 (</s>) is wrong for v1! 
#         # Instead, we need to mask based on role markers
        
#         # Tokenize the system message
#         system_msg = conv.system
#         system_tokens = tokenizer(system_msg, add_special_tokens=False).input_ids
#         system_len = len(system_tokens)
        
#         # Start by masking system message + BOS token
#         cur_len = 1  # BOS token
#         target[:cur_len] = IGNORE_INDEX
        
#         # Mask the system message
#         cur_len += system_len
#         target[1:cur_len] = IGNORE_INDEX
        
#         # Process each turn
#         for j in range(0, len(sources[0]), 2):  # Every pair (human, gpt)
#             # Human turn
#             if j < len(sources[0]):
#                 human_msg = sources[0][j]["value"]
#                 human_header = conv.sep + conv.roles[0] + ": "
#                 human_full = human_header + human_msg
#                 human_tokens = tokenizer(human_full, add_special_tokens=False).input_ids
#                 human_len = len(human_tokens)
                
#                 # Mask the entire human turn
#                 if cur_len + human_len <= total_len:
#                     target[cur_len:cur_len + human_len] = IGNORE_INDEX
#                     cur_len += human_len
            
#             # Assistant turn (DO NOT MASK - this is what we train on)
#             if j + 1 < len(sources[0]):
#                 gpt_msg = sources[0][j + 1]["value"]
#                 gpt_header = conv.sep + conv.roles[1] + ": "
#                 gpt_full = gpt_header + gpt_msg + conv.sep2  # includes </s>
#                 gpt_tokens = tokenizer(gpt_full, add_special_tokens=False).input_ids
#                 gpt_len = len(gpt_tokens)
                
#                 # Mask only the header " ASSISTANT: "
#                 gpt_header_tokens = tokenizer(gpt_header, add_special_tokens=False).input_ids
#                 gpt_header_len = len(gpt_header_tokens)
                
#                 if cur_len + gpt_header_len <= total_len:
#                     target[cur_len:cur_len + gpt_header_len] = IGNORE_INDEX
#                     cur_len += gpt_header_len
                
#                 # Keep the actual response (don't mask)
#                 response_len = gpt_len - gpt_header_len
#                 cur_len += response_len
        
#         # Mask any padding
#         target[cur_len:] = IGNORE_INDEX
        
#         # Debug check
#         non_ignore = (target != IGNORE_INDEX).sum().item()
#         if non_ignore == 0:
#             print(f"WARNING: All tokens masked!")
#             print(f"  Total length: {total_len}, cur_len: {cur_len}")
#             print(f"  Conversation: {conversation[:200]}...")
#         else:
#             print(f"✓ Sample has {non_ignore}/{total_len} trainable tokens")

#     return dict(
#         input_ids=input_ids,
#         labels=targets,
#     )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    1. Add speaker and start/end signal to each sentence.
    2. Concatenate conversations.
    3. Tokenize.
    4. Mask human tokens with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "v1":
        return preprocess_v1(sources, tokenizer)

    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)

    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn(
            [header] + [s["value"] for s in source], tokenizer
        )["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)
    
    return dict(input_ids=input_ids, labels=targets)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 multimodal_cfg: dict):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.multimodal_cfg = multimodal_cfg
        
        if self.multimodal_cfg.get('is_multimodal', False) and self.multimodal_cfg.get('image_folder', None):
            filtered, missing = filter_samples_with_existing_images(self.list_data_dict, self.multimodal_cfg['image_folder'])
            self.list_data_dict = filtered
            self.missing_images = missing
            print(f"Number of missing images (filtered in dataset): {self.missing_images}")
            print(f"Number of valid samples: {len(self.list_data_dict)}")
        else:
            self.missing_images = 0

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"
        
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.multimodal_cfg['image_folder']
            processor = self.multimodal_cfg['image_processor']
            try:
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            except Exception as exn:
                print(exn)
                import random
                return random.choice(self)

            if self.multimodal_cfg['image_aspect_ratio'] == 'keep':
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 448, 224
                shortest_edge = int(min(max_len / aspect_ratio, min_len))
                image = processor.preprocess(image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge})['pixel_values'][0]
            elif self.multimodal_cfg['image_aspect_ratio'] == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            image_token_len = self.multimodal_cfg["image_token_len"]
            patch_size = int(image.shape[1]//math.sqrt(image_token_len))
            cur_token_len = (image.shape[1]//patch_size) * (image.shape[2]//patch_size)

            try:
                sources = copy.deepcopy([e["conversations"] for e in sources])
            except:
                sources = copy.deepcopy([e["conversations"] for e in sources])

            sources = preprocess_multimodal(
                sources,
                self.multimodal_cfg, cur_token_len)
        else:

            try:
                sources = copy.deepcopy([e["conversations"] for e in sources])
            except:
                sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.multimodal_cfg['is_multimodal']:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.multimodal_cfg['image_processor'].crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (LazySupervisedDataset
                   if data_args.lazy_preprocess else SupervisedDataset)
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                multimodal_cfg=dict(
                                    is_multimodal=data_args.is_multimodal,
                                    image_token_len=data_args.image_token_len,
                                    image_folder=data_args.image_folder,
                                    image_aspect_ratio=data_args.image_aspect_ratio,
                                    use_im_start_end=getattr(data_args, 'mm_use_im_start_end', False),
                                    image_processor=getattr(data_args, 'image_processor', None)))
    eval_dataset = None
    if getattr(data_args, 'eval_data_path', None):
        eval_dataset = dataset_cls(tokenizer=tokenizer,
                                   data_path=data_args.eval_data_path,
                                   multimodal_cfg=dict(
                                       is_multimodal=data_args.is_multimodal,
                                       image_token_len=data_args.image_token_len,
                                       image_folder=data_args.image_folder,
                                       image_aspect_ratio=data_args.image_aspect_ratio,
                                       use_im_start_end=getattr(data_args, 'mm_use_im_start_end', False),
                                       image_processor=getattr(data_args, 'image_processor', None)))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if model_args.vision_tower is not None:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
    model.config.use_cache = False

    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    if model_args.vision_tower is not None:
        mm_init = getattr(model, "initialize_vision_modules", None) \
              or getattr(getattr(model, "model", None), "initialize_vision_modules", None)
        assert mm_init is not None, "initialize_vision_modules not found on model or model.model"
        model_vision_dict = mm_init(model_args)
        dtype = torch.float32
        if training_args.fp16:
            dtype = torch.float16
        if training_args.bf16:
            dtype = torch.bfloat16
        model.model.vision_tower.to(dtype=dtype, device=training_args.device)
        
        if model_vision_dict is not None and 'vision_config' in model_vision_dict:
            vision_config = model_vision_dict['vision_config']
        else:
            raise ValueError("model_vision_dict is None or missing 'vision_config'")

        data_args.image_token_len = model_vision_dict['image_token_len']
        data_args.image_processor = model_vision_dict['image_processor']
        data_args.is_multimodal = True

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.model.mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.model.mm_projector.parameters():
                p.requires_grad = False

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        vision_config.use_im_start_end = training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.initialize_vision_tokenizer(model_args, tokenizer)

        params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
        if len(params_no_grad) > 0:
            if training_args.fsdp is not None and len(training_args.fsdp) > 0:
                if len(params_no_grad) < 10:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
                else:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(len(params_no_grad), ', '.join(params_no_grad[:10])))
                print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
                print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
                def patch_FSDP_use_orig_params(func):
                    def wrap_func(*args, **kwargs):
                        use_orig_params = kwargs.pop('use_orig_params', True)
                        return func(*args, **kwargs, use_orig_params=use_orig_params)
                    return wrap_func

                FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters:", num_trainable)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=False)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)
    
    if hasattr(data_module['train_dataset'], 'missing_images'):
        print(f"Number of missing images (after training): {data_module['train_dataset'].missing_images}")

if __name__ == "__main__":
    train()


