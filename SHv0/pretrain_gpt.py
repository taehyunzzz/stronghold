# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain GPT"""
# import sys
# def trace(frame, event, arg):
#     print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
#     return trace
# sys.settrace(trace)

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel, ModelType
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group

from megatron.utils import get_parameters_in_billions, _unwrap_model


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0("building GPT model ...")
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
    )

    # if mpu.get_data_parallel_rank() == 0:
    #     billion_params = get_parameters_in_billions(model)
    #     print(f' > number of parameters on pipeline model parallel rank {mpu.get_pipeline_model_parallel_rank()}, \
    #         tensor model parallel rank {mpu.get_tensor_model_parallel_rank()} \
    #         {round(billion_params, 3)} Billion',
    #         flush=True)

    # unwrapped_model = _unwrap_model(model)
    # _param_count = lambda m: sum([_.numel() for _ in m.parameters()])
    # _param_sum = _param_count(unwrapped_model)
    # for module_name, module in unwrapped_model.named_modules():
    #     print(f"{module_name}:\n\tparam_count={_param_count(module)}, param_ratio={round(_param_count(module)/_param_sum * 100, 4)}%")

    # def model2moe(model, hidden_size, intermediate_size, num_experts, k):
    #     print("Converting to MoE model")

    #     mlp_list = []
    #     for n,p in model.named_modules():
    #         if n.split(".")[-1] == "mlp":
    #             mlp_list.append((n,p))

    #     for mlp_idx in range(len(mlp_list)):
    #         mlp_layer = mlp_list[mlp_idx]
    #         model_obj = model
    #         for n, word in enumerate(mlp_layer[0].split(".")):
    #             if word == "mlp":
    #                 from moe import MoE
    #                 setattr(model_obj, word,
    #                     MoE(hidden_size, hidden_size, 
    #                         num_experts, intermediate_size, 
    #                         k=k, noisy_gating=True,
    #                         expert_module=mlp_layer[1]
    #                         ))
    #                 break
    #             elif word.isnumeric():
    #                 model_obj = model_obj[int(word)]
    #             else :
    #                 model_obj = getattr(model_obj, word)
    #     total=0
    #     for n,p in model.named_parameters():
    #         total += p.numel()
    #     # print(total)

    #     return model

    # args = get_args()
    # hidden_size = args.hidden_size
    # intermediate_size = args.ffn_hidden_size
    # model = model2moe(model, 
    #                   hidden_size,
    #                   intermediate_size,
    #                   num_experts=10, 
    #                   k=1
    #                   )

    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ["text"]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b["text"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {"lm loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator").start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers("batch-generator").stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0("> building train, validation, and test datasets " "for GPT ...")
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
    )
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )