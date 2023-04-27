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

"""Transformer."""

import math
import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import mpu
from megatron.mpu import LayerNorm
from megatron.module import MegatronModule
from megatron.checkpointing import get_checkpoint_version
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.utils import openai_gelu, erf_gelu

import deepspeed

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
        attention_mask_func: a function that takes `unmaksed-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
               masked-attention-scores = attention_mask_func(
                                     unmaksed-attention-scores, attention-mask)
"""

class SparseDispatcher(object):

    def __init__(self, num_experts, gates):

        self._gates = gates
        self._num_experts = num_experts
        # sort experts : rows of (batch index, expert index), there could be multiple experts per batch
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        # print(expert_out)
        # print(expert_out.shape)

        for n in range(len(expert_out)) :
            expert_out[n] = expert_out[n].squeeze(0)

            # eout = eout.squeeze(0)

        for eout in expert_out :
            print(eout.shape)

        import numpy as np

        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MoE(torch.nn.Module):

    def __init__(self, input_size, output_size, num_experts, intermediate_size, noisy_gating=True, k=4, expert_module=None):
        super(MoE, self).__init__()
        args = get_args()

        self.noisy_gating = noisy_gating
        self.num_experts = num_experts

        assert output_size == input_size, "input size should match output size"
        self.output_size = output_size
        self.input_size = input_size
        self.intermediate_size = intermediate_size
        self.k = k

        import copy
        self.experts    = torch.nn.ModuleList([copy.deepcopy(expert_module) for i in range(self.num_experts)])
        self.w_gate     = torch.nn.Parameter(torch.rand(input_size, num_experts, device=torch.cuda.current_device()), requires_grad=True)
        self.w_noise    = torch.nn.Parameter(torch.rand(input_size, num_experts, device=torch.cuda.current_device()), requires_grad=True)
        self.softplus   = torch.nn.Softplus()
        self.softmax    = torch.nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

        self.time_record_mlp = None

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        from torch.distributions.normal import Normal
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate # matmul (batch,seq_len,hidden) x (hidden,experts)
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        mode=0
        if mode == 0:

            # TODO : consider batch_size and sequence size separately
            x_shape = x.shape
            
            x = x.reshape((-1, self.output_size))

            gates, load = self.noisy_top_k_gating(x, self.training)
            importance = gates.sum(0)
            loss = self.cv_squared(importance) + self.cv_squared(load)
            loss *= loss_coef

            dispatcher = SparseDispatcher(self.num_experts, gates) # configure with batch should be dispatched to each expert
            expert_inputs = dispatcher.dispatch(x) # distribute batch inputs for each expert
            gates = dispatcher.expert_to_gates() # the weight for each batch on each expert

            expert_outputs = [self.experts[i](expert_inputs[i].unsqueeze(0)) for i in range(self.num_experts)] # for each expert, inference on assorted batch
            
            if isinstance(expert_outputs[0],tuple) and len(expert_outputs[0]) == 2:
                _, mlp_bias = expert_outputs[0]
                expert_outputs_only = [ex_out for ex_out, _ in expert_outputs]

                y = dispatcher.combine(expert_outputs_only) # get weighted sum of expert outputs
                y = y.reshape(x_shape)
                return y , mlp_bias

            else :
                y = dispatcher.combine(expert_outputs) # get weighted sum of expert outputs
                y = y.reshape(x_shape)
                return y, loss
                # return y #, loss

        elif mode == 1:
            import time
            times=[]

            # reshape1
            times.append(time.time())
            x_shape = x.shape
            x = x.reshape((-1, self.output_size))

            # topk
            times.append(time.time())
            gates, load = self.noisy_top_k_gating(x, self.training)

            # imp1
            times.append(time.time())
            importance = gates.sum(0)

            # imp2
            times.append(time.time())
            loss = self.cv_squared(importance) + self.cv_squared(load)
            loss *= loss_coef

            # init_dispatch
            times.append(time.time())
            dispatcher = SparseDispatcher(self.num_experts, gates) # configure with batch should be dispatched to each expert

            # dispatch
            times.append(time.time())
            expert_inputs = dispatcher.dispatch(x) # distribute batch inputs for each expert

            # ex_gate
            times.append(time.time())
            gates = dispatcher.expert_to_gates() # the weight for each batch on each expert

            # ex_output
            times.append(time.time())
            expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)] # for each expert, inference on assorted batch

            # ex_combine
            times.append(time.time())
            y = dispatcher.combine(expert_outputs) # get weighted sum of expert outputs

            # reshape2
            times.append(time.time())
            y = y.reshape(x_shape)

            times.append(time.time())
            self.time_record_mlp = times

            # return y, loss
            return y, loss
        '''
        elif mode == 2:
            with profiler.record_function("prof_noisy_top_k_gating"):
                gates, load = self.noisy_top_k_gating(x, self.training)

            # calculate importance loss
            with profiler.record_function("prof_get_importance"):
                importance = gates.sum(0)
            #
            with profiler.record_function("prof_get_loss"):
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    loss = self.cv_squared(importance) + self.cv_squared(load)
                    loss *= loss_coef

            with profiler.record_function("prof_dispatch"):
                dispatcher = SparseDispatcher(self.num_experts, gates) # configure with batch should be dispatched to each expert
                expert_inputs = dispatcher.dispatch(x) # distribute batch inputs for each expert

            with profiler.record_function("prof_expert_to_gates"):
                gates = dispatcher.expert_to_gates() # the weight for each batch on each expert

            with profiler.record_function("prof_expert_output"):
                expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)] # for each expert, inference on assorted batch

            with profiler.record_function("prof_expert_combine"):
                y = dispatcher.combine(expert_outputs) # get weighted sum of expert outputs

            return y, loss
        else : assert 0, "Unknown Mode Selected"
        '''

class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(self, init_method, output_layer_init_method):
        super(ParallelMLP, self).__init__()
        args = get_args()

        # Project to 4h.
        if not args.memory_centric_tiled_linear:
            self.dense_h_to_4h = mpu.ColumnParallelLinear(
                args.hidden_size,
                4 * args.hidden_size,
                gather_output=False,
                init_method=init_method,
                skip_bias_add=True)
        else:
            self.dense_h_to_4h = deepspeed.zero.TiledLinearReturnBias(
                in_features=args.hidden_size,
                out_features=4*args.hidden_size,
                linear_cls=mpu.ColumnParallelLinear,
                in_splits=args.tile_factor,
                out_splits=4*args.tile_factor,
                combine_out_splits=True,
                gather_output=False,
                init_method=init_method,
                skip_bias_add=True)

        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu
        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu

        # Project back to h.
        if not args.memory_centric_tiled_linear:
            self.dense_4h_to_h = mpu.RowParallelLinear(
                4 * args.hidden_size,
                args.hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method,
                skip_bias_add=True)
        else:
            self.dense_4h_to_h = deepspeed.zero.TiledLinearReturnBias(
                in_features=4*args.hidden_size,
                out_features=args.hidden_size,
                linear_cls=mpu.RowParallelLinear,
                in_splits=4*args.tile_factor,
                out_splits=args.tile_factor,
                input_is_already_split=False,
                combine_out_splits=True,
                input_is_parallel=True,
                init_method=output_layer_init_method,
                skip_bias_add=True)
         
    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
            intermediate_parallel = \
                    bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = \
                self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class ParallelSelfAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, attention_mask_func, init_method,
                 output_layer_init_method, layer_number):
        super(ParallelSelfAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16

        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        # Per attention head and per partition values.
        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(args.hidden_size,
                                                    world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            args.hidden_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(
            args.num_attention_heads, world_size)

        # Strided linear layer.
        if not args.memory_centric_tiled_linear:
            self.query_key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                3 * args.hidden_size,
                gather_output=False,
                init_method=init_method)
        else:
            self.query_key_value = deepspeed.zero.TiledLinearReturnBias(
                in_features=args.hidden_size,
                out_features=3*args.hidden_size,
                linear_cls=mpu.ColumnParallelLinear,
                gather_output=False,
                init_method=init_method,
                in_splits=args.tile_factor,
                out_splits=3*args.tile_factor,
                combine_out_splits=True
            )

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16,
            args.scaled_upper_triang_masked_softmax_fusion,
            args.scaled_masked_softmax_fusion,
            self.attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

        # Output.
        if not args.memory_centric_tiled_linear:
            self.dense = mpu.RowParallelLinear(
                args.hidden_size,
                args.hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method,
                skip_bias_add=True)
        else:
            self.dense = deepspeed.zero.TiledLinearReturnBias(
                in_features=args.hidden_size,
                out_features=args.hidden_size,
                linear_cls=mpu.RowParallelLinear,
                input_is_parallel=True,
                init_method=output_layer_init_method,
                skip_bias_add=True,
                out_splits=args.tile_factor,
                in_splits=args.tile_factor,
                combine_out_splits=True
            )


        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def _transpose_last_dim(self, mixed_layer, num_splits, num_splits_first):
        input_shape = mixed_layer.size();
        if num_splits_first:
            """[s, b, num_splits * np * hn] 
            -->(view) [s, b, num_splits, np, hn] 
            -->(tranpose) [s, b, np, num_splits, hn] 
            -->(view) [s, b, np * num_splits * hn] """

            intermediate_shape = input_shape[:-1] +\
                (num_splits, self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head)

            mixed_layer = mixed_layer.view(*intermediate_shape)
            mixed_layer = mixed_layer.transpose(-2, -3).contiguous()
        else:
            """[s, b, np * hn * num_splits] 
            -->(view) [s, b, np, hn, num_splits] 
            -->(tranpose) [s, b, np, num_splits, hn] 
            -->(view) [s, b, np * num_splits * hn] """

            intermediate_shape = input_shape[:-1] +\
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head, num_splits)

            mixed_layer = mixed_layer.view(*intermediate_shape)
            mixed_layer = mixed_layer.transpose(-1, -2).contiguous()
        mixed_layer = mixed_layer.view(*input_shape)
        
        return mixed_layer

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False):
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        checkpoint_version = get_checkpoint_version()
        if checkpoint_version is not None:
           if checkpoint_version == 0:
               # [s, b, (3 * np * hn)] --> [s, b, (np * 3 * hn)]
               mixed_x_layer = self._transpose_last_dim(mixed_x_layer, 3, True)
           elif checkpoint_version == 1.0:
               # [s, b, (np * hn * 3)] --> [s, b, (np * 3 * hn)]
               mixed_x_layer = self._transpose_last_dim(mixed_x_layer, 3, False)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
             3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer,
         key_layer,
         value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=0)
        if get_key_value:
            present = (key_layer, value_layer)


        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================
        
        # [b, np, sq, sk]
        output_size = (query_layer.size(1), 
                       query_layer.size(2), 
                       query_layer.size(0), 
                       key_layer.size(0))
        
        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0]*output_size[1], 
            output_size[2], 
            output_size[3],
            dtype=query_layer.dtype, 
            device=torch.cuda.current_device())

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(matmul_result, 
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0,1).transpose(1, 2),  #[b * np, hn, sk]
            beta=0.0, alpha=(1.0/self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)


        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[
                        ...,
                        attention_scores.size(3) - 1,
                        :attention_scores.size(3)].unsqueeze(2)
                else:
                    attention_mask = attention_mask[
                        ...,
                        :attention_scores.size(3),
                        :attention_scores.size(3)]


        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)


        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1), 
                       value_layer.size(2), 
                       query_layer.size(0), 
                       value_layer.size(3)) 

        # change view [sk, b * np, hn] 
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)
        
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)
        
        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0,1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)


        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output, bias


def bias_dropout_add(x, bias, residual, prob, training) :
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x, bias, residual, prob) :
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x, bias, residual, prob) :
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, False)


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, attention_mask_func, init_method, 
                 output_layer_init_method, layer_number):
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        # Memory-saving optimization
        self.scattered_attn_output = args.scattered_embeddings

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        # Self attention.
        self.attention = ParallelSelfAttention(attention_mask_func, init_method,
                                               output_layer_init_method,
                                               layer_number)
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        # MLP
        self.mlp = ParallelMLP(init_method,
                               output_layer_init_method)


    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False):
        # hidden_states: [b, s, h]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, attention_bias = \
            self.attention(layernorm_output,
                           attention_mask,
                           layer_past=layer_past,
                           get_key_value=get_key_value)

        if get_key_value:
            attention_output, presents = attention_output

        if self.scattered_attn_output:
            attention_output = mpu.scatter_to_model_parallel_region(attention_output)
            attention_bias = mpu.scatter_to_model_parallel_region(attention_bias)
    
        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        if self.scattered_attn_output:
            residual = mpu.scatter_to_model_parallel_region(residual)

        # jit scripting for a nn.module (with dropout) is not 
        # trigerring the fusion kernel. For now, we use two 
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        #re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias.expand_as(residual),
                residual,
                self.hidden_dropout)

        # Collect the scattered result from the fused dropout.
        if self.scattered_attn_output:
            layernorm_input = mpu.gather_from_model_parallel_region(layernorm_input)
            # Attention output/bias are not used again, so no need to gather

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)
        
        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        #re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            output = bias_dropout_add_func(
                mlp_output,
                mlp_bias.expand_as(residual),
                residual,
                self.hidden_dropout)

        if get_key_value:
            output = [output, presents]

        return output

class ParallelTransformerLayerPart1(MegatronModule):
    """A single transformer layer.

    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, attention_mask_func, init_method, 
                 output_layer_init_method, layer_number):
        args = get_args()

        super(ParallelTransformerLayerPart1, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        # Self attention.
        self.attention = ParallelSelfAttention(attention_mask_func, init_method,
                                               output_layer_init_method,
                                               layer_number)
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion


    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False):
        # hidden_states: [b, s, h]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, attention_bias = \
            self.attention(layernorm_output,
                           attention_mask,
                           layer_past=layer_past,
                           get_key_value=get_key_value)

        presents = None
        if get_key_value:
            raise NotImplementedError('get_key_value param is not yet supported with split-transformers')
            attention_output, presents = attention_output

    
        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        if self.scattered_attn_output:
            residual = mpu.scatter_to_model_parallel_region(residual)

        # jit scripting for a nn.module (with dropout) is not 
        # trigerring the fusion kernel. For now, we use two 
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        #re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias.expand_as(residual),
                residual,
                self.hidden_dropout)

        return layernorm_input

class ParallelTransformerLayerPart2(MegatronModule):
    """A single transformer layer.

    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, attention_mask_func, init_method, 
                 output_layer_init_method, layer_number):
        args = get_args()

        super(ParallelTransformerLayerPart2, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        # Memory-saving optimization
        self.scattered_attn_output = args.scattered_embeddings

        # MLP
        # is_moe=False
        is_moe=True

        if is_moe :
            # CONFIGURE MOE PARAMETERS
            hidden_size = args.hidden_size
            ffn_hidden_size = 4*hidden_size
            num_experts = 5
            k = 1

            self.mlp = MoE(hidden_size,hidden_size, 
                            num_experts,ffn_hidden_size, 
                            noisy_gating=True, k=k,
                            expert_module=ParallelMLP(init_method, output_layer_init_method))
        else :
            self.mlp = ParallelMLP(init_method,
                                output_layer_init_method)

        pass


    def forward(self, layernorm_input, attention_mask, presents=None, layer_past=None,
                get_key_value=False):
        # hidden_states: [b, s, h]
        
        # Collect the scattered result from the fused dropout.
        if self.scattered_attn_output:
            layernorm_input = mpu.gather_from_model_parallel_region(layernorm_input)
            # Attention output/bias are not used again, so no need to gather

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP. input [seq_len,batch,embedding size]
        mlp_output, mlp_bias = self.mlp(layernorm_output)
        
        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        # jit scripting for a nn.module (with dropout) is not 
        # trigerring the fusion kernel. For now, we use two 
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        #re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            output = bias_dropout_add_func(
                mlp_output,
                mlp_bias.expand_as(residual),
                residual,
                self.hidden_dropout)

        if get_key_value:
            output = [output, presents]

        return output

class ParallelTransformerLayerPart1(MegatronModule):
    """A single transformer layer.

    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, attention_mask_func, init_method, 
                 output_layer_init_method, layer_number):
        args = get_args()

        super(ParallelTransformerLayerPart1, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        # Self attention.
        self.attention = ParallelSelfAttention(attention_mask_func, init_method,
                                               output_layer_init_method,
                                               layer_number)
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion


    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False):
        # hidden_states: [b, s, h]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, attention_bias = \
            self.attention(layernorm_output,
                           attention_mask,
                           layer_past=layer_past,
                           get_key_value=get_key_value)

        presents = None
        if get_key_value:
            raise NotImplementedError('get_key_value param is not yet supported with split-transformers')
            attention_output, presents = attention_output
    
        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # jit scripting for a nn.module (with dropout) is not 
        # trigerring the fusion kernel. For now, we use two 
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        #re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias.expand_as(residual),
                residual,
                self.hidden_dropout)

        return layernorm_input

class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, attention_mask_func,
                 init_method, output_layer_init_method):
        super(ParallelTransformer, self).__init__()
        args = get_args()

        # Store activation checkpoiting flag.
        self.checkpoint_activations = args.checkpoint_activations
        self.checkpoint_num_layers = args.checkpoint_num_layers

        # Number of layers:
        self.num_layers = args.num_layers
        self.num_unique_layers = args.num_unique_layers
        if self.num_unique_layers is None:
            self.num_unique_layers = self.num_layers
        assert self.num_layers % self.num_unique_layers == 0, \
            'number of layers should be divisible by number of unique layers'
        self.param_sharing_style = args.param_sharing_style

        # Transformer layers.
        def build_layer(layer_number):
            return ParallelTransformerLayer(
                attention_mask_func, init_method,
                output_layer_init_method, layer_number)

        def build_layer_part1(layer_number):
            return ParallelTransformerLayerPart1(
                attention_mask_func, init_method,
                output_layer_init_method, layer_number)
        def build_layer_part2(layer_number):
            return ParallelTransformerLayerPart2(
                attention_mask_func, init_method,
                output_layer_init_method, layer_number)

        if args.split_transformers:
            layers = []
            for i in range(self.num_unique_layers):
                layers.append(build_layer_part1(i + 1))
                layers.append(build_layer_part2(i + 1))
            self.layers = torch.nn.ModuleList(layers)
            self.num_layers *= 2
            self.num_unique_layers *= 2
        else:
            self.layers = torch.nn.ModuleList(
                [build_layer(i + 1) for i in range(self.num_unique_layers)])

        # Print layer ordering.
        if self.num_layers != self.num_unique_layers:
            if torch.distributed.get_rank() == 0:
                print('> will be using the following layer ordering:')
                for i in range(self.num_layers):
                    print('   layer id: {:3d} --> unique layer id: '
                          '{:3d}'.format(i, self._get_layer_index(i)),
                          flush=True)

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def _get_layer_index(self, layer_number):
        if self.param_sharing_style == 'grouped':
            return layer_number % self.num_unique_layers
        if self.param_sharing_style == 'spaced':
            return layer_number // (self.num_layers // self.num_unique_layers) 
        assert False, 'should not be here'

    def _get_layer(self, layer_number):
        return self.layers[self._get_layer_index(layer_number)]

    def _checkpointed_forward(self, hidden_states, attention_mask):
        """Forward method with activation checkpointing."""
        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, inputs[1])
                return x_
            return custom_forward

        # Make sure memory is freed.
        mpu.reset_checkpointed_activations_memory_buffer()
        l = 0
        while l < self.num_layers:
            hidden_states = mpu.checkpoint(
                custom(l, min(l + self.checkpoint_num_layers, self.num_layers)),
                hidden_states, attention_mask)
            l += self.checkpoint_num_layers

        return hidden_states

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False):

        # Checks
        if layer_past is not None:
            assert get_key_value, \
                'for not None values in layer_past, ' \
                'expected get_key_value to be set'
        if get_key_value:
            assert not self.checkpoint_activations, \
                'get_key_value does not work with ' \
                'activation checkpointing'

        # data format change to avoid explicit tranposes : [b s h] --> [s b h]
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        if self.checkpoint_activations:
            hidden_states = self._checkpointed_forward(hidden_states,
                                                       attention_mask)
        else:
            if get_key_value:
                presents = []
            for index in range(self.num_layers):
                layer = self._get_layer(index)
                past = None
                if layer_past is not None:
                    past = layer_past[index]
                hidden_states = layer(hidden_states,
                                      attention_mask,
                                      layer_past=past,
                                      get_key_value=get_key_value)
                if get_key_value:
                    hidden_states, present = hidden_states
                    presents.append(present)
        
        # reverting data format change [s b h] --> [b s h]
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        # Final layer norm.
        output = self.final_layernorm(hidden_states)
        if get_key_value:
            output = [output, presents]

        return output
