# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from contextlib import nullcontext
import itertools

import torch
import torch.distributed
from lightning.pytorch.trainer.trainer import Trainer
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.utils import divide
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
    get_ltor_masks_and_position_ids,
)
from nemo.collections.nlp.parts.mixins.nlp_adapter_mixins import NLPAdapterModelMixin
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.collections.llm.gpt.model.base import get_packed_seq_params
from nemo.utils import logging

from nemo_aligner.models.alignable_interface import AlignableGenerativeInterface
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.distributed import (
    broadcast_2d_tensor_within_pp,
    calculate_distributed_entropy,
    from_parallel_logits_to_logprobs,
)
from nemo_aligner.utils.ppo_utils import calculate_kl_penalty_joschu2020
from nemo_aligner.utils.text_generation_utils import (
    TrackLengthGPTModelTextGenerationStrategy,
    verify_is_valid_and_clamp_range_,
)
from nemo_aligner.utils.train_utils import (
    grad_reductions,
    prepare_for_training_step,
    set_eval,
    set_sync_funcs,
    set_train,
)
from nemo_aligner.utils.trt_llm import GPTGenerateTRTLLM
from nemo_aligner.utils.utils import (
    adapter_control,
    clear_memory,
    configure_batch_sizes,
    cpu_weight_swap,
    masked_mean,
    offload_distributed_adam,
)


class MegatronGPTReinforceActorModel(NLPAdapterModelMixin, MegatronGPTModel, AlignableGenerativeInterface):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.automatic_optimization = False

        self.init_policy_state_dict = None
        self.distributed_adam_offload_manager = None

        # length parameters for generation
        self._length_params = OmegaConf.to_container(self.cfg.reinforce.length_params, resolve=True)
        # sampling parameters for generation
        self._sampling_params = OmegaConf.to_container(self.cfg.reinforce.sampling_params, resolve=True)

        self.to_offload_adam_states = self.cfg.reinforce.offload_adam_states and self.with_distributed_adam
        self.forward_micro_batch_size = self.cfg.reinforce.forward_micro_batch_size

        self.use_trtllm_generation = "trt_llm" in self.cfg.reinforce and self.cfg.reinforce.trt_llm.enable
        if self.use_trtllm_generation:
            self.trtllm_generate = GPTGenerateTRTLLM(
                model_cfg=self.cfg,
                max_generation_length=self.cfg.reinforce.length_params.get("max_length", 1024),
                max_input_len=self.cfg.reinforce.trt_llm.get("max_input_len", 1024),
                generation_batch_size=self.cfg.reinforce.get("rollout_micro_batch_size", 4)
                * self.cfg.reinforce.get("prompt_rollouts_per_microbatch", 1),
                unload_engine_train=self.cfg.reinforce.trt_llm.get("unload_engine_train", False),
                trt_model_type=self.cfg.reinforce.trt_llm.get("model_type", "llama"),
                end_strings=self.cfg.reinforce.sampling_params["end_strings"],
                reshard_model=self.cfg.reinforce.trt_llm.get("reshard", False),
                sample_temperature=self.cfg.reinforce.sampling_params["temperature"],
                sample_top_k=self.cfg.reinforce.sampling_params["top_k"],
                sample_top_p=self.cfg.reinforce.sampling_params["top_p"],
                repetition_penalty=self.cfg.reinforce.sampling_params["repetition_penalty"],
                use_greedy=self.cfg.reinforce.sampling_params.get("use_greedy", False),
                tokenizer=self.tokenizer,
                seed=self.cfg.reinforce.trt_llm.get("seed", self.cfg.seed),
            )

    # training calls
    def get_actor_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(data_iterator, model):
            _batch = next(data_iterator)

            torch.distributed.breakpoint()

            required_keys = set()
            if parallel_state.is_pipeline_first_stage():
                required_keys.add("response_tokens")
            if parallel_state.is_pipeline_last_stage():
                required_keys.update(("response_tokens", "baseline", "mask", "is_end", "init_policy_kl", "init_log_probs", "rewards", "prompt_mask", "log_probs"))

            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in _batch.items()}
                
            # input_ids =batch['response_tokens']
            # packed_seq_params = None


            # import megatron.core
            # megatron.core.models.gpt.gpt_model.debug_flag = True


            # parallel_logits = model(
            #     input_ids=input_ids, 
            #     position_ids=None, 
            #     attention_mask=None, 
            #     packed_seq_params=None,
            #     labels=None,
            # )
            packing = True
            if packing:
                cu_seqlens = [0]
                for resp_length in _batch["response_lengths"]:
                    cu_seqlens.append(cu_seqlens[-1] + resp_length)
                
                _batch["cu_seqlens"] = torch.tensor(cu_seqlens, dtype=torch.int, device=torch.cuda.current_device())
                _batch["cu_seqlens_argmin"] = torch.tensor(len(_batch["response_lengths"])+1, dtype=torch.int)
                _batch["max_seqlen"] = torch.tensor(max(_batch["response_lengths"]), dtype=torch.int)

                packed_seq_params = get_packed_seq_params(_batch)
                response_tokens = torch.cat([seq[:length] for seq, length in zip(_batch["response_tokens"], _batch["response_lengths"])])
                input_ids = torch.unsqueeze(response_tokens, dim=0)
            else:
                packed_seq_params = None
                input_ids = _batch["response_lengths"]

            parallel_logits = model(
                input_ids=input_ids, 
                position_ids=None, 
                attention_mask=None, 
                packed_seq_params=packed_seq_params,
                labels=None,
            )

            if packing:
                parallel_logits = torch.split(parallel_logits.squeeze(), tuple(_batch["response_lengths"]), dim=0)
                parallel_logits = torch.nn.utils.rnn.pad_sequence(parallel_logits, batch_first=True)
                #pad back to the global max seqlen in the batch, #TODO add packed seqlen loss
                seq_dim = 1
                global_seqlen = _batch["response_tokens"].shape[seq_dim]
                pad_amount = global_seqlen - parallel_logits.shape[seq_dim]
                parallel_logits = torch.nn.functional.pad(parallel_logits, pad=(0, 0, pad_amount, 0))

            def loss_func(parallel_logits):
                mask = batch["mask"]
                # rewards_with_kl = batch["rewards_with_kl"]
                rewards = batch["rewards"]
                init_policy_kl = batch["init_policy_kl"]
                init_log_probs = batch["init_log_probs"]
                generation_log_probs = batch["log_probs"]
                baseline = batch["baseline"]
                tokens = batch["response_tokens"]
                is_end = batch["is_end"]
                prompt_mask = batch["prompt_mask"]

                is_end_mask = mask * is_end.view(-1, 1)

                curr_log_probs = from_parallel_logits_to_logprobs(
                    vocab_parallel_logits=parallel_logits, target=tokens, higher_stability=True
                )

                # scaled_entropy = torch.tensor(0.0, dtype=parallel_logits.dtype, device=parallel_logits.device)
                # if self.entropy_bonus > 0:
                #    scaled_entropy = calculate_distributed_entropy(parallel_logits, is_end_mask) * self.entropy_bonus

                # reinforce_loss = -1 * curr_log_probs * (rewards_with_kl.unsqueeze(-1) - baseline.unsqueeze(-1))
                kl = self.cfg.reinforce.initial_policy_kl_penalty * calculate_kl_penalty_joschu2020(
                    log_probs_policy=curr_log_probs, log_probs_reference=init_log_probs
                )
                if self.cfg.reinforce.use_grpo_loss:
                    # GRPO
                    assert (
                        self.cfg.reinforce.disable_baseline == False
                    ), "baseline disable is not supported with grpo loss"

                    advantages = rewards.unsqueeze(-1) - baseline.unsqueeze(-1)
                    ratios = (curr_log_probs - generation_log_probs).exp()
                    ratios_clamped = ratios.clamp(1.0 - self.cfg.reinforce.grpo_eps, 1.0 + self.cfg.reinforce.grpo_eps)

                    unclamped_loss = -advantages * ratios
                    clamped_loss = -advantages * ratios_clamped
                    reinforce_loss = torch.max(unclamped_loss, clamped_loss)
                else:
                    # RLOO/REINFORCE
                    if self.cfg.reinforce.disable_baseline:
                        reinforce_loss = -1 * curr_log_probs * (rewards.unsqueeze(-1)) + kl
                    else:
                        reinforce_loss = -1 * curr_log_probs * (rewards.unsqueeze(-1) - baseline.unsqueeze(-1)) + kl

                if is_end_mask.sum() > 0:
                    loss = masked_mean(reinforce_loss, mask * prompt_mask.view(-1, 1))
                else:
                    # hack to disable this update since there are no valid tokens
                    loss = reinforce_loss.view(-1)[0] * 0

                reduced_actor_loss = average_losses_across_data_parallel_group([loss])
                return (
                    loss,
                    {"loss": reduced_actor_loss,},
                )

            return parallel_logits, loss_func

        return fwd_output_and_loss_func

    def prepare_for_training(self):
        configure_batch_sizes(
            mbs=self.cfg.micro_batch_size,
            gbs=self.cfg.global_batch_size,
            dp=parallel_state.get_data_parallel_world_size(),
        )
        self.onload_adam_states()

    def prepare_for_training_step(self):
        # custom trainers will always zero grad for us
        prepare_for_training_step(self, zero_grad=False)

    def get_loss_and_metrics(self, batch, forward_only):
        packing = True
        if not packing:
            sequence_length = batch["response_tokens"].size(1)

            attention_mask, _, position_ids = self.get_ltor_masks_and_position_ids(tokens=batch["response_tokens"])
            batch["attention_mask"] = attention_mask
            batch["position_ids"] = position_ids

            data_iter = get_iterator_k_split(batch, get_num_microbatches())
            num_microbatches = get_num_microbatches()
        else:
            num_microbatches = len(batch)
            data_iter = iter(batch)

            
        set_sync_funcs(self, forward_only)
        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_actor_forward_output_and_loss_func(),
            data_iterator=self._make_data_iterator_list(data_iter),
            model=self.model,
            num_microbatches=num_microbatches,
            forward_only=forward_only,
            seq_length=None,
            micro_batch_size=self.cfg.micro_batch_size,
        )

        metrics = {}

        for key in ["loss"]:
            if losses_reduced_per_micro_batch:
                metric_mean = torch.stack(
                    [loss_reduced[key] for loss_reduced in losses_reduced_per_micro_batch]
                ).mean()
            else:
                metric_mean = torch.tensor(0.0, device=torch.cuda.current_device())

            torch.distributed.broadcast(metric_mean, get_last_rank())

            metrics[key] = metric_mean.cpu().item()

        return metrics["loss"], metrics

    def finish_training_step(self):
        grad_reductions(self)

    def finish_training(self):
        """no need to offload adam states here
        """


    def pack_and_split_batch(self, batch, tokens_per_mbs_per_gpu=4096):
        total_tokens_per_mbs = tokens_per_mbs_per_gpu * parallel_state.get_context_parallel_world_size()
        
        mbs_bins = [[0, []]]

        response_lengths = batch['response_lengths']
        if torch.is_tensor(response_lengths):
            response_lengths = response_lengths.tolist()

        for idx, resp_len in enumerate(response_lengths):
            mbs_bins = sorted(mbs_bins, key=lambda x: x[0])
            if mbs_bins[0][0] != 0 and (mbs_bins[0][0] + resp_len > total_tokens_per_mbs):
                mbs_bins.insert(0, [0, []])
            
            mbs_bins[0][0] += resp_len
            mbs_bins[0][1].append(idx)

        # split and pack the batch into microbatches
        microbatches = []
        for total_tokens, indices in mbs_bins:
            mbs = {}
            # split the batch into an microbatch
            for k, v in batch.items():
                if torch.is_tensor(v):
                    indices = torch.tensor(indices, dtype=torch.int, device=torch.cuda.current_device())
                    mbs[k] = torch.index_select(v, dim=0, index=indices).clone()
                elif v is None:
                    mbs[k] = None
                else:
                    mbs[k] = [v[i] for i in indices]

            # pack the response tokens along the sequence dim
            # remove the extra padding
            max_resp_len = max(mbs["response_lengths"])
            unpacked_response_tokens = mbs["response_tokens"][...,:max_resp_len].clone()
            mbs['unpacked_response_tokens'] = unpacked_response_tokens

            packed_responses = torch.cat([seq[:length] for seq, length in zip(unpacked_response_tokens, batch["response_lengths"])])
            packed_responses = torch.unsqueeze(packed_responses, dim=0)
            mbs['response_tokens'] = packed_responses

            #create packed seq params
            cu_seqlens = [0]
            for resp_length in response_lengths:
                cu_seqlens.append(cu_seqlens[-1] + resp_length)

            mbs['cu_seqlens'] = torch.tensor(cu_seqlens, dtype=torch.int, device=torch.cuda.current_device())
            mbs['cu_seqlens_argmin'] = torch.tensor(len(response_lengths)+1, dtype=torch.int)
            mbs['max_seqlen'] = torch.tensor(max(response_lengths), dtype=torch.int)

            microbatches.append(mbs)

        num_microbatches = len(microbatches)
        microbatches = itertools.chain(microbatches)
        # divide the thd batch along sequence dim for CP #TODO
        if parallel_state.get_context_parallel_world_size() > 1:
            pass
        
        return microbatches, num_microbatches
    

    # inference calls
    def get_logprob_output_only_func(self, inference_only=True):
        def log_prob_output_only_func(dataloader_iter, model):

            batch = next(dataloader_iter)
            if 'cu_seqlens' in batch:
                packed_seq_params = get_packed_seq_params(batch)

            output_tensor = model(
                input_ids=batch['response_tokens'], 
                position_ids=None, 
                attention_mask=None, 
                packed_seq_params=packed_seq_params,
                labels=None,
            )

            #unpack the outputs
            if 'cu_seqlens' in batch:
                output_tensor = torch.split(output_tensor.squeeze(), tuple(batch["response_lengths"]), dim=0)
                output_tensor = torch.nn.utils.rnn.pad_sequence(output_tensor, batch_first=True)
                response_tokens = batch['unpacked_response_tokens']
            else:
                response_tokens = batch['response_tokens']

            def id_func(output_tensor, non_loss_data=True):
                logprobs = from_parallel_logits_to_logprobs(
                    vocab_parallel_logits=output_tensor,
                    target=response_tokens,
                    inference_only=inference_only,
                    higher_stability=True,
                )
                return logprobs

            return output_tensor, id_func

        return log_prob_output_only_func

    @torch.no_grad()
    def get_inference_log_probs(self, response_tokens, response_lengths=None, forward_micro_batch_size=None):
        if forward_micro_batch_size is None:
            forward_micro_batch_size = self.forward_micro_batch_size

        set_sync_funcs(self, forward_only=True)

        attention_mask, _, position_ids = self.get_ltor_masks_and_position_ids(response_tokens)

        _batch = {
            "response_tokens" : response_tokens,
            "response_lengths" : response_lengths, 
        }

        #divide batch into microbatches
        self.sequence_packing = True
        if self.sequence_packing:
            assert response_lengths is not None

            _batch.update(
                    {
                        "attention_mask" : None,
                        "position_ids" : None,
                    }
                )

            batch_iter, num_microbatches = self.pack_and_split_batch(
                _batch, 
                tokens_per_mbs_per_gpu=4096
            ) 

        else:
            mbs, seq_length = response_tokens.size()
            num_microbatches = divide(mbs, forward_micro_batch_size)

            _batch.update(
                {
                    "attention_mask" : attention_mask,
                    "position_ids" : position_ids,
                }
            )
            batch_iter = get_iterator_k_split(_batch, num_microbatches)

        fwd_bwd_function = get_forward_backward_func()
        logprobs_list = fwd_bwd_function(
            forward_step_func=self.get_logprob_output_only_func(inference_only=True),
            data_iterator=self._make_data_iterator_list(batch_iter),
            model=self.model,
            num_microbatches=num_microbatches,
            forward_only=True,
            seq_length=None, # unused
            micro_batch_size=None, # unused
            collect_non_loss_data=True,
        )
        logprobs = torch.cat(logprobs_list) if len(logprobs_list) > 0 else None

        # Broadcast it from last PP stage to everything else.
        logprobs = broadcast_2d_tensor_within_pp(logprobs)

        return logprobs

    def prepare_for_inference(self):
        """normally we would configure the micro batch calculator here
            but the nemo generation already does the configuration"""
        self._reset_activation_checkpointing_args()
        self._reset_sequence_parallelism_args()
        set_eval(self)
        self.offload_adam_states()

        if self.use_trtllm_generation:
            # TODO this might be optimized to avoid calling `refit()` twice in a row after a validation step
            self.trtllm_generate.refit(self.model)
            clear_memory()

    @torch.no_grad()
    def infer(self, inference_batch):
        prompt_tokens = inference_batch["problem"].cuda(non_blocking=True)
        prompt_lengths = inference_batch["length"].cuda(non_blocking=True)
        ground_truths = inference_batch["ground_truth"]  # string list
        inputs = (prompt_tokens, prompt_lengths)

        strategy = TrackLengthGPTModelTextGenerationStrategy(
            model=self, context_lengths=prompt_lengths, max_length=self._length_params["max_length"]
        )

        if self.use_trtllm_generation:
            actor_output = self.trtllm_generate.generate(inputs)
            response_tokens = actor_output["response_tokens"]
            response_lengths = actor_output["response_lengths"]
        else:
            actor_output = self.generate(
                inputs=inputs,
                length_params=self._length_params,
                sampling_params=self._sampling_params,
                strategy=strategy,
            )
            response_tokens = torch.cuda.LongTensor(actor_output["token_ids"]) if actor_output else None
            response_tokens = broadcast_2d_tensor_within_pp(response_tokens, dtype=torch.long)
            response_lengths = strategy.get_lengths()

            max_response_length = response_lengths.max().item()

            # Sanity check to validate response length.
            if max_response_length != response_tokens.size(1):
                # This may actually happen because NeMo does not always stop generation after `max_length` in batch mode
                # => `response_tokens` may contain up to `max_length + max_context_length` tokens.
                # TODO once NeMo fixes this issue we should be able to always raise an exception when the check above fails,
                # and remove the `if` below.
                if (
                    max_response_length >= response_tokens.size(1)
                    or response_tokens.size(1) != prompt_lengths.max().item() + self._length_params["max_length"]
                ):
                    raise AssertionError(
                        f"max response length ({max_response_length}) does not match the size of "
                        f"`response_tokens` ({response_tokens.size(1)})"
                    )

        # sometimes backends like TRT-LLM will generate invalid tokens
        # so we need to also inplace mutate the response_tokens to be within the tokenizer range
        is_valid = verify_is_valid_and_clamp_range_(
            response_tokens,
            response_lengths,
            strategy,
            self.tokenizer,
            self.cfg.reinforce.sampling_params["end_strings"],
        )

        response_sentences = [
            self.tokenizer.ids_to_text(response_tokens[i][prompt_lengths[i] : response_lengths[i]].tolist())
            for i in range(response_lengths.shape[0])
        ]

        # print(
        #     [
        #         response_tokens[i][response_lengths[i] - 2 : response_lengths[i]].tolist()
        #         for i in range(response_lengths.shape[0])
        #     ]
        # )  # print last 2 tokens from each seq
        # print(f"response_lengths {response_lengths}")
        rollout_batch = {
            "response_tokens": response_tokens,
            "response_lengths": response_lengths,
            "response_sentences": response_sentences,
            "prompt_lengths": prompt_lengths,
            "ground_truths": ground_truths,
            "is_end": is_valid,
        }

        # return in GPU, trainer needs to move to cpu

        return rollout_batch

    def get_init_policy_logprobs(self, response_tokens, response_lengths=None):
        use_peft_init_policy = self.use_peft and self.init_policy_state_dict is None

        context_mgr = (
            adapter_control(self)
            if use_peft_init_policy
            else cpu_weight_swap(self, self.init_policy_state_dict, megatron_amp_O2=self.megatron_amp_O2)
        )

        with context_mgr:
            out = self.get_inference_log_probs(response_tokens, response_lengths)
            print(f"{torch.cuda.current_device()} fflag2!")
            return out

    def finish_inference(self):
        # training will onload the adam states, no need to onload it here
        self._restore_activation_checkpointing_args()
        self._restore_sequence_parallelism_args()

        if self.use_trtllm_generation:
            self.trtllm_generate.free()

        set_train(self)

    def offload_adam_states(self):
        if self.distributed_adam_offload_manager is None:

            self.distributed_adam_offload_manager = (
                offload_distributed_adam(
                    self._optimizer.state_dict(state_dict_format=1, gather_on_root=False), force_clear_memory=True
                )
                if self.to_offload_adam_states
                else nullcontext()
            )

            # offload onto cpu
            self.distributed_adam_offload_manager.__enter__()

    def onload_adam_states(self):
        if self.distributed_adam_offload_manager is not None:
            # load back onto GPU
            self.distributed_adam_offload_manager.__exit__(None, None, None)

        self.distributed_adam_offload_manager = None

    def get_ltor_masks_and_position_ids(self, tokens):
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            data=tokens,
            eod_token=self.tokenizer.eos_id,
            reset_position_ids=self.cfg.data.get("reset_position_ids", False),
            reset_attention_mask=self.cfg.data.get("reset_attention_mask", False),
            eod_mask_loss=False,  # since we ignore the loss mask here
        )
        attention_mask = attention_mask.expand(tokens.size(0), -1, -1, -1)
        position_ids = position_ids.expand(tokens.size(0), -1)

        return attention_mask, loss_mask, position_ids
