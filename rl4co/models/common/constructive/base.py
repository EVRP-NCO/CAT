import abc
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union
import math
import torch.nn as nn
import torch
from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)
from rl4co.utils.ops import calculate_entropy
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)
#torch.set_printoptions(profile="full")

class ConstructiveEncoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for the encoder of constructive models"""

    @abc.abstractmethod
    def forward(self, td: TensorDict) -> Tuple[Any, Tensor]:
        """Forward pass for the encoder

        Args:
            td: TensorDict containing the input data

        Returns:
            Tuple containing:
              - latent representation (any type)
              - initial embeddings (from feature space to embedding space)
        """
        raise NotImplementedError("Implement me in subclass!")


class ConstructiveDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Base decoder model for constructive models. The decoder is responsible for generating the logits for the action"""

    @abc.abstractmethod
    def forward(
        self, td: TensorDict, hidden: Any = None, num_starts: int = 0
    ) -> Tuple[Tensor, Tensor]:
        """Obtain logits for current action to the next ones

        Args:
            td: TensorDict containing the input data
            hidden: Hidden state from the encoder. Can be any type
            num_starts: Number of starts for multistart decoding

        Returns:
            Tuple containing the logits and the action mask
        """
        raise NotImplementedError("Implement me in subclass!")

    def pre_decoder_hook(
        self, td: TensorDict, env: RL4COEnvBase, hidden: Any = None, num_starts: int = 0
    ) -> Tuple[TensorDict, Any, RL4COEnvBase]:
        """By default, we don't need to do anything here.

        Args:
            td: TensorDict containing the input data
            hidden: Hidden state from the encoder
            env: Environment for decoding
            num_starts: Number of starts for multistart decoding

        Returns:
            Tuple containing the updated hidden state, TensorDict, and environment
        """
        return td, env, hidden


class NoEncoder(ConstructiveEncoder):
    """Default encoder decoder-only models, i.e. autoregressive models that re-encode all the state at each decoding step."""

    def forward(self, td: TensorDict) -> Tuple[Tensor, Tensor]:
        """Return Nones for the hidden state and initial embeddings"""
        return None, None


class ConstructivePolicy(nn.Module):
    """
    Base class for constructive policies. Constructive policies take as input and instance and output a solution (sequence of actions).
    "Constructive" means that a solution is created from scratch by the model.

    The structure follows roughly the following steps:
        1. Create a hidden state from the encoder
        2. Initialize decoding strategy (such as greedy, sampling, etc.)
        3. Decode the action given the hidden state and the environment state at the current step
        4. Update the environment state with the action. Repeat 3-4 until all sequences are done
        5. Obtain log likelihood, rewards etc.

    Note that an encoder is not strictly needed (see :class:`NoEncoder`).). A decoder however is always needed either in the form of a
    network or a function.

    Note:
        There are major differences between this decoding and most RL problems. The most important one is
        that reward may not defined for partial solutions, hence we have to wait for the environment to reach a terminal
        state before we can compute the reward with `env.get_reward()`.

    Warning:
        We suppose environments in the `done` state are still available for sampling. This is because in NCO we need to
        wait for all the environments to reach a terminal state before we can stop the decoding process. This is in
        contrast with the TorchRL framework (at the moment) where the `env.rollout` function automatically resets.
        You may follow tighter integration with TorchRL here: https://github.com/ai4co/rl4co/issues/72.

    Args:
        encoder: Encoder to use
        decoder: Decoder to use
        env_name: Environment name to solve (used for automatically instantiating networks)
        temperature: Temperature for the softmax during decoding
        tanh_clipping: Clipping value for the tanh activation (see Bello et al. 2016) during decoding
        mask_logits: Whether to mask the logits or not during decoding
        train_decode_type: Decoding strategy for training
        val_decode_type: Decoding strategy for validation
        test_decode_type: Decoding strategy for testing
    """

    def __init__(
        self,
        encoder: Union[ConstructiveEncoder, Callable],
        decoder: Union[ConstructiveDecoder, Callable],
        env_name: str = "tsp",
        temperature: float = 1.0,
        tanh_clipping: float = 0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kw,
    ):
        super(ConstructivePolicy, self).__init__()

        # if len(unused_kw) > 0:
        #     log.error(f"Found {len(unused_kw)} unused kwargs: {unused_kw}")

        self.env_name = env_name

        # Encoder and decoder
        if encoder is None:
            log.warning("`None` was provided as encoder. Using `NoEncoder`.")
            encoder = NoEncoder()
        self.encoder = encoder
        self.decoder = decoder

        # Decoding strategies
        self.temperature = temperature
        self.tanh_clipping = tanh_clipping
        self.mask_logits = mask_logits
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def forward(
        self,
        td: TensorDict,
        env: Optional[Union[str, RL4COEnvBase]] = None,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = False,
        return_entropy: bool = False,
        return_hidden: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        actions=None,
        max_steps=1_000_000,
        # return_tw=True,
        **decoding_kwargs,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
            phase: Phase of the algorithm (train, val, test)
            calc_reward: Whether to calculate the reward
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            return_hidden: Whether to return the hidden state
            return_init_embeds: Whether to return the initial embeddings
            return_sum_log_likelihood: Whether to return the sum of the log likelihood
            actions: Actions to use for evaluating the policy.
                If passed, use these actions instead of sampling from the policy to calculate log likelihood
            max_steps: Maximum number of decoding steps for sanity check to avoid infinite loops if envs are buggy (i.e. do not reach `done`)
            decoding_kwargs: Keyword arguments for the decoding strategy. See :class:`rl4co.utils.decoding.DecodingStrategy` for more information.

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # Encoder: get encoder output and initial embeddings from initial state
        # Enable attention collection during testing/evaluation
        if phase in ["test", "val"] and hasattr(self.encoder, 'forward'):
            try:
                # Try to get attention weights during testing
                encoder_output = self.encoder(td, return_attention=True)
                if len(encoder_output) == 3:
                    hidden, init_embeds, attention_weights = encoder_output
                    log.info(f"成功收集到注意力权重，层数: {len(attention_weights) if attention_weights else 0}")
                else:
                    hidden, init_embeds = encoder_output
            except Exception as e:
                log.warning(f"无法收集注意力权重: {e}，使用标准forward")
                hidden, init_embeds = self.encoder(td)
        else:
            hidden, init_embeds = self.encoder(td)
        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        # Get decode type depending on phase and whether actions are passed for evaluation
        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # Setup decoding strategy
        # we pop arguments that are not part of the decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            **decoding_kwargs,
        )

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)

        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)

        # Initialize log likelihood and entropy
        log_likelihood = torch.zeros(
            *td.batch_size, 1, device=td.device, dtype=torch.float32
        )
        entropy = torch.zeros(*td.batch_size, device=td.device, dtype=torch.float32)

        # Initialize accumulated auxiliary loss (e.g., for PIP-D)
        accumulated_aux_loss = torch.tensor(0.0, device=td.device)

        # Main decoding: loop until all sequences are done
        step = 0
        while not td["done"].all():            
            # Obtain logits from decoder based on current state and encoder hidden state
            # We pass the 'hidden' state which might be the PrecomputedCache from the hook
            decoder_output = self.decoder(td, hidden, num_starts)

            # Handle dict or tuple return from decoder
            if isinstance(decoder_output, dict):
                logits = decoder_output["logits"]
                mask = decoder_output["mask"]
                # Get auxiliary loss if present and add to accumulator
                step_aux_loss = decoder_output.get("pip_aux_loss", torch.tensor(0.0, device=logits.device))
                accumulated_aux_loss = accumulated_aux_loss + step_aux_loss
            elif isinstance(decoder_output, tuple) and len(decoder_output) == 2:
                # If decoder returns a tuple (legacy or non-PIP case), assume no aux loss
                logits, mask = decoder_output
                step_aux_loss = torch.tensor(0.0, device=logits.device) # Ensure aux loss is zero
                # Note: accumulated_aux_loss does not increase here
            else:
                raise ValueError(f"Decoder output must be a dict or a tuple of (logits, mask), got {type(decoder_output)}")

            # Decode action based on decoding strategy
            # The decode_strategy.step updates the td in place and stores logp/actions internally
            td = decode_strategy.step(logits, mask, td, action=actions[..., step] if actions is not None else None)

            # Update environment state using the action selected by the strategy
            td = env.step(td)["next"]

            # Check termination condition
            if step > max_steps:
                log.error(
                    f"Max steps {max_steps} reached. This may indicate an infinite loop. "
                    f"Check the environment step function or set a higher max_steps."
                )
                td["done"] = torch.ones_like(td["done"])

            step += 1

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)

        # Output dictionary construction
        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        outdict = {
            "path": td["path"],
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs, actions, td.get("mask", None), return_sum_log_likelihood
            ),
        }
        if phase == "test":
            # 将第一步的 actions 写入文件
            # log.info(f"Step 0: Capturing logits with shape {logits.shape}")
            # first_step_actions = actions[...,0]
            # with open("actions_output.txt", "w", encoding="utf-8") as f:
            #     f.write("--- Actions Tensor (Step 0) ---\n")
            #     f.write(str(first_step_actions.cpu().detach().numpy()))
            #     f.write("\n\n")
            print(actions)
            # action_temp = actions[td["reward"].max(-1).indices]
            # file = env.test_file.rsplit("/", maxsplit=1)[1].rsplit(".", maxsplit=1)[0]
            #reward_out = td["reward"].max()
            # path = td["path"][td["reward"].max(-1).indices]
            # algorithm = self.__module__.rsplit(".", maxsplit=2)[1]
            # load_data(action_temp, reward_out, path, file, algorithm)
        if return_actions:
            outdict["actions"] = actions
        if return_entropy:
            outdict["entropy"] = calculate_entropy(logprobs)
        if return_hidden:
            outdict["hidden"] = hidden
        if return_init_embeds:
            outdict["init_embeds"] = init_embeds

        # Aggregate results
        outdict["pip_aux_loss"] = accumulated_aux_loss / step

        return outdict

    def route_feature(self, tour, td, embeddings):

        batch_size, _, embed_dim = embeddings.size()
        if len(tour) != 0:
            batch_tour = torch.stack(tour, -1)  # [batch_size, tour_len]
            tour_con = torch.gather(
                embeddings,  # [batch_size, graph_size, embed_dim]
                1,
                (batch_tour.clone())[..., None]
                .contiguous()  # [batch_size, tour_len]
                .expand(batch_size, batch_tour.size(-1), embed_dim),
            ).view(
                batch_size, batch_tour.size(-1), embed_dim
            )  # [batch_size, tour_len, embed_dim]
            mean_tour = torch.max(tour_con, dim=1)[0]
            veh_context = torch.cat(
                (
                    td["used_capacity"],
                    td["used_length"] / 3,
                    td["current_time"] / 9,
                ),
                -1,
            )
        else:
            mean_tour = torch.zeros([batch_size, embed_dim]).float().cuda()
            veh_context = torch.cat(
                (
                    td["used_capacity"],
                    td["used_length"] / 3,
                    td["current_time"] / 9,
                ),
                -1,
            )
        veh_context1 = self.decoder.tour(veh_context)
        cat_context = torch.cat((veh_context1, mean_tour), -1).view(
            batch_size, embed_dim * 2
        )
        context = self.decoder.FF_tour(cat_context)

        return context
    
    def route_feature_clh(self, tour, td, embeddings):

        batch_size, _, embed_dim = embeddings.size()
        if len(tour) != 0:
            batch_tour = torch.stack(tour, -1)  # [batch_size, tour_len]
            tour_con = torch.gather(
                embeddings,  # [batch_size, graph_size, embed_dim]
                1,
                (batch_tour.clone())[..., None]
                .contiguous()  # [batch_size, tour_len]
                .expand(batch_size, batch_tour.size(-1), embed_dim),
            ).view(
                batch_size, batch_tour.size(-1), embed_dim
            )  # [batch_size, tour_len, embed_dim]
            mean_tour = torch.mean(tour_con, dim=1)
            veh_context = torch.cat(
                (
                    td["used_capacity"],
                    td["used_length"] / 3,
                    td["current_time"] / 9,
                ),
                -1,
            )
        else:
            mean_tour = torch.zeros([batch_size, embed_dim]).float().cuda()
            veh_context = torch.cat(
                (
                    td["used_capacity"],
                    td["used_length"] / 3,
                    td["current_time"] / 9,
                ),
                -1,
            )
        veh_context1 = self.decoder.tour(veh_context)
        cat_context = torch.cat((veh_context1, mean_tour), -1).view(
            batch_size, embed_dim * 2
        )
        context = self.decoder.FF_tour(cat_context)

        return context

    def route_feature_mul(self, tour, td, embeddings):

        batch_size, _, embed_dim = embeddings.size()
        if len(tour) != 0:
            tour_size = tour[-1].size(-1)
            batch_tour = torch.stack(tour, -1).view(
                batch_size, -1
            )  # [batch_size, tour_len]
            tour_con = torch.gather(
                embeddings,  # [batch_size, graph_size, embed_dim]
                1,
                (batch_tour.clone())[..., None]
                .contiguous()  # [batch_size, tour_len]
                .expand(batch_size, batch_tour.size(-1), embed_dim),
            ).view(
                tour_size, len(tour), embed_dim
            )  # [batch_size, tour_len, embed_dim]
            mean_tour = torch.mean(tour_con, dim=1)
            veh_context = torch.cat(
                (
                    td["used_capacity"],
                    td["used_length"] / 3,
                    td["current_time"] / 9,
                ),
                -1,
            )
        else:
            mean_tour = torch.zeros([batch_size, embed_dim]).float().cuda()
            veh_context = torch.cat(
                (
                    td["used_capacity"],
                    td["used_length"] / 3,
                    td["current_time"] / 9,
                ),
                -1,
            )
        veh_context1 = self.decoder.tour(veh_context)
        cat_context = torch.cat((veh_context1, mean_tour), -1)
        context = self.decoder.FF_tour(cat_context)

        return context.view(batch_size, -1, embed_dim)
