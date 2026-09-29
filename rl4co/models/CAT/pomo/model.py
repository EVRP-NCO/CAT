from typing import Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from rl4co.data.transforms import StateAugmentation
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.models.CAT.am import AttentionModelPolicy
from rl4co.utils.ops import gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)
# torch.set_printoptions(profile="full")

class POMO(REINFORCE):
    """POMO Model for neural combinatorial optimization based on REINFORCE
    Based on Kwon et al. (2020) http://arxiv.org/abs/2010.16011.

    Note:
        If no policy kwargs is passed, we use the Attention Model policy with the following arguments:
        Differently to the base class:
        - `num_encoder_layers=6` (instead of 3)
        - `normalization="instance"` (instead of "batch")
        - `use_graph_context=False` (instead of True)
        The latter is due to the fact that the paper does not use the graph context in the policy, which seems to be
        helpful in overfitting to the training graph size.

    Args:
        env: TorchRL Environment
        policy: Policy to use for the algorithm
        policy_kwargs: Keyword arguments for policy
        baseline: Baseline to use for the algorithm. Note that POMO only supports shared baseline,
            so we will throw an error if anything else is passed.
        num_augment: Number of augmentations (used only for validation and test)
        augment_fn: Function to use for augmentation, defaulting to dihedral8
        first_aug_identity: Whether to include the identity augmentation in the first position
        feats: List of features to augment
        num_starts: Number of starts for multi-start. If None, use the number of available actions
        pip_decoder: Whether to use Predictive Improvement Decoder (PIP-D) 
        decision_boundary: Decision boundary for PI mask prediction
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module = None,
        policy_kwargs={},
        baseline: str = "shared",
        num_augment: int = 8,
        augment_fn: Union[str, callable] = "dihedral8",
        first_aug_identity: bool = True,
        feats: list = None,
        num_starts: int = None,
        pip_decoder: bool = False,
        decision_boundary: float = 0.5,
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)

        if policy is None:
            policy_kwargs_with_defaults = {
                "num_encoder_layers": 6,
                "normalization": "instance",
                "use_graph_context": False,
            }
            policy_kwargs_with_defaults.update(policy_kwargs)
            
            # Add PIP decoder parameters to policy
            if pip_decoder:
                policy_kwargs_with_defaults["pip_decoder"] = True
                policy_kwargs_with_defaults["decision_boundary"] = decision_boundary
            
            policy = AttentionModelPolicy(
                env_name=env.name, **policy_kwargs_with_defaults
            )

        assert baseline == "shared", "POMO only supports shared baseline"

        # Initialize with the shared baseline
        super(POMO, self).__init__(env, policy, baseline, **kwargs)

        self.num_starts = num_starts
        self.num_augment = num_augment
        if self.num_augment > 1:
            self.augment = StateAugmentation(
                num_augment=self.num_augment,
                augment_fn=augment_fn,
                first_aug_identity=first_aug_identity,
                feats=feats,
            )
        else:
            self.augment = None

        # Add `_multistart` to decode type for train, val and test in policy
        for phase in ["train", "val", "test"]:
            self.set_decode_type_multistart(phase)
        
        # PIP decoder settings
        self.pip_decoder = pip_decoder
        self.decision_boundary = decision_boundary
        
        # 初始化最佳 PIP-D 相关状态
        # self.best_pip_state = None
        # self.current_best_val_reward = -float('inf') # 或者根据实际奖励范围设定初始值
        self.pip_loss_weight = 0.3 # 辅助损失的权重，可以设为超参数

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = self.env.reset(batch)
        n_aug, n_start = self.num_augment, self.num_starts
        n_start = self.env.get_num_starts(td) if n_start is None else n_start

        # During training, we do not augment the data
        if phase == "train":
            n_aug = 0
            # 训练时使用当前的 PIP-D 参数
            if self.pip_decoder and hasattr(self.policy.decoder, 'pointer_pip'):
                pass # 默认使用模型当前的参数
        elif phase == "val" or phase == "test":
            #  # 验证和测试时，如果有保存的最佳参数，则加载最佳参数
            #  if self.pip_decoder and self.best_pip_state is not None and hasattr(self.policy.decoder, 'pointer_pip'):
            #      log.info("加载最佳 PIP-D 参数进行评估...")
            #      self.policy.decoder.pointer_pip.load_state_dict(self.best_pip_state['pointer_pip'])
            #      self.policy.decoder.project_node_embeddings_pip.load_state_dict(self.best_pip_state['project_node_embeddings_pip'])

             if n_aug > 1:
                 td = self.augment(td)
        else: # 其他阶段（例如 'predict'）
             if n_aug > 1:
                 td = self.augment(td)

        # Evaluate policy
        out = self.policy(
            td, self.env, phase=phase, num_starts=n_start, return_actions=True
        )

        # 从策略输出中获取辅助损失
        pip_aux_loss = out.get("pip_aux_loss", torch.tensor(0.0, device=out["reward"].device))

        # Make sure path exists in the output dictionary
        if "path" not in out:
            # Create path from actions if it doesn't exist
            if "actions" in out:
                out["path"] = out["actions"]
            else:
                # Fallback to set path same as reward shape
                out["path"] = out["reward"].clone()
                
        # Unbatchify reward to [batch_size, num_augment, num_starts].
        reward = unbatchify(out["reward"], (n_aug, n_start))
        path = unbatchify(out["path"], (n_aug, n_start))
        
        # Training phase
        if phase == "train":
            assert n_start > 1, "num_starts must be > 1 during training"
            log_likelihood = unbatchify(out["log_likelihood"], (n_aug, n_start))
            # 将辅助损失传递给 calculate_loss
            self.calculate_loss(td, batch, out, reward, log_likelihood, aux_loss=pip_aux_loss * self.pip_loss_weight)
            max_reward, max_idxs = reward.max(dim=-1)
            out.update({"max_reward": max_reward,"max_path":gather_by_index(path,max_idxs,dim=max_idxs.dim())})
            # 训练后恢复模型的 PIP-D 参数（如果之前加载了最佳参数）
            if (phase == "val" or phase == "test") and self.pip_decoder and self.best_pip_state is not None:
                 # 这一步通常不需要，因为训练是在另一个步骤进行的，
                 # 但如果shared_step可能被用于训练后的评估，则可能需要恢复
                 pass # 通常在 on_validation_epoch_end 中处理参数恢复

        # Get multi-start (=POMO) rewards and best actions only during validation and test
        else:
            if n_start > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward,"max_path":gather_by_index(path,max_idxs,dim=max_idxs.dim())})

                if out.get("actions", None) is not None:
                    # Reshape batch to [batch_size, num_augment, num_starts, ...]
                    actions = unbatchify(out["actions"], (n_aug, n_start))
                    out.update(
                        {
                            "best_multistart_actions": gather_by_index(
                                actions, max_idxs, dim=max_idxs.dim()
                            )
                        }
                    )
                    out["actions"] = actions
                   
            # Get augmentation score only during inference
            if n_aug > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                path_=gather_by_index(path,max_idxs,dim=max_idxs.dim()) if n_start>1 else path
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward,"max_aug_path":gather_by_index(path_,max_idxs,dim=max_idxs.dim())})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})
                    zero_index = torch.nonzero(out["best_aug_actions"])[:, -1].max() + 2 # 可能出问题
                    solution= out["best_aug_actions"][:, :zero_index]
                    log.info(solution)
            else:
                log.info(actions) # actions 可能很大

        # 在验证阶段结束时检查并更新最佳 PIP-D
        # 注意：这部分逻辑更适合放在 on_validation_epoch_end 回调中
        # 这里仅作示例，实际应移至回调函数
        # if phase == "val":
        #     current_val_reward = metrics.get("val/reward", -float('inf')) # 假设log_metrics返回 'val/reward'
        #     if current_val_reward > self.current_best_val_reward:
        #         log.info(f"发现更好的 PIP-D 参数，验证奖励: {current_val_reward:.4f} (旧: {self.current_best_val_reward:.4f})")
        #         self.current_best_val_reward = current_val_reward
        #         if self.pip_decoder and hasattr(self.policy.decoder, 'pointer_pip'):
        #             self.best_pip_state = {
        #                 'pointer_pip': self.policy.decoder.pointer_pip.state_dict(),
        #                 'project_node_embeddings_pip': self.policy.decoder.project_node_embeddings_pip.state_dict()
        #             }

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        
        # 评估后恢复模型当前参数（如果评估时加载了最佳参数）
        # 注意：这也应在回调中完成，确保训练不受影响
        # if (phase == "val" or phase == "test") and self.pip_decoder and self.best_pip_state is not None:
        #      # 这里恢复可能不安全，因为模型状态可能已被其他回调更改
        #      # 最佳实践是在 on_validation_epoch_end 中处理
        #      pass

        return {"loss": out.get("loss", None), **metrics}

    # # 添加验证周期结束回调来处理最佳 PIP-D 参数
    # def on_validation_epoch_end(self):
    #     if not self.pip_decoder or not hasattr(self.policy.decoder, 'pointer_pip'):
    #         return

    #     # 从 trainer 获取平均验证奖励 (需要确保 'val/max_aug_reward' 被记录)
    #     # trainer.callback_metrics 在 Lightning 2.0 中可能需要调整访问方式
    #     # 或者直接从 self.logged_metrics 获取
    #     metric_to_track = "val/max_aug_reward"
    #     current_val_metric = self.trainer.callback_metrics.get(metric_to_track, None)

    #     if current_val_metric is None:
    #          # 尝试从 logged_metrics 获取，移除 /step 后缀
    #          metrics_no_step = {k.replace('/step',''): v for k,v in self.trainer.logged_metrics.items()}
    #          current_val_metric = metrics_no_step.get(metric_to_track, -float('inf'))

    #     # POMO/AM 通常目标是最小化 cost/reward (例如 TSP 长度), 所以我们检查是否 *小于* 当前最佳
    #     # 但如果环境返回的是负奖励 (例如 -cost), 则需要最大化 (检查是否 *大于*)
    #     # 假设奖励是需要最大化的 (值越大越好)
    #     if current_val_metric > self.current_best_val_reward:
    #         self.current_best_val_reward = current_val_metric
    #         log.info(f"更新最佳 PIP-D 参数，基于 {metric_to_track}: {current_val_metric:.4f}")
    #         # 使用 deepcopy 确保状态字典被完全复制
    #         self.best_pip_state = {
    #             'pointer_pip': copy.deepcopy(self.policy.decoder.pointer_pip.state_dict()),
    #             'project_node_embeddings_pip': copy.deepcopy(self.policy.decoder.project_node_embeddings_pip.state_dict())
    #         }
    #     else:
    #         log.info(f"当前 PIP-D 未超越最佳记录 ({current_val_metric:.4f} based on {metric_to_track} vs best {self.current_best_val_reward:.4f})")

    #     # 调用父类的回调（如果存在）
    #     # Note: Check if super().on_validation_epoch_end() exists or is needed
    #     # super().on_validation_epoch_end()

    # # 添加保存和加载 checkpoint 的钩子以包含/恢复 best_pip_state
    # def on_save_checkpoint(self, checkpoint):
    #     if self.pip_decoder:
    #         checkpoint['best_pip_state'] = self.best_pip_state
    #         checkpoint['current_best_val_reward'] = self.current_best_val_reward # 这个值现在代表最佳 val/max_aug_reward

    # def on_load_checkpoint(self, checkpoint):
    #     if self.pip_decoder:
    #         self.best_pip_state = checkpoint.get('best_pip_state', None)
    #         self.current_best_val_reward = checkpoint.get('current_best_val_reward', -float('inf'))
    #         if self.best_pip_state:
    #              log.info(f"从 Checkpoint 加载了最佳 PIP-D 状态 (基于之前记录的最佳 val/max_aug_reward: {self.current_best_val_reward:.4f})")
