from typing import Optional

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.common.base import RL4COEnvBase

try:
    from .local_search import local_search
except:
    # In case some dependencies are not installed (e.g., pyvrp)
    local_search = None
from .render import render
import os
import torch
from tensordict.tensordict import TensorDict
from rl4co.utils.ops import gather_by_index, get_tour_length, get_distance
from rl4co.data.utils import load_evrp_to_tensordict, load_npz_to_tensordict
from rl4co.utils.pylogger import get_pylogger
from .generator import EVRPGenerator


log = get_pylogger(__name__)


class EVRPEnv(RL4COEnvBase):

    name = "evrp"

    def __init__(
        self,
        generator: EVRPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = EVRPGenerator(**generator_params)
        self.generator = generator
        # self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"][:, None]
        n_loc = td["demand"].size(-1)
        demand = torch.nn.functional.pad(td["demand"], (1, 0), mode="constant", value=0)
        num_station=td["locs"].size(-2)-td["demand"].size(-1)-1
        selected_demand = gather_by_index(
            demand,
            torch.clamp(current_node - num_station, 0, n_loc),
            squeeze=False,
        )
        # Increase capacity if depot is not visited, otherwise set to 0
        used_capacity = (td["used_capacity"] + selected_demand) * (
            current_node != 0
        ).float()
        used_length = (
            (
                td["locs"][torch.arange(td.batch_size[0])[:, None], td["current_node"]]
                - td["locs"][torch.arange(td.batch_size[0])[:, None], current_node]
            ).norm(p=2, dim=-1)
            + td["used_length"]
        ) * (current_node > num_station).float()
        visited = torch.nn.functional.pad(td["visited"], (1, 0), mode="constant", value=0)
        visit = visited.scatter(
            -1, torch.clamp(current_node - num_station, 0, n_loc), 1
        )[:, 1:]
        # SECTION: get done
        done = (visit.sum(-1, keepdim=True) == visit.size(-1)) & (current_node == 0)
        reward = torch.zeros_like(done)

        td.update(
            {
                "current_node": current_node,
                "used_capacity": used_capacity,
                "visited": visit,
                "reward": reward,
                "done": done,
                "used_length": used_length,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        device = td.device

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "locs": torch.cat(
                    (td["stations"], td["locs"]), -2
                ),
                "demand": td["demand"],
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=device
                ),
                "used_length": torch.zeros((*batch_size, 1), device=device),
                "used_capacity": torch.zeros((*batch_size, 1), device=device),
                "visited": torch.zeros(
                    (*batch_size, td["locs"].shape[-2]),
                    dtype=torch.uint8,
                    device=device,
                ),
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        if "factor" in td.sorted_keys:
            td_reset.set(
                "factor",
                td["factor"],
            )
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = td["demand"] + td["used_capacity"] > 3
        batch_index = torch.arange(td.batch_size[0])[:, None]
        num_station=td["locs"].size(-2)-td["demand"].size(-1)-1
        length_loc = (
            (
                td["locs"][batch_index, td["current_node"]]
                - td["locs"][:, 1 + num_station :]
            ).norm(p=2, dim=-1)[:, :, None]
            + (
                td["locs"][:, 1 + num_station :, None]
                - td["locs"][:, None, : 1 + num_station]
            ).norm(p=2, dim=-1)
            + td["used_length"][:, :, None]
        )
        exceeds_length = (length_loc + 1e-4 > 3.0).all(-1)
        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = td["visited"].to(exceeds_cap.dtype) | exceeds_cap | exceeds_length
        length_other = (
            td["locs"][batch_index, td["current_node"]]
            - td["locs"][:, : 1 + num_station]
        ).norm(p=2, dim=-1) + td["used_length"]
        exceeds_station = length_other[:, 1:] > 3.0
        mask_station = exceeds_station | (
            td["current_node"] <= num_station
        ).expand_as(exceeds_station)
        # Cannot visit the depot if just visited and still unserved nodes
        exceeds_depot = length_other[:, :1] > 3.0
        mask_depot = (
            (td["current_node"] == 0) & ~(td["visited"].bool().all(-1, keepdim=True))
        ) | exceeds_depot
        return ~torch.cat((mask_depot, mask_station, mask_loc), -1)

    def _get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        # Gather locations in order of tour (add depot since we start and end there)
        locs_ordered = torch.cat(
            [
                td["locs"][..., 0:1, :],  # depot
                gather_by_index(td["locs"], actions),  # order locations
            ],
            dim=1,
        )
        if "factor" in td.sorted_keys:
            return -get_tour_length(locs_ordered) * td["factor"][0]
        else:
            return -get_tour_length(locs_ordered)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        batch_size, graph_size = td["demand"].size()
        station_size = td["locs"].size(-2) - graph_size - 1
        sorted_pi = actions.data.sort(1)[0]
        assert (
            torch.arange(
                station_size + 1, graph_size + 1 + station_size, out=sorted_pi.data.new()
            )
            .view(1, -1)
            .expand(batch_size, graph_size)
            == sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] < station_size + 1).all(), "Invalid tour"
        demand_all = torch.cat(
            (
                -torch.ones(
                    (batch_size, 1),
                    dtype=td["demand"].dtype,
                    device=td["demand"].device,
                ),
                torch.zeros(
                    (batch_size, station_size),
                    dtype=td["demand"].dtype,
                    device=td["demand"].device,
                ),
                td["demand"],
            ),
            1,
        )
        d = demand_all.gather(1, actions)
        used_cap = torch.zeros_like(td["demand"][:, 0])
        for i in range(actions.size(1)):
            used_cap += d[:, i]
            used_cap[used_cap < 0] = 0
            assert (used_cap <= 1).all(), "Used more than capacity"
        batch_size = td["locs"].shape[0]
        locs_ordered = gather_by_index(td["locs"], actions).reshape(
            [batch_size, actions.size(-1), 2]
        )
        ordered_locs_next = torch.roll(locs_ordered, 1, dims=-2)
        distance = get_distance(ordered_locs_next, locs_ordered)
        used_len = torch.zeros_like(distance[:, 0])
        for j in range(actions.size(1)):
            used_len += distance[:, j]
            assert (used_len <= 3.0).all(), "Used more than energy"
            used_len[actions[:, j] < station_size + 1] = 0

    @staticmethod
    def load_data(fpath, batch_size=[]):
        """Dataset loading from file
        Normalize demand by capacity to be in [0, 1]
        """
        if os.path.splitext(fpath)[1] == ".evrp":
            td_load = load_evrp_to_tensordict(fpath)
        elif os.path.splitext(fpath)[1] == ".npz":
            td_load = load_npz_to_tensordict(fpath)
        return td_load


    @staticmethod
    def local_search(td: TensorDict, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        assert (
            local_search is not None
        ), "Cannot import local_search module. Check if `pyvrp` is installed."
        return local_search(td, actions, **kwargs)

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        return render(td, actions, ax)
