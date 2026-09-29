from typing import Callable, Union, List

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.routing.evrp.generator import EVRPGenerator
from rl4co.utils.ops import get_distance


class EVRPTWGenerator(EVRPGenerator):

    def __init__(
        self,
        num_loc: int = 20,
        num_station: int = 4,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[int, float, str, type, Callable] = Uniform,
        depot_distribution: Union[int, float, str, type, Callable] = Uniform,
        station_distribution: Union[int, float, str, type, Callable] = Uniform,
        min_demand: float = 0,
        max_demand: float = 1,
        demand_distribution: Union[int, float, type, Callable] = Uniform,
        vehicle_capacity: float = 1.0,
        max_time: float =9,
        charge_time: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            num_loc=num_loc,
            num_station=num_station,
            min_loc=min_loc,
            max_loc=max_loc,
            loc_distribution=loc_distribution,
            depot_distribution=depot_distribution,
            station_distribution=station_distribution,
            min_demand=min_demand,
            max_demand=max_demand,
            demand_distribution=demand_distribution,
            vehicle_capacity=vehicle_capacity,
            **kwargs,
        )
        self.max_time = max_time
        self.charge_time = charge_time
        # self.batch_size = None
        # self.rand = None
        # self.num_locs = [5, 5, 5, 10, 10, 10, 15, 15, 15, 15, 15, 15]
        # self.num_stations = [4, 3, 2, 5, 4, 3, 8, 7, 6, 5, 4, 3]
        # self.factor = 0
        # self.count = 1

    def _generate(self, batch_size) -> TensorDict:
        # if self.batch_size is None:
        #     self.batch_size = batch_size
        # elif self.batch_size == batch_size:
        #     self.count = self.count + 1
        #     if self.count > 80:
        #         self.rand = torch.randint(
        #             high=len(self.num_locs),
        #             size=(1,),
        #         )[0].item()
        #         self.num_loc = self.num_locs[self.rand]
        #         self.num_station = self.num_stations[self.rand]
        #         self.factor=1
        td = super()._generate(batch_size)
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        durations_custom = (
            torch.rand(
                *batch_size,
                self.num_loc,
                dtype=torch.float32,
            )
            * 0.16
        ).cuda()
        durations_other = torch.full(
            (*batch_size, self.num_station + 1),
            self.charge_time,
            dtype=torch.float32,
        ).cuda()
        durations = torch.cat((durations_other, durations_custom), dim=-1)
        durations[:, 0] = 0
        temp = torch.cat((td["stations"], td["locs"]), dim=-2)
        dist = get_distance(td["depot"], temp.transpose(0, 1)).transpose(0, 1)
        dist = torch.cat((torch.zeros(*batch_size, 1).cuda(), dist), dim=1)
        upper_bound = self.max_time - dist - durations

        # 3. create random values between 0 and 1
        ts_1 = torch.rand(*batch_size, self.num_loc + 1 + self.num_station).cuda()
        ts_2 = torch.rand(*batch_size, self.num_loc + 1 + self.num_station).cuda()

        # 4. scale values to lie between their respective min_time and max_time and convert to integer values
        min_ts = dist + (upper_bound - dist) * ts_1
        max_ts = dist + (upper_bound - dist) * ts_2

        # 5. set the lower value to min, the higher to max
        min_times = torch.min(min_ts, max_ts)
        max_times = torch.max(min_ts, max_ts)

        # 6. reset times for depot
        min_times[..., :, : 1 + self.num_station] = 0.0
        max_times[..., :, 0] = self.max_time
        max_times[..., :, 1 : 1 + self.num_station] = (
            self.max_time - dist[..., :, 1 : 1 + self.num_station]
        )

        # 7. ensure min_times < max_times to prevent numerical errors in attention.py
        # min_times == max_times may lead to nan values in _inner_mha()
        mask = min_times == max_times
        if torch.any(mask):
            min_tmp = min_times.clone()
            min_tmp[mask] = torch.max(
                dist[mask], min_tmp[mask] - 1
            )  # we are handling integer values, so we can simply substract 1
            min_times = min_tmp

            mask = min_times == max_times  # update mask to new min_times
            if torch.any(mask):
                max_tmp = max_times.clone()
                max_tmp[mask] = torch.min(
                    torch.floor(upper_bound[mask]),
                    torch.max(
                        torch.ceil(min_tmp[mask] + durations[mask]),
                        max_tmp[mask] + 1,
                    ),
                )
                max_times = max_tmp

        # 8. stack to tensor time_windows
        time_windows = torch.stack((min_times, max_times), dim=-1)


        assert torch.all(
            min_times < max_times
        ), "Please make sure the relation between max_loc and max_time allows for feasible solutions."
        td.update(
            {
                "durations": durations,
                "time_windows": time_windows,  # [batch_size,graph,2]
            }
        )
        return td
