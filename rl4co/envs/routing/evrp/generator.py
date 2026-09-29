from typing import Callable, Union

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


# From Kool et al. 2019, Hottung et al. 2022, Kim et al. 2023


class EVRPGenerator(Generator):

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
        max_length: float = 3.0,
        capacity: float = None,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.num_station = num_station
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.vehicle_capacity = vehicle_capacity
        self.max_length = max_length
        # self.batch_size = None
        # self.rand = None
        # self.num_locs = [25, 50, 75, 100]
        # self.num_stations = [6, 9, 9, 9]
        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )

        # Depot distribution
        if kwargs.get("depot_sampler", None) is not None:
            self.depot_sampler = kwargs["depot_sampler"]
        else:
            self.depot_sampler = (
                get_sampler("depot", depot_distribution, min_loc, max_loc, **kwargs)
                if depot_distribution is not None
                else None
            )

        # station distribution
        if kwargs.get("station_sampler", None) is not None:
            self.station_sampler = kwargs["station_sampler"]
        else:
            self.station_sampler_pre = (
                get_sampler(
                    "station_pre", station_distribution, min_loc, max_loc / 2, **kwargs
                )
                if station_distribution is not None
                else None
            )
            self.station_sampler_post = (
                get_sampler(
                    "station_post", station_distribution, max_loc / 2, max_loc, **kwargs
                )
                if station_distribution is not None
                else None
            )

        # Demand distribution
        if kwargs.get("demand_sampler", None) is not None:
            self.demand_sampler = kwargs["demand_sampler"]
        else:
            self.demand_sampler = get_sampler(
                "demand", demand_distribution, min_demand, max_demand, **kwargs
            )

    def _generate(self, batch_size) -> TensorDict:
        # Sample locations: depot and customers
        # if self.batch_size is None:
        #     self.batch_size = batch_size
        #     self.rand = len(self.num_locs) - 1
        # elif self.batch_size == batch_size:
        #     self.rand = (
        #         (self.rand - 1) if (self.rand - 1) >= -len(self.num_locs) else len(self.num_locs) - 1
        #     )
        #     self.num_loc = self.num_locs[self.rand]
        #     self.num_station = self.num_stations[self.rand]
        if self.depot_sampler is not None:
            depot = self.depot_sampler.sample((*batch_size, 2)).cuda()
            locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2)).cuda()
            stations = torch.zeros((*batch_size, self.num_station, 2)).cuda()
            for i in range(self.num_station):
                if i % 4 == 0:
                    stations[:, i] = torch.cat(
                        (
                            self.station_sampler_pre.sample((*batch_size, 1)).cuda(),
                            self.station_sampler_pre.sample((*batch_size, 1)).cuda(),
                        ),
                        dim=-1,
                    )
                elif i % 4 == 1:
                    stations[:, i] = torch.cat(
                        (
                            self.station_sampler_pre.sample((*batch_size, 1)).cuda(),
                            self.station_sampler_post.sample((*batch_size, 1)).cuda(),
                        ),
                        dim=-1,
                    )
                elif i % 4 == 2:
                    stations[:, i] = torch.cat(
                        (
                            self.station_sampler_post.sample((*batch_size, 1)).cuda(),
                            self.station_sampler_post.sample((*batch_size, 1)).cuda(),
                        ),
                        dim=-1,
                    )
                else:
                    stations[:, i] = torch.cat(
                        (
                            self.station_sampler_post.sample((*batch_size, 1)).cuda(),
                            self.station_sampler_pre.sample((*batch_size, 1)).cuda(),
                        ),
                        dim=-1,
                    )
        else:
            # if depot_sampler is None, sample the depot from the locations
            locs = self.loc_sampler.sample(
                (*batch_size, self.num_loc + self.num_station + 1, 2)
            )
            depot = locs[..., 0, :]
            locs = locs[..., self.num_station + 1 :, :]
            stations = locs[..., 1 : self.num_station + 1, :]

        # Sample demands
        demand = self.demand_sampler.sample((*batch_size, self.num_loc)).cuda()

        return TensorDict(
            {"locs": locs, "depot": depot, "demand": demand, "stations": stations},
            batch_size=batch_size,
        )
