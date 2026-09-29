from typing import Optional
import os
import torch
import logging

from tensordict.tensordict import TensorDict

from rl4co.data.utils import (
    load_txt_to_tensordict,
    load_npz_to_tensordict,
    load_evrp_to_tensordict,
)
from rl4co.envs.routing.evrp.env import EVRPEnv
from rl4co.utils.ops import gather_by_index
from ..evrp.generator import EVRPGenerator
from .generator import EVRPTWGenerator

from .render import render

# Initialize logger
log = logging.getLogger(__name__)
# Set logging level (if needed)
# logging.basicConfig(level=logging.INFO) # If not already configured elsewhere

# Define TIME_EPSILON for time window checks
TIME_EPSILON = 1e-6
MAX_ENERGY = 3.0 # Module-level constant for maximum energy capacity

# Add these near the top of the class or function
class EVRPTWEnv(EVRPEnv):

    name = "evrptw"
    # MAX_ENERGY = 3.0 removed, will use module-level constant

    def __init__(
        self,
        generator: EVRPTWGenerator = None,
        generator_params: dict = {},
        use_pi_mask: bool = False, # Whether to use PI mask
        pip_step: int = 1,
        use_dynamic_charging: bool = True, # Whether to use dynamic charging time
        # MAX_ENERGY = 3.0 ,     # PI mask lookahead depth (only 0 or 1 supported now) # User's comment, can be kept or removed
        #time_window_penalty_factor: float = 1.5, # Penalty factor for time window violations
        # max_energy = 2.0, # Maximum energy capacity -- Removed (using class const self.MAX_ENERGY)
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = EVRPTWGenerator(**generator_params)
        self.generator = generator
        self.use_pi_mask = use_pi_mask
        self.use_dynamic_charging = use_dynamic_charging
        # self.k_sparse = k_sparse # Removed
        # #self.MAX_ENERGY = MAX_ENERGY # This line caused NameError, should remain commented or removed
        self.pip_step = pip_step
        # self.max_energy = max_energy # Store max_energy -- Removed
        #self.time_window_penalty_factor = time_window_penalty_factor # Store the penalty factor
        # self.at_custom=None

    def calculate_dynamic_charging_time(self, td: TensorDict, current_node: torch.Tensor) -> torch.Tensor:
        """Calculate dynamic charging time based on current energy consumption (only called at stations)"""
        batch_size = td["locs"].shape[0]
        
        # Calculate current energy level and energy needed to charge
        current_energy = MAX_ENERGY - td["used_length"]
        energy_to_charge = MAX_ENERGY - current_energy
        
        # Get normalized charging rate from TensorDict
        inverse_refueling_rate = td["inverse_refueling_rate"].squeeze(-1)  # [batch_size]
        
        # Calculate dynamic charging time
        dynamic_charging_time = energy_to_charge * inverse_refueling_rate.unsqueeze(-1)
        
        return dynamic_charging_time

    def _step(self, td: TensorDict) -> TensorDict:
        batch_size = td["locs"].shape[0]
        current_node = td["action"][:, None]
        batch_index = torch.arange(td.batch_size[0])[:, None].to(device=td.device)
        num_station = td["locs"].size(-2) - td["demand"].size(-1) - 1
        # if self.at_custom is None:
        #     self.at_custom=current_node>num_station
        # self.at_custom[current_node>num_station]=True
        # update current_time
        prev_node_idx = td["current_node"].squeeze(-1)
        action_node_idx = current_node.squeeze(-1)
        distance = td["dist_matrix"][batch_index.squeeze(-1), prev_node_idx, action_node_idx].unsqueeze(-1)

        # Get duration - use dynamic charging time only when at a station
        static_duration = gather_by_index(td["durations"], td["action"]).reshape([batch_size, 1])
        
        # Check if current node is a station (not depot)
        is_station = ((current_node > 0) & (current_node <= num_station)).float()
        
        # Only calculate dynamic charging time when at a station and enabled
        if self.use_dynamic_charging and "inverse_refueling_rate" in td and is_station.any():
            # Calculate dynamic charging time only for stations
            dynamic_charging_time = self.calculate_dynamic_charging_time(td, current_node)
            # Use dynamic charging time for stations, static duration for others
            duration = static_duration * (1 - is_station) + dynamic_charging_time * is_station
        else:
            # Use static duration for all nodes
            duration = static_duration
            
        start_times = gather_by_index(td["time_windows"], td["action"])[..., 0].reshape(
            [batch_size, 1]
        )
        end_times = gather_by_index(td["time_windows"], td["action"])[..., 1].reshape(
            [batch_size, 1]
        )
        arrival_time_val = td['current_time'] + distance
        service_start_time_val = torch.max(arrival_time_val, start_times)
        departure_time_val = service_start_time_val + duration # This is what td['current_time'] will become
        
        # Condition for time window violation: arrival time is later than the end of the time window
        # arrival_time_val is [batch_size, 1], end_times is [batch_size, 1]
        time_violation_mask = (arrival_time_val > end_times)
        # Condition for depot time window violation
        time_violation_mask_depot = (arrival_time_val > end_times[:, :1])

        # Penalty should only be applied if there's a time violation AND the destination is not the depot
        # Both masks are [batch_size, 1], so direct logical AND is fine
        apply_penalty_final_mask = time_violation_mask  # Shape [batch_size, 1]

        # Calculate the penalty amount (how much time exceeded the window)
        # This is arrival_time_val - end_times. It's only added if apply_penalty_final_mask is true.
        penalty_amount_if_violated = arrival_time_val - end_times # Shape [batch_size, 1]

        # Add penalty only where apply_penalty_final_mask is True
        # Create a zero tensor and add penalties selectively
        penalties_to_add_this_step = torch.zeros_like(td["penalty_time"]) # Shape [batch_size, 1]

        # Selectively assign penalty amounts where the final mask is true.
        # td["penalty_time"] is [batch_size, 1].
        # apply_penalty_final_mask is boolean [batch_size, 1].
        # penalty_amount_if_violated is float [batch_size, 1].
        # PyTorch allows direct assignment using boolean mask of same shape.
        penalties_to_add_this_step[apply_penalty_final_mask] = penalty_amount_if_violated[apply_penalty_final_mask]
        
        # Accumulate the penalty
        td["penalty_time"] = td["penalty_time"] + penalties_to_add_this_step
        
        td["current_time"] = (current_node != 0) * departure_time_val
        # current_node is updated to the selected action
        n_loc = td["demand"].size(-1)
        demand = torch.nn.functional.pad(td["demand"], (1, 0), mode="constant", value=0)
        selected_demand = gather_by_index(
            demand,
            torch.clamp(current_node - num_station, 0, n_loc),
            squeeze=False,
        )
        used_capacity = (td["used_capacity"] + selected_demand) * (
            current_node != 0
        ).float()
        used_length = (
            (
                td["locs"][
                    torch.arange(td.batch_size[0], device=td.device)[:, None],
                    td["current_node"],
                ]
                - td["locs"][
                    torch.arange(td.batch_size[0], device=td.device)[:, None],
                    current_node,
                ]
            ).norm(p=2, dim=-1)
            + td["used_length"]
        ) * (current_node > num_station).float()
        visited = td["visited"].clone()
        visit = visited.scatter(-1, current_node, 1)
        visit[..., 1 : 1 + num_station][
            ((current_node > num_station)).expand(batch_size, num_station)#|(self.at_custom&(current_node==0))
        ] = 0
        # self.at_custom[current_node==0]=False

        td.update(
            {
                "current_node": current_node,
                "used_capacity": used_capacity,
                "visited": visit,
                "used_length": used_length,
                "current_time": td["current_time"], # Ensure current_time is updated before action mask
                "penalty_time": td["penalty_time"], # Ensure penalty_time is updated
                "penalty_energy": td["penalty_energy"], # Ensure penalty_energy is updated
            }
        )

        # Update action mask first
        action_mask = self.get_action_mask(td)
        td.set("action_mask", action_mask)

        # Calculate and set PI mask if enabled
        if self.use_pi_mask:
            pi_mask = self.get_PI_mask(td, self.pip_step)
            td.set("pi_mask", pi_mask)

        # ii = 25
        # current_energy_val = MAX_ENERGY - td["used_length"] # Calculate current energy using module-level MAX_ENERGY
        # log.info(f"Step {td['i'][ii].item()}: Prev Node={td['current_node'][ii].item()}, Action={td['action'][ii].item()}, "
        #              f"Arrival Time={arrival_time_val[ii].item():.2f} (TW: [{start_times[ii].item():.2f}, {end_times[ii].item():.2f}]), "
        #              f"Service Start={service_start_time_val[ii].item():.2f}, Departure Time={departure_time_val[ii].item():.2f}, "
        #              f"Current Energy={current_energy_val[ii].item():.2f}")

       # Update step count
        td.update({"i": td["i"] + 1})

        # Check for termination: back at depot and all customers visited
        all_customers_visited = td["visited"][..., 1 + num_station :].all(dim=-1)
        done = (current_node.squeeze(-1) == 0) & all_customers_visited
        # Add done flag to the TensorDict if not already present or update it
        # Note: The reward calculation happens in the policy/model, not here
        td.update({"done": done.unsqueeze(-1)}) # Ensure done has shape [B, 1]

        return td

    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        # Determine the target device from the input td or default to CPU if td is None
        if td is not None:
            device = td.device
        else:
            device = getattr(self.generator, 'device', torch.device("cpu"))
            # Regenerate td using the generator if td is None
            if batch_size is None:
                batch_size = self.generator.batch_size
            td_init = self.generator(batch_size=batch_size) # Generate initial data
            td = td_init.to(device) # Move generated data to device


        # Ensure tensors from input td are on the correct device
        locs_input = td["locs"].to(device)
        depot_input = td["depot"].to(device)
        stations_input = td["stations"].to(device)
        demand_input = td["demand"].to(device)
        durations_input = td["durations"].to(device)
        time_windows_input = td["time_windows"].to(device)

        # Determine actual batch size from the input tensor
        actual_batch_size = locs_input.shape[:-2] # Use shape of input tensor

        # Create reset TensorDict, ensuring all components are on the correct device
        all_locs_for_reset = torch.cat(
            (depot_input[..., None, :], stations_input, locs_input), -2
        )
        #print(all_locs_for_reset.shape)
        td_reset = TensorDict(
            {
                "locs": all_locs_for_reset,
                "demand": demand_input,
                "current_node": torch.zeros(
                    *actual_batch_size, 1, dtype=torch.long, device=device
                ),
                "current_time": torch.zeros(
                    *actual_batch_size,
                    1,
                    dtype=torch.float32,
                    device=device,
                ),
                "penalty_time": torch.zeros(
                    *actual_batch_size,
                    1,
                    dtype=torch.float32,
                    device=device,
                ),
                "penalty_energy": torch.zeros(
                    *actual_batch_size,
                    1,
                    dtype=torch.float32,
                    device=device,
                ),
                "used_length": torch.zeros((*actual_batch_size, 1), device=device),
                "used_capacity": torch.zeros((*actual_batch_size, 1), device=device),
                "visited": torch.zeros(
                    (*actual_batch_size, locs_input.shape[-2] + stations_input.size(-2) + 1),
                    dtype=torch.bool,
                    device=device,
                ).scatter_(-1, torch.zeros(*actual_batch_size, 1, dtype=torch.long, device=device), 1), # Mark depot as visited initially
                "durations": durations_input,
                "time_windows": time_windows_input,
                "i": torch.zeros(*actual_batch_size, 1, dtype=torch.int64, device=device), # Initialize step counter
            },
            batch_size=actual_batch_size,
            device=device # Set the device for the entire TensorDict
        )
        # Note: Setting device=device for TensorDict helps, but explicit .to(device) for inputs is safer.

        # Pre-calculate distance matrix
        dist_matrix = torch.cdist(all_locs_for_reset, all_locs_for_reset, p=2.0)
        td_reset.set("dist_matrix", dist_matrix)

        action_mask = self.get_action_mask(td_reset)
        td_reset.set("action_mask", action_mask)

        # Apply PI mask if enabled during reset, storing it separately
        if self.use_pi_mask:
            # Pass default None/False for sparsity args
            pi_mask = self.get_PI_mask(td_reset, self.pip_step)
            td_reset.set("pi_mask", pi_mask) # Store PI mask separately

        # Copy factor if present
        if "factor" in td.keys():
            td_reset.set(
                "factor",
                td["factor"].to(device), # Ensure factor is also on the correct device
            )

        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        """Create action masks based on instance type (EVRPTW).

        Applies capacity, time window, visited, and energy constraints.
        Includes heuristics for low capacity, high energy (for stations),
        and time window urgency.
        """
        # --- 1. Extract Data and Basic Setup ---
        locs = td["locs"]
        current_node = td["current_node"]
        demand = td["demand"]
        used_capacity = td["used_capacity"]
        used_length = td["used_length"]
        visited = td["visited"]
        current_time = td["current_time"]
        time_windows = td["time_windows"]
        durations = td["durations"]

        batch_size = td.batch_size[0]
        device = td.device
        batch_index = torch.arange(batch_size, device=device)[:, None]
        num_nodes = locs.size(-2)
        num_customers = demand.size(-1)
        num_station = num_nodes - num_customers - 1

        # --- 2. Calculate Base Constraint Masks ---

        # 2.1 Visited Constraint
        visited_customers = visited[..., num_station + 1 :]
        visited_stations = visited[..., 1 : num_station + 1]
        # Cannot visit depot if just visited and unserved nodes exist
        depot_just_visited_and_unserved = (
            (current_node == 0) & ~(visited_customers.bool().all(-1, keepdim=True))
        )
        # Cannot visit station if depot is current and all customers are visited
        depot_current_and_all_visited = (
            (current_node == 0) & (visited_customers.bool().all(-1, keepdim=True))
        )

        # 2.2 Capacity Constraint (for customers)
        # Shape: [batch_size, num_customers]
        exceeds_cap = (demand + used_capacity > 1.0).bool()

        # 2.3 Time Window Constraint (for all nodes)
        # Shape: [batch_size, num_nodes]
        current_node_idx = current_node.squeeze(-1)
        travel_times = td["dist_matrix"][batch_index.squeeze(-1), current_node_idx, :]
        arrival_times = current_time + travel_times
        service_start_times = torch.max(arrival_times, time_windows[..., 0])
        # Ensure durations has the same shape as service_start_times
        durations_expanded = durations.expand_as(service_start_times)
        service_end_times = service_start_times + durations_expanded # Potential departure time from the *next* node if action is taken
        end_times_all = time_windows[..., 1]
        # Mask if arrival time is later than the end of the time window
        time_violation_mask = (arrival_times>(end_times_all + TIME_EPSILON)).bool()
        time_violation_mask_station = (arrival_times> (end_times_all + TIME_EPSILON)).bool()
        time_violation_mask_depot = (arrival_times> (end_times_all + TIME_EPSILON)).bool()

        # Using pre-calculated dist_matrix
        dist_matrix_full = td["dist_matrix"]
        idx_current_node_expanded = current_node.squeeze(-1) # Shape [B]
        
        customer_indices_slice = slice(1 + num_station, num_nodes)
        depot_station_indices_slice = slice(0, 1 + num_station)

        # Distances from current node to all customers: [B, num_customers]
        dist_curr_to_cust = dist_matrix_full[batch_index.squeeze(-1), idx_current_node_expanded, customer_indices_slice]

        # Distances from all customers to all depot/stations: [B, num_customers, 1 + num_station]
        # Need to select carefully: dist_matrix[b, cust_idx, depot_station_idx]
        # Create indices for broadcasting
        all_customer_indices = torch.arange(1 + num_station, num_nodes, device=device) # Shape [num_customers]
        all_depot_station_indices = torch.arange(0, 1 + num_station, device=device) # Shape [1 + num_station]

        # Sub-matrix for customer rows: dist_matrix_full[:, customer_indices_slice, :] -> [B, num_customers, num_nodes]
        # Then select depot/station columns: -> [B, num_customers, 1 + num_station]
        dist_cust_to_depot_station = dist_matrix_full[:, customer_indices_slice, :][:, :, depot_station_indices_slice]


        energy_to_customer_then_refuel = dist_curr_to_cust.unsqueeze(-1) + dist_cust_to_depot_station

        # Check if trip: current -> customer -> depot/station is feasible
        # Shape: [batch_size, num_customers]
        energy_violation_customer = (energy_to_customer_then_refuel + used_length[:, :, None]  > MAX_ENERGY).all(-1).bool()

        # Check energy required to reach depot or stations directly
        # Shape: [batch_size, 1 + num_station]
        # energy_to_depot_station = (locs[batch_index, current_node] - locs[:, : 1 + num_station]).norm(p=2, dim=-1) + used_length
        dist_curr_to_depot_station_direct = dist_matrix_full[batch_index.squeeze(-1), idx_current_node_expanded, depot_station_indices_slice]
        energy_to_depot_station = dist_curr_to_depot_station_direct + used_length

        # Shape: [batch_size, num_station]
        energy_violation_station = (energy_to_depot_station[:, 1:] > MAX_ENERGY).bool()
        # Shape: [batch_size, 1]
        energy_violation_depot = (energy_to_depot_station[:, :1] > MAX_ENERGY).bool()
        high_energy_threshold = 0.3*MAX_ENERGY #deepaco需要越高越好0.7-0.9
        current_energy = MAX_ENERGY - used_length
        depot_reachable_energy = ~energy_violation_depot 
        is_high_energy_and_depot_reachable = ((current_energy > high_energy_threshold) & depot_reachable_energy).bool()
        # 4.1 Mask for Customers (loc)
        mask_loc = (
            visited_customers.bool()             # Already visited
            | exceeds_cap                        # Exceeds capacity
            | time_violation_mask[:, 1+num_station:] # Violates time window
            | energy_violation_customer          # Energy constraint
        )
        # 4.2 Mask for Stations
        mask_station = (
            visited_stations.bool()              # Already visited
            |depot_current_and_all_visited.expand(-1, num_station) # Cannot visit station from depot if all served
            #| time_violation_mask_station[:, 1 : num_station + 1] # Time window constraint (optional, currently commented)
            | energy_violation_station           # Violates energy constraint
            | is_high_energy_and_depot_reachable.expand(-1, num_station) # Mask if energy high and depot reachable
        )
        # 4.3 Mask for Depot
        mask_depot = (
            depot_just_visited_and_unserved.bool() # Cannot revisit depot if unserved customers exist
            #| time_violation_mask_depot[:, 0:1]       # Time window constraint (optional, currently commented)
            | energy_violation_depot           # Violates energy constraint
        )

        # --- 5. Final Combination ---
        # Combine all masks: True means action is masked (invalid)
        final_mask_bool = torch.cat((mask_depot, mask_station, mask_loc), -1)
        # Calculate the allowed mask: True means action is allowed
        final_mask_allowed = ~final_mask_bool

        return final_mask_allowed.bool()

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        total_penalty = td["penalty_time"].squeeze(-1) 
        td.set("path",total_penalty)

        return super()._get_reward(td, actions)-total_penalty

    @staticmethod
    def load_data(fpath, batch_size=[]):
        """Dataset loading from file
        Normalize demand by capacity to be in [0, 1]
        """
        if os.path.splitext(fpath)[1] == ".txt":
            td_load = load_txt_to_tensordict(fpath)
        elif os.path.splitext(fpath)[1] == ".npz":
            td_load = load_npz_to_tensordict(fpath)
        elif os.path.splitext(fpath)[1] == ".evrp":
            td_load = load_evrp_to_tensordict(fpath)
        return td_load

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        return render(td, actions, ax)

    @staticmethod


    # Assume these constants are defined elsewhere and accessible
    # MAX_ENERGY = 1.0 
    # TIME_EPSILON = 1e-6
    # log = logging.getLogger(__name__)

    def get_PI_mask(td: 'TensorDict', pip_step: int) -> torch.Tensor:
        """
        计算预测改进(PI)掩码 (支持 pip_step 0, 1)

        通过前瞻一步，模拟并检查所有潜在的“下下一步”动作，来判断当前哪些动作是"有前途的"。
        这个函数假设输入的td["action_mask"]已经是即时可行的。

        Args:
            td: 当前环境状态张量字典 (t=0).
            pip_step: 前瞻深度 (0 or 1).

        Returns:
            torch.Tensor: PI掩码 (True表示允许, False表示禁止).
        """
        # --- 0. Initial Setup & Base Cases ---
        action_mask_t0 = td.get("action_mask")
        if pip_step == 0:
            return action_mask_t0
        
        if pip_step > 1:
            # log.warning(f"This get_PI_mask version only supports pip_step=1. Running with pip_step=1.")
            pass

        batch_size, num_nodes, _ = td["locs"].shape
        num_customers = td["demand"].size(-1)
        num_station = num_nodes - num_customers - 1
        device = td.device
        batch_idx_range = torch.arange(batch_size, device=device)

        # --- Tensors from current state (t=0) ---
        locs_t0 = td["locs"]
        demand_t0 = td["demand"]
        time_windows_t0 = td["time_windows"]
        durations_t0 = td["durations"]
        current_node_t0 = td["current_node"]
        current_time_t0 = td["current_time"]
        used_capacity_t0 = td["used_capacity"]
        used_length_t0 = td["used_length"]
        visited_t0 = td["visited"]
        dist_matrix_t0 = td["dist_matrix"]

        demand_padded_t0 = torch.nn.functional.pad(demand_t0, (1, 0), mode="constant", value=0)
        
        # --- 1. Identify Initial Candidate Actions 'j' ---
        candidate_actions_j = action_mask_t0.nonzero(as_tuple=False)
        if candidate_actions_j.numel() == 0:
            return action_mask_t0

        num_candidates = candidate_actions_j.shape[0]
        batch_indices_j = candidate_actions_j[:, 0]
        node_indices_j = candidate_actions_j[:, 1]

        # --- 2. Calculate the "Consequence State" at t=1 after taking action 'j' ---
        dist_to_j = dist_matrix_t0[batch_indices_j, current_node_t0[batch_indices_j].squeeze(-1), node_indices_j].unsqueeze(-1)
        
        arrival_time_at_j = current_time_t0[batch_indices_j] + dist_to_j
        start_time_j = time_windows_t0[batch_indices_j, node_indices_j, 0:1]
        is_depot_j = (node_indices_j == 0).unsqueeze(-1)
        departure_time_from_j = torch.max(arrival_time_at_j, start_time_j) + durations_t0[batch_indices_j, node_indices_j].unsqueeze(-1)
        current_time_t1 = torch.where(is_depot_j, arrival_time_at_j, departure_time_from_j)

        demand_lookup_indices_j = torch.clamp(node_indices_j - num_station, 0, num_customers)
        selected_demand_j = demand_padded_t0[batch_indices_j, demand_lookup_indices_j].unsqueeze(-1)
        used_capacity_t1 = (used_capacity_t0[batch_indices_j] + selected_demand_j) * (~is_depot_j).float()

        is_station_j = ((node_indices_j > 0) & (node_indices_j <= num_station)).unsqueeze(-1)
        used_length_t1 = used_length_t0[batch_indices_j] + dist_to_j
        used_length_t1 = used_length_t1 * (~is_station_j).float()
        
        visited_t1 = visited_t0[batch_indices_j].clone().scatter_(-1, node_indices_j.unsqueeze(-1), 1)

        # --- 3. TRUE FORWARD SIMULATION: Check all next actions 'k' from each consequence state ---
        dist_j_to_k = dist_matrix_t0[batch_indices_j, node_indices_j, :].unsqueeze(-1)
        
        arrival_time_at_k = current_time_t1.unsqueeze(1) + dist_j_to_k
        end_times_k = time_windows_t0[batch_indices_j, :, 1:2]
        time_violated_t2 = (arrival_time_at_k > end_times_k + TIME_EPSILON).squeeze(-1)

        # --- 3. TRUE FORWARD SIMULATION: Check all next actions 'k' from each consequence state ---
        # ... (前面的代码) ...
        # Check Capacity constraint for trip j -> k
        node_indices_k = torch.arange(num_nodes, device=device).expand(num_candidates, -1)
        demand_lookup_indices_k = torch.clamp(node_indices_k - num_station, 0, num_customers)
        # --- FIX START ---
        # Expand batch_indices_j to match the shape of demand_lookup_indices_k
        expanded_batch_indices_j = batch_indices_j.unsqueeze(1).expand(-1, num_nodes)
        selected_demand_k = demand_padded_t0[expanded_batch_indices_j, demand_lookup_indices_k].unsqueeze(-1)
        # --- FIX END ---

        is_depot_k = (node_indices_k == 0).unsqueeze(-1)
        used_capacity_t2 = (used_capacity_t1.unsqueeze(1) + selected_demand_k) * (~is_depot_k).float()
        capacity_violated_t2 = (used_capacity_t2 > 1.0).squeeze(-1)

        used_length_t2 = used_length_t1.unsqueeze(1) + dist_j_to_k
        energy_violated_t2 = (used_length_t2 > MAX_ENERGY).squeeze(-1)
        
        # --- FIX ---
        # 'visited_t1' and 'node_indices_k' both have shape [num_candidates, num_nodes].
        # We use 'node_indices_k' to gather the visited status for each potential next step 'k'.
        # The 'index' tensor for gather must have the same number of dimensions as the 'input' tensor.
        visited_violated_t2 = visited_t1.gather(1, node_indices_k)

        step2_violated = (
            capacity_violated_t2 | time_violated_t2 | energy_violated_t2 | visited_violated_t2
        )
        
        # --- 4. Determine Promising Actions 'j' ---
        exists_valid_next_step = torch.any(~step2_violated, dim=1)

        # --- 5. Construct the Intermediate PI Mask ---
        pi_mask_step1 = torch.zeros_like(action_mask_t0, dtype=torch.bool)
        promising_indices = candidate_actions_j[exists_valid_next_step]
        if promising_indices.numel() > 0:
            pi_mask_step1[promising_indices[:, 0], promising_indices[:, 1]] = True
        intermediate_pi_mask = pi_mask_step1
        
        # --- 6. Apply Heuristics to the promising actions ---
        # Heuristics prune the set of actions that have passed the lookahead checks.
        current_node_idx_t0 = current_node_t0.squeeze(-1)
        depot_station_slice_t0 = slice(0, 1 + num_station)
        dist_curr_to_depot_station_t0 = dist_matrix_t0[batch_idx_range, current_node_idx_t0, depot_station_slice_t0]
        energy_to_depot_station_t0 = dist_curr_to_depot_station_t0 + used_length_t0
        depot_reachable_t0 = (energy_to_depot_station_t0[:, :1] <= MAX_ENERGY)

        is_low_capacity_t0 = ((1.0 - used_capacity_t0) < 0.15).bool()
        is_high_energy_t0 = (used_length_t0 < (MAX_ENERGY * 0.7)).bool()

        heuristic_mask_loc = is_low_capacity_t0.expand(-1, num_customers)
        heuristic_mask_station = (is_high_energy_t0 & depot_reachable_t0).expand(-1, num_station)
        heuristic_mask_depot = torch.zeros_like(is_low_capacity_t0, dtype=torch.bool)
        heuristic_mask_combined = torch.cat((heuristic_mask_depot, heuristic_mask_station, heuristic_mask_loc), -1)

        intermediate_pi_mask = intermediate_pi_mask & (~heuristic_mask_combined)

        # --- 7. Sparsification (k-Nearest Neighbors) ---
        k_sparse = 100
        if 0 < k_sparse < num_nodes:
            dists_from_t0 = dist_matrix_t0[batch_idx_range, current_node_idx_t0]
            dists_from_t0_clone = dists_from_t0.clone()
            dists_from_t0_clone.scatter_(1, current_node_t0, float('inf'))

            k_actual = min(k_sparse, num_nodes - 1)
            _, neighbor_indices = torch.topk(dists_from_t0_clone, k=k_actual, dim=-1, largest=False)

            neighbor_mask = torch.zeros_like(action_mask_t0, dtype=torch.bool)
            neighbor_mask.scatter_(1, neighbor_indices, True)
            neighbor_mask[:, 0] = True

            final_pi_mask = intermediate_pi_mask & neighbor_mask
        else:
            final_pi_mask = intermediate_pi_mask

        # --- 8. Finalization and Fallback ---
        final_pi_mask = final_pi_mask & action_mask_t0

        all_masked = ~final_pi_mask.any(dim=-1)
        if all_masked.any():
            # log.warning(f"PI mask (pip_step=1) was too strict for {all_masked.sum()} instances. Reverting to base action mask.")
            final_pi_mask[all_masked] = action_mask_t0[all_masked]

        return final_pi_mask.bool()