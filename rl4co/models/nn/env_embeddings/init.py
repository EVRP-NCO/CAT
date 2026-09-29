import torch
import torch.nn as nn


def env_init_embedding(env_name: str, config: dict) -> nn.Module:
    """Return the initial embedding used by the EVRPTW model."""
    if env_name != "evrptw":
        raise ValueError(f"This CAT snapshot supports only EVRPTW, not {env_name!r}")
    return EVRPTWInitEmbedding(**config)

class EVRPTWInitEmbedding(nn.Module):
    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 6):
        super(EVRPTWInitEmbedding, self).__init__()
        node_dim = node_dim  # 6: x, y, demand, time_windows(2), durations
        # Complete embeddings (Q)
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(5, embed_dim, linear_bias)  # depot embedding
        self.init_embed_station = nn.Linear(5, embed_dim, linear_bias)
        
        # Constraint embeddings (K,V) - for demand, time windows, durations
        self.init_embed_constraints = nn.Linear(4, embed_dim, linear_bias)  # demand, time_windows(2), durations
        # use same 4-dim embedding for depot and station constraints
        self.init_embed_depot_constraints = nn.Linear(2, embed_dim, linear_bias)
        self.init_embed_station_constraints = nn.Linear(3, embed_dim, linear_bias)

    def forward(self, td):
        station_size = td["locs"].size(-2) - 1 - td["demand"].size(-1)
        depot, stations, cities = (
            td["locs"][:, :1, :],
            td["locs"][:, 1 : 1 + station_size, :],
            td["locs"][:, 1 + station_size :, :],
        )

        # Complete embeddings (Q)
        depot_emb = self.init_embed_depot(
            torch.cat(
                (
                    depot,
                    td["time_windows"][:, 0:1, :],
                    td["durations"][:, 0:1, None],
                ),
                -1,
            )
        )
        station_emb = self.init_embed_station(
            torch.cat(
                (
                    stations,
                    td["time_windows"][:, 1 : 1 + station_size, :],
                    td["durations"][:, 1 : 1 + station_size, None],
                ),
                -1,
            )
        )
        cities_emb = self.init_embed(
            torch.cat(
                (
                    cities,
                    td["demand"][..., None],
                    td["time_windows"][:, 1 + station_size :],
                    td["durations"][:, 1 + station_size :, None],
                ),
                -1,
            )
        )
        complete_embedding = torch.cat((depot_emb, station_emb, cities_emb), -2)

        # Constraint embeddings (K,V)
        # depot constraint embedding: zero demand, time window, duration
        depot_const_emb = self.init_embed_depot_constraints(
            torch.cat(
                (
                    #torch.zeros_like(td["durations"][:, :1, None]),  # (B, 1, 1)
                    td["time_windows"][:, :1, :],                     # (B, 1, 2)
                    #td["durations"][:, :1, None],                     # (B, 1, 1)
                ),
                -1,
            )
        )  # (B, 1, 4)

        # station constraint embedding: zero demand, time window, duration
        station_const_emb = self.init_embed_station_constraints(
            torch.cat(
                (
                    #torch.zeros_like(td["durations"][:, 1 : 1 + station_size, None]),  # (B, station_size, 1)
                    td["time_windows"][:, 1 : 1 + station_size, :],                      # (B, station_size, 2)
                    td["durations"][:, 1 : 1 + station_size, None],                      # (B, station_size, 1)
                ),
                -1,
            )
        )  # (B, station_size, 4)
        
        cities_constraints = torch.cat(
            (
                td["demand"][..., None],
                td["time_windows"][:, 1 + station_size :],
                td["durations"][:, 1 + station_size :, None],
            ),
            -1,
        )
        cities_const_emb = self.init_embed_constraints(cities_constraints)
        
        constraint_embedding = torch.cat((depot_const_emb, station_const_emb, cities_const_emb), -2)

        return complete_embedding, constraint_embedding


