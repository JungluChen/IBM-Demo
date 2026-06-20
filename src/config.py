"""
Default data, presets, and constants for the Supply Chain Optimizer.
"""

DEFAULT_PLANTS = ["A", "B"]
DEFAULT_MARKETS = ["North", "East", "West"]
DEFAULT_PLANT_CAPACITY = {"A": 600, "B": 500}
DEFAULT_PROD_COST = {"A": 5.2, "B": 4.7}
DEFAULT_DEMAND = {"North": 350, "East": 280, "West": 220}
DEFAULT_TRANSPORT_COST = {
    ("A", "North"): 1.2,
    ("A", "East"): 1.4,
    ("A", "West"): 1.6,
    ("B", "North"): 1.3,
    ("B", "East"): 1.1,
    ("B", "West"): 1.5,
}
DEFAULT_ROUTE_CAPACITY = {
    ("A", "North"): 400,
    ("A", "East"): 300,
    ("A", "West"): 300,
    ("B", "North"): 350,
    ("B", "East"): 330,
    ("B", "West"): 300,
}

PRESETS = {
    "Default": {},
    "High Demand": {k: v * 1.2 for k, v in DEFAULT_DEMAND.items()},
    "Low Capacity": {k: v * 0.8 for k, v in DEFAULT_PLANT_CAPACITY.items()},
}


def load_preset(name: str):
    """Load a preset scenario by name."""
    plants = DEFAULT_PLANTS.copy()
    markets = DEFAULT_MARKETS.copy()
    plant_capacity = DEFAULT_PLANT_CAPACITY.copy()
    prod_cost = DEFAULT_PROD_COST.copy()
    demand = DEFAULT_DEMAND.copy()
    transport_cost = dict(DEFAULT_TRANSPORT_COST)
    route_capacity = dict(DEFAULT_ROUTE_CAPACITY)

    if name == "High Demand":
        demand = {k: v * 1.2 for k, v in demand.items()}
    elif name == "Low Capacity":
        plant_capacity = {k: v * 0.8 for k, v in plant_capacity.items()}

    return plants, markets, plant_capacity, prod_cost, demand, transport_cost, route_capacity
