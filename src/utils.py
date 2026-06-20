"""
Utility functions for the Supply Chain Optimizer Streamlit app.
"""

import streamlit as st


def ui_data_editor(data, **kwargs):
    """Streamlit data_editor wrapper supporting `width` parameter mapping."""
    width = kwargs.pop("width", None)
    if width is not None:
        kwargs["use_container_width"] = str(width).lower() == "stretch"
    return st.data_editor(data, **kwargs)


def ui_dataframe(data, **kwargs):
    """Streamlit dataframe wrapper supporting `width` parameter mapping."""
    width = kwargs.pop("width", None)
    if width is not None:
        kwargs["use_container_width"] = str(width).lower() == "stretch"
    return st.dataframe(data, **kwargs)


def validate_inputs_dicts(
    plants, markets, plant_capacity, prod_cost, demand,
    transport_cost, route_capacity
):
    """Validate user input dictionaries and return a list of warning messages."""
    messages = []

    if any(not p for p in plants):
        messages.append("Plants list contains empty names.")
    if len(set(plants)) != len(plants):
        messages.append("Duplicate Plant names are not allowed.")
    if any(not m for m in markets):
        messages.append("Markets list contains empty names.")
    if len(set(markets)) != len(markets):
        messages.append("Duplicate Market names are not allowed.")

    for p in plants:
        if float(plant_capacity.get(p, 0)) < 0:
            messages.append(f"Capacity for plant {p} must be non-negative.")
        if float(prod_cost.get(p, 0)) < 0:
            messages.append(f"ProdCost for plant {p} must be non-negative.")

    for m in markets:
        if float(demand.get(m, 0)) < 0:
            messages.append(f"Demand for market {m} must be non-negative.")

    for p in plants:
        for m in markets:
            if float(transport_cost.get(p, {}).get(m, 0)) < 0:
                messages.append(f"Transport Cost for {p}→{m} must be non-negative.")
            if float(route_capacity.get(p, {}).get(m, 0)) < 0:
                messages.append(f"Route Capacity for {p}→{m} must be non-negative.")

    return messages


def parse_lines(text: str):
    """Split multiline text into a list of stripped, non-empty strings."""
    return [line.strip() for line in text.splitlines() if line.strip()]
