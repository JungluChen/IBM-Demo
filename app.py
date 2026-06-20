"""
Simple Supply Chain Optimizer — Streamlit frontend.

Run with:  streamlit run app.py
"""

import sys

import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_DEMAND,
    DEFAULT_MARKETS,
    DEFAULT_PLANT_CAPACITY,
    DEFAULT_PLANTS,
    DEFAULT_PROD_COST,
    DEFAULT_ROUTE_CAPACITY,
    DEFAULT_TRANSPORT_COST,
    load_preset,
)
from src.solver import (
    build_and_solve,
    solve_with_ortools,
    extract_shipment_results,
)
from src.utils import (
    ui_data_editor,
    ui_dataframe,
    validate_inputs_dicts,
)

st.set_page_config(
    page_title="Simple Supply Chain Optimizer",
    page_icon="\U0001f4e6",
    layout="wide",
)
st.title("Simple Supply Chain Optimizer")
st.caption(
    "Innovation XLab \u2022 IBM Decision Optimization "
    "(CPLEX Modeling for Python) \u2014 plan shipments to meet demand at minimal cost."
)
st.markdown(
    """
    Built by **Innovation XLab** for the IBM project, this app uses
    **IBM Decision Optimization CPLEX Modeling for Python (docplex)** to
    optimize shipments between plants and markets.

    **How to use:**
    - Add Plants and Markets in the tables below.
    - Enter each plant's capacity and production cost, and each market's demand.
    - Set transport costs and route capacities for each Plant \u2192 Market pair.
    - Choose a service level (%) in the sidebar and click **Solve**.
    - Review results: total cost, shipped quantities, service metrics, and the shipment plan table.
    """
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def ensure_session_defaults():
    if "plants" not in st.session_state:
        st.session_state.plants = DEFAULT_PLANTS.copy()
    if "markets" not in st.session_state:
        st.session_state.markets = DEFAULT_MARKETS.copy()
    if "plants_text" not in st.session_state:
        st.session_state.plants_text = "\n".join(st.session_state.plants)
    if "markets_text" not in st.session_state:
        st.session_state.markets_text = "\n".join(st.session_state.markets)
    if "plant_capacity" not in st.session_state:
        st.session_state.plant_capacity = DEFAULT_PLANT_CAPACITY.copy()
    if "prod_cost" not in st.session_state:
        st.session_state.prod_cost = DEFAULT_PROD_COST.copy()
    if "demand" not in st.session_state:
        st.session_state.demand = DEFAULT_DEMAND.copy()
    if "transport_cost" not in st.session_state:
        tc = {p: {} for p in st.session_state.plants}
        for p in st.session_state.plants:
            for m in st.session_state.markets:
                tc[p][m] = DEFAULT_TRANSPORT_COST.get((p, m), 0.0)
        st.session_state.transport_cost = tc
    if "route_capacity" not in st.session_state:
        rc = {p: {} for p in st.session_state.plants}
        for p in st.session_state.plants:
            for m in st.session_state.markets:
                rc[p][m] = DEFAULT_ROUTE_CAPACITY.get((p, m), 0.0)
        st.session_state.route_capacity = rc
    if "results" not in st.session_state:
        st.session_state.results = None
    if "plants_df" not in st.session_state:
        st.session_state.plants_df = pd.DataFrame(
            {"Plant": st.session_state.plants}
        )
    if "markets_df" not in st.session_state:
        st.session_state.markets_df = pd.DataFrame(
            {"Market": st.session_state.markets}
        )


def apply_preset(name: str):
    plants, markets, plant_capacity, prod_cost, demand, transport_cost, route_capacity = (
        load_preset(name)
    )
    st.session_state.plants = plants
    st.session_state.markets = markets
    st.session_state.plants_text = "\n".join(plants)
    st.session_state.markets_text = "\n".join(markets)
    st.session_state.plant_capacity = plant_capacity
    st.session_state.prod_cost = prod_cost
    st.session_state.demand = demand

    tc = {p: {} for p in plants}
    rc = {p: {} for p in plants}
    for p in plants:
        for m in markets:
            tc[p][m] = transport_cost.get((p, m), 0.0)
            rc[p][m] = route_capacity.get((p, m), 0.0)
    st.session_state.transport_cost = tc
    st.session_state.route_capacity = rc


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

ensure_session_defaults()

with st.sidebar:
    st.header("Controls")
    service_level = (
        st.slider(
            "Service level (%)",
            min_value=0,
            max_value=100,
            value=100,
            step=1,
            help="Percentage of demand to satisfy.",
        )
        / 100
    )
    run_solve = st.button("Solve", type="primary")

# ---------------------------------------------------------------------------
# Entity tables
# ---------------------------------------------------------------------------

c1, c2 = st.columns(2)
with c1:
    plants_df = ui_data_editor(
        st.session_state.plants_df,
        num_rows="dynamic",
        width="stretch",
        key="plants_editor",
        column_config={
            "Plant": st.column_config.TextColumn(
                "Plant", help="Add/modify Plant names here"
            )
        },
    )
    st.session_state.plants_df = plants_df
with c2:
    markets_df = ui_data_editor(
        st.session_state.markets_df,
        num_rows="dynamic",
        width="stretch",
        key="markets_editor",
        column_config={
            "Market": st.column_config.TextColumn(
                "Market", help="Add/modify Market names here"
            )
        },
    )
    st.session_state.markets_df = markets_df

# Sync entities from tables
new_plants = list(st.session_state.plants_df["Plant"].dropna().astype(str))
new_markets = list(st.session_state.markets_df["Market"].dropna().astype(str))
st.session_state.plants = new_plants
st.session_state.markets = new_markets

# Align parameter dicts to current entities
for p in list(st.session_state.plant_capacity.keys()):
    if p not in new_plants:
        st.session_state.plant_capacity.pop(p)
        st.session_state.prod_cost.pop(p, None)
        st.session_state.transport_cost.pop(p, None)
        st.session_state.route_capacity.pop(p, None)
for m in list(st.session_state.demand.keys()):
    if m not in new_markets:
        st.session_state.demand.pop(m)

for p in new_plants:
    st.session_state.plant_capacity.setdefault(
        p, float(DEFAULT_PLANT_CAPACITY.get(p, 0))
    )
    st.session_state.prod_cost.setdefault(
        p, float(DEFAULT_PROD_COST.get(p, 0))
    )
    st.session_state.transport_cost.setdefault(p, {})
    st.session_state.route_capacity.setdefault(p, {})
    for m in new_markets:
        st.session_state.transport_cost[p].setdefault(
            m, float(DEFAULT_TRANSPORT_COST.get((p, m), 0))
        )
        st.session_state.route_capacity[p].setdefault(
            m, float(DEFAULT_ROUTE_CAPACITY.get((p, m), 0))
        )
for m in new_markets:
    st.session_state.demand.setdefault(
        m, float(DEFAULT_DEMAND.get(m, 0))
    )

st.info("Edit the tables above; changes to Entities auto-sync with parameters below.")

# ---------------------------------------------------------------------------
# Parameter inputs
# ---------------------------------------------------------------------------

st.subheader("Plant")
for p in st.session_state.plants:
    cA, cB = st.columns(2)
    with cA:
        cap = st.number_input(
            f"Capacity \u2014 {p}",
            min_value=0.0,
            value=float(st.session_state.plant_capacity.get(p, 0.0)),
            key=f"cap_{p}",
        )
    with cB:
        pc = st.number_input(
            f"ProdCost \u2014 {p}",
            min_value=0.0,
            value=float(st.session_state.prod_cost.get(p, 0.0)),
            key=f"pc_{p}",
        )
    st.session_state.plant_capacity[p] = float(cap)
    st.session_state.prod_cost[p] = float(pc)

st.subheader("Market")
for m in st.session_state.markets:
    dem = st.number_input(
        f"Demand \u2014 {m}",
        min_value=0.0,
        value=float(st.session_state.demand.get(m, 0.0)),
        key=f"dem_{m}",
    )
    st.session_state.demand[m] = float(dem)

st.subheader("Route Parameters")

st.write("Transport Cost")
for p in st.session_state.plants:
    with st.expander(f"from {p}", expanded=False):
        for m in st.session_state.markets:
            val = st.number_input(
                f"{p} \u2192 {m}",
                min_value=0.0,
                value=float(
                    st.session_state.transport_cost.get(p, {}).get(m, 0.0)
                ),
                key=f"tc_{p}_{m}",
            )
            st.session_state.transport_cost[p][m] = float(val)

st.write("Route Capacity")
for p in st.session_state.plants:
    with st.expander(f"from {p}", expanded=False):
        for m in st.session_state.markets:
            val = st.number_input(
                f"{p} \u2192 {m}",
                min_value=0.0,
                value=float(
                    st.session_state.route_capacity.get(p, {}).get(m, 0.0)
                ),
                key=f"rc_{p}_{m}",
            )
            st.session_state.route_capacity[p][m] = float(val)

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

if run_solve:
    plants = list(st.session_state.plants)
    markets = list(st.session_state.markets)

    if len(plants) == 0 or len(markets) == 0:
        st.session_state.results = {
            "error": "Please define at least one plant and one market.",
            "flows": None,
            "objective": None,
        }
    else:
        warnings = validate_inputs_dicts(
            plants,
            markets,
            st.session_state.plant_capacity,
            st.session_state.prod_cost,
            st.session_state.demand,
            st.session_state.transport_cost,
            st.session_state.route_capacity,
        )
        for w in warnings:
            st.warning(w)

        plant_capacity = {
            p: float(st.session_state.plant_capacity.get(p, 0.0))
            for p in plants
        }
        prod_cost = {
            p: float(st.session_state.prod_cost.get(p, 0.0)) for p in plants
        }
        demand = {
            m: float(st.session_state.demand.get(m, 0.0)) for m in markets
        }
        transport_cost = {
            (p, m): float(
                st.session_state.transport_cost.get(p, {}).get(m, 0.0)
            )
            for p in plants
            for m in markets
        }
        route_capacity = {
            (p, m): float(
                st.session_state.route_capacity.get(p, {}).get(m, 0.0)
            )
            for p in plants
            for m in markets
        }

        with st.spinner("Solving the optimization model..."):
            solution, mdl, err = build_and_solve(
                plants,
                markets,
                plant_capacity,
                demand,
                prod_cost,
                transport_cost,
                route_capacity,
                service_level,
            )

        # ------ post-processing ------
        if err:
            if "no cplex runtime" in str(err).lower():
                with st.spinner("CPLEX missing; trying OR-Tools fallback..."):
                    alt_res, alt_err = solve_with_ortools(
                        plants,
                        markets,
                        plant_capacity,
                        demand,
                        prod_cost,
                        transport_cost,
                        route_capacity,
                        service_level,
                    )
                if alt_err is None and alt_res:
                    flows_df = alt_res["flows_df"]
                    shipped_by_market = alt_res["shipped_by_market"]
                    total_shipped = (
                        float(flows_df["Quantity"].sum())
                        if not flows_df.empty
                        else 0.0
                    )
                    total_demand = float(sum(demand.values()))
                    ratios = []
                    for m in markets:
                        dem = demand.get(m, 0.0)
                        ship = shipped_by_market.get(m, 0.0)
                        ratios.append(ship / dem if dem > 0 else 1.0)
                    min_service = min(ratios) if ratios else 0.0

                    st.session_state.results = {
                        "error": None,
                        "flows": flows_df,
                        "objective": alt_res["objective"],
                        "total_shipped": total_shipped,
                        "total_demand": total_demand,
                        "min_service": min_service,
                        "target_service": service_level,
                        "solver": "OR-Tools (GLOP)",
                    }
                    st.info(
                        "Solved with OR-Tools (GLOP) because CPLEX runtime isn't available."
                    )
                else:
                    st.session_state.results = {
                        "error": err,
                        "flows": None,
                        "objective": None,
                    }
                    st.warning(
                        "No CPLEX runtime found on Streamlit Cloud. "
                        "Try running locally with CPLEX or export LP to solve offline."
                    )
                    st.code(
                        f"{sys.executable} -m pip install cplex",
                        language="bash",
                    )
                    st.code(
                        f"{sys.executable} -m pip install ortools",
                        language="bash",
                    )
                    if mdl is not None:
                        with st.expander("Export model as LP (for offline solving)"):
                            try:
                                st.code(
                                    mdl.export_as_lp_string()[:5000]
                                    + "\n... (truncated)",
                                    language="lp",
                                )
                            except Exception:
                                st.info("LP export not available.")
            else:
                st.session_state.results = {
                    "error": err,
                    "flows": None,
                    "objective": None,
                }
        elif (solution is None) or (mdl is None):
            with st.spinner("Trying OR-Tools fallback..."):
                alt_res, alt_err = solve_with_ortools(
                    plants,
                    markets,
                    plant_capacity,
                    demand,
                    prod_cost,
                    transport_cost,
                    route_capacity,
                    service_level,
                )
            if alt_err is None and alt_res:
                flows_df = alt_res["flows_df"]
                shipped_by_market = alt_res["shipped_by_market"]
                total_shipped = (
                    float(flows_df["Quantity"].sum())
                    if not flows_df.empty
                    else 0.0
                )
                total_demand = float(sum(demand.values()))
                ratios = []
                for m in markets:
                    dem = demand.get(m, 0.0)
                    ship = shipped_by_market.get(m, 0.0)
                    ratios.append(ship / dem if dem > 0 else 1.0)
                min_service = min(ratios) if ratios else 0.0
                st.session_state.results = {
                    "error": None,
                    "flows": flows_df,
                    "objective": alt_res["objective"],
                    "total_shipped": total_shipped,
                    "total_demand": total_demand,
                    "min_service": min_service,
                    "target_service": service_level,
                    "solver": "OR-Tools (GLOP)",
                }
            else:
                retry_msg = None
                try:
                    if mdl is not None:
                        solution = mdl.solve(log_output=True)
                except Exception as e2:
                    retry_msg = str(e2)

                if solution is None:
                    total_capacity = float(sum(plant_capacity.values()))
                    required = float(service_level * sum(demand.values()))
                    tips = []
                    if total_capacity + 1e-9 < required:
                        tips.append(
                            f"Total capacity ({total_capacity:.1f}) < "
                            f"required service ({required:.1f})."
                        )
                    msg = "No feasible solution found."
                    if retry_msg:
                        msg += f" Retry error: {retry_msg}"
                    st.session_state.results = {
                        "error": msg,
                        "flows": None,
                        "objective": None,
                    }
                    if tips:
                        st.info(
                            "Feasibility tips:\n- " + "\n- ".join(tips)
                        )
                    if mdl is not None:
                        with st.expander("Export model as LP (for offline solving)"):
                            try:
                                st.code(
                                    mdl.export_as_lp_string()[:5000]
                                    + "\n... (truncated)",
                                    language="lp",
                                )
                            except Exception:
                                st.info("LP export not available.")
                else:
                    (
                        flows_df,
                        shipped_by_market,
                        total_shipped,
                        total_demand,
                        min_service,
                    ) = extract_shipment_results(
                        mdl, plants, markets, prod_cost, transport_cost
                    )
                    st.session_state.results = {
                        "error": None,
                        "flows": flows_df,
                        "objective": mdl.objective_value,
                        "total_shipped": total_shipped,
                        "total_demand": total_demand,
                        "min_service": min_service,
                        "target_service": service_level,
                    }
        else:
            (
                flows_df,
                shipped_by_market,
                total_shipped,
                total_demand,
                min_service,
            ) = extract_shipment_results(
                mdl, plants, markets, prod_cost, transport_cost
            )
            st.session_state.results = {
                "error": None,
                "flows": flows_df,
                "objective": mdl.objective_value,
                "total_shipped": total_shipped,
                "total_demand": total_demand,
                "min_service": min_service,
                "target_service": service_level,
            }

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

st.subheader("Results")

res = st.session_state.get("results")
if not res:
    st.info("Use the sidebar to set service level and click Solve.")
elif res.get("error"):
    st.error("Failed to solve the model.")
    msg = str(res["error"])
    if "no cplex runtime" in msg.lower():
        st.warning(
            "No CPLEX runtime found. "
            "Install the CPLEX Python package or configure a solver."
        )
        st.code(f"{sys.executable} -m pip install cplex", language="bash")
    else:
        st.write(msg)
else:
    st.success("Solution Found")
    c1, c2, c3 = st.columns(3)
    c1.metric(label="Total Cost", value=f"${res['objective']:,.2f}")
    c2.metric(label="Total Shipped", value=f"{res['total_shipped']:,.1f}")
    c3.metric(label="Total Demand", value=f"{res['total_demand']:,.1f}")
    if res.get("solver"):
        st.caption(f"Solved with {res['solver']}")
    c4, c5 = st.columns(2)
    c4.metric(
        label="Min Market Service", value=f"{res['min_service']*100:.1f}%"
    )
    c5.metric(
        label="Target Service", value=f"{res['target_service']*100:.1f}%"
    )
    st.divider()
    st.subheader("Shipment Plan")
    flows = res["flows"]
    if flows is not None and not flows.empty:
        ui_dataframe(flows, width="stretch")
    else:
        st.write("All flows are zero.")
