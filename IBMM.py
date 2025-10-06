import sys
import pandas as pd
import streamlit as st
from docplex.mp.model import Model


# UI helpers to support `width` parameter mapping
def ui_data_editor(data, **kwargs):
    width = kwargs.pop('width', None)
    if width is not None:
        kwargs['use_container_width'] = (str(width).lower() == 'stretch')
    return st.data_editor(data, **kwargs)


def ui_dataframe(data, **kwargs):
    width = kwargs.pop('width', None)
    if width is not None:
        kwargs['use_container_width'] = (str(width).lower() == 'stretch')
    return st.dataframe(data, **kwargs)


def build_and_solve(plants, markets, plant_capacity, demand, prod_cost, transport_cost, route_capacity, service_level: float):
    mdl = Model(name="Simple Supply Chain Optimizer")

    # decision variables
    x = {(p, m): mdl.continuous_var(lb=0, name=f"x_{p}_{m}") for p in plants for m in markets}

    # objective: minimize total cost
    mdl.minimize(
        mdl.sum((prod_cost[p] + transport_cost[(p, m)]) * x[(p, m)] for p in plants for m in markets)
    )

    # constraints
    for p in plants:
        mdl.add_constraint(mdl.sum(x[(p, m)] for m in markets) <= plant_capacity[p], ctname=f"capacity_{p}")

    for p in plants:
        for m in markets:
            mdl.add_constraint(x[(p, m)] <= route_capacity[(p, m)], ctname=f"route_{p}_{m}")

    for m in markets:
        mdl.add_constraint(mdl.sum(x[(p, m)] for p in plants) >= service_level * demand[m], ctname=f"demand_{m}")

    try:
        solution = mdl.solve()
    except Exception as e:
        return None, mdl, str(e)

    return solution, mdl, None


st.set_page_config(page_title="Simple Supply Chain Optimizer", page_icon="📦", layout="wide")
st.title("Simple Supply Chain Optimizer")
st.caption("Innovation XLab • IBM Decision Optimization (CPLEX Modeling for Python) — plan shipments to meet demand at minimal cost.")
st.markdown(
    """
    Built by **Innovation XLab** for the IBM project, this app uses **IBM Decision Optimization CPLEX Modeling for Python (docplex)** to optimize shipments between plants and markets.

    How to use:
    - Add Plants and Markets in the tables below.
    - Enter each plant's capacity and production cost, and each market's demand.
    - Set transport costs and route capacities for each Plant → Market pair.
    - Choose a service level (%) in the sidebar and click Solve.
    - Review results: total cost, shipped quantities, service metrics, and the shipment plan table.
    """
)

# Default data
DEFAULT_PLANTS = ['A', 'B']
DEFAULT_MARKETS = ['North', 'East', 'West']
DEFAULT_PLANT_CAPACITY = {'A': 600, 'B': 500}
DEFAULT_PROD_COST = {'A': 5.2, 'B': 4.7}
DEFAULT_DEMAND = {'North': 350, 'East': 280, 'West': 220}
DEFAULT_TRANSPORT_COST = {
    ('A', 'North'): 1.2, ('A', 'East'): 1.4, ('A', 'West'): 1.6,
    ('B', 'North'): 1.3, ('B', 'East'): 1.1, ('B', 'West'): 1.5
}
DEFAULT_ROUTE_CAPACITY = {
    ('A', 'North'): 400, ('A', 'East'): 300, ('A', 'West'): 300,
    ('B', 'North'): 350, ('B', 'East'): 330, ('B', 'West'): 300
}

def ensure_session_defaults():
    # Core entities
    if 'plants' not in st.session_state:
        st.session_state['plants'] = DEFAULT_PLANTS.copy()
    if 'markets' not in st.session_state:
        st.session_state['markets'] = DEFAULT_MARKETS.copy()
    # Text inputs for entities
    if 'plants_text' not in st.session_state:
        st.session_state['plants_text'] = "\n".join(st.session_state['plants'])
    if 'markets_text' not in st.session_state:
        st.session_state['markets_text'] = "\n".join(st.session_state['markets'])
    # Parameter dictionaries
    if 'plant_capacity' not in st.session_state:
        st.session_state['plant_capacity'] = DEFAULT_PLANT_CAPACITY.copy()
    if 'prod_cost' not in st.session_state:
        st.session_state['prod_cost'] = DEFAULT_PROD_COST.copy()
    if 'demand' not in st.session_state:
        st.session_state['demand'] = DEFAULT_DEMAND.copy()
    if 'transport_cost' not in st.session_state:
        # nested dict: plant -> market -> cost
        tc = {p: {} for p in st.session_state['plants']}
        for p in st.session_state['plants']:
            for m in st.session_state['markets']:
                tc[p][m] = DEFAULT_TRANSPORT_COST.get((p, m), 0.0)
        st.session_state['transport_cost'] = tc
    if 'route_capacity' not in st.session_state:
        rc = {p: {} for p in st.session_state['plants']}
        for p in st.session_state['plants']:
            for m in st.session_state['markets']:
                rc[p][m] = DEFAULT_ROUTE_CAPACITY.get((p, m), 0.0)
        st.session_state['route_capacity'] = rc
    if 'results' not in st.session_state:
        st.session_state['results'] = None
    # Entities tables defaults
    if 'plants_df' not in st.session_state:
        st.session_state['plants_df'] = pd.DataFrame({'Plant': st.session_state['plants']})
    if 'markets_df' not in st.session_state:
        st.session_state['markets_df'] = pd.DataFrame({'Market': st.session_state['markets']})


def load_preset(name: str):
    plants = DEFAULT_PLANTS.copy()
    markets = DEFAULT_MARKETS.copy()

    plant_capacity = DEFAULT_PLANT_CAPACITY.copy()
    prod_cost = DEFAULT_PROD_COST.copy()
    demand = DEFAULT_DEMAND.copy()
    transport_cost = DEFAULT_TRANSPORT_COST.copy()
    route_capacity = DEFAULT_ROUTE_CAPACITY.copy()

    if name == "High Demand":
        demand = {k: v * 1.2 for k, v in demand.items()}
    elif name == "Low Capacity":
        plant_capacity = {k: v * 0.8 for k, v in plant_capacity.items()}

    return plants, markets, plant_capacity, prod_cost, demand, transport_cost, route_capacity


def apply_preset(name: str):
    plants, markets, plant_capacity, prod_cost, demand, transport_cost, route_capacity = load_preset(name)
    # Update entities
    st.session_state['plants'] = plants
    st.session_state['markets'] = markets
    st.session_state['plants_text'] = "\n".join(plants)
    st.session_state['markets_text'] = "\n".join(markets)
    # Update parameters
    st.session_state['plant_capacity'] = plant_capacity
    st.session_state['prod_cost'] = prod_cost
    st.session_state['demand'] = demand
    # Update route parameters (nested dicts)
    tc = {p: {} for p in plants}
    rc = {p: {} for p in plants}
    for p in plants:
        for m in markets:
            tc[p][m] = transport_cost.get((p, m), 0.0)
            rc[p][m] = route_capacity.get((p, m), 0.0)
    st.session_state['transport_cost'] = tc
    st.session_state['route_capacity'] = rc


def validate_inputs_dicts(plants, markets, plant_capacity, prod_cost, demand, transport_cost, route_capacity):
    messages = []
    # names
    if any(not p for p in plants):
        messages.append("Plants list contains empty names.")
    if len(set(plants)) != len(plants):
        messages.append("Duplicate Plant names are not allowed.")
    if any(not m for m in markets):
        messages.append("Markets list contains empty names.")
    if len(set(markets)) != len(markets):
        messages.append("Duplicate Market names are not allowed.")
    # plant parameters 
    for p in plants:
        if float(plant_capacity.get(p, 0)) < 0:
            messages.append(f"Capacity for plant {p} must be non-negative.")
        if float(prod_cost.get(p, 0)) < 0:
            messages.append(f"ProdCost for plant {p} must be non-negative.")
    # market demand 
    for m in markets:
        if float(demand.get(m, 0)) < 0:
            messages.append(f"Demand for market {m} must be non-negative.")
    # route parameters 
    for p in plants:
        for m in markets:
            if float(transport_cost.get(p, {}).get(m, 0)) < 0:
                messages.append(f"Transport Cost for {p}→{m} must be non-negative.")
            if float(route_capacity.get(p, {}).get(m, 0)) < 0:
                messages.append(f"Route Capacity for {p}→{m} must be non-negative.")
    return messages


def parse_lines(text: str):
    return [line.strip() for line in text.splitlines() if line.strip()]


ensure_session_defaults()

with st.sidebar:
    st.header("Controls")
    service_level = st.slider("Service level (%)", min_value=0, max_value=100, value=100, step=1, help="Percentage of demand to satisfy.") / 100
    run_solve = st.button("Solve", type="primary")

c1, c2 = st.columns(2)
with c1:
    plants_df = ui_data_editor(
        st.session_state['plants_df'],
        num_rows='dynamic',
        width='stretch',
        key='plants_editor',
        column_config={
            'Plant': st.column_config.TextColumn('Plant', help="Add/modify Plant names here")
        }
    )
    st.session_state['plants_df'] = plants_df
with c2:
    markets_df = ui_data_editor(
        st.session_state['markets_df'],
        num_rows='dynamic',
        width='stretch',
        key='markets_editor',
        column_config={
            'Market': st.column_config.TextColumn('Market', help="Add/modify Market names here")
        }
    )
    st.session_state['markets_df'] = markets_df
# Parse and sync entities from tables
new_plants = list(st.session_state['plants_df']['Plant'].dropna().astype(str))
new_markets = list(st.session_state['markets_df']['Market'].dropna().astype(str))
st.session_state['plants'] = new_plants
st.session_state['markets'] = new_markets
# Align parameter dicts to new entities
for p in list(st.session_state['plant_capacity'].keys()):
    if p not in new_plants:
        st.session_state['plant_capacity'].pop(p)
        st.session_state['prod_cost'].pop(p, None)
        st.session_state['transport_cost'].pop(p, None)
        st.session_state['route_capacity'].pop(p, None)
for m in list(st.session_state['demand'].keys()):
    if m not in new_markets:
        st.session_state['demand'].pop(m)
for p in new_plants:
    st.session_state['plant_capacity'].setdefault(p, float(DEFAULT_PLANT_CAPACITY.get(p, 0)))
    st.session_state['prod_cost'].setdefault(p, float(DEFAULT_PROD_COST.get(p, 0)))
    st.session_state['transport_cost'].setdefault(p, {})
    st.session_state['route_capacity'].setdefault(p, {})
    for m in new_markets:
        st.session_state['transport_cost'][p].setdefault(m, float(DEFAULT_TRANSPORT_COST.get((p, m), 0)))
        st.session_state['route_capacity'][p].setdefault(m, float(DEFAULT_ROUTE_CAPACITY.get((p, m), 0)))
for m in new_markets:
    st.session_state['demand'].setdefault(m, float(DEFAULT_DEMAND.get(m, 0)))
st.info("Edit the tables above; changes to Entities auto-sync with parameters below.")
st.subheader("Entities")


# st.subheader("Plant & Market Parameters")


st.subheader("Plant")
for p in st.session_state['plants']:
    cA, cB = st.columns(2)
    with cA:
        cap = st.number_input(f"Capacity — {p}", min_value=0.0, value=float(st.session_state['plant_capacity'].get(p, 0.0)), key=f"cap_{p}")
    with cB:
        pc = st.number_input(f"ProdCost — {p}", min_value=0.0, value=float(st.session_state['prod_cost'].get(p, 0.0)), key=f"pc_{p}")
    st.session_state['plant_capacity'][p] = float(cap)
    st.session_state['prod_cost'][p] = float(pc)

st.subheader("Market")
for m in st.session_state['markets']:
    dem = st.number_input(f"Demand — {m}", min_value=0.0, value=float(st.session_state['demand'].get(m, 0.0)), key=f"dem_{m}")
    st.session_state['demand'][m] = float(dem)


st.subheader("Route Parameters")
rleft, rright = st.columns(2)

st.write("Transport Cost")
for p in st.session_state['plants']:
    with st.expander(f"from {p}", expanded=False):
        for m in st.session_state['markets']:
            val = st.number_input(
                f"{p} → {m}", min_value=0.0,
                value=float(st.session_state['transport_cost'].get(p, {}).get(m, 0.0)),
                key=f"tc_{p}_{m}"
            )
            st.session_state['transport_cost'][p][m] = float(val)

st.write("Route Capacity")
for p in st.session_state['plants']:
    with st.expander(f"from {p}", expanded=False):
        for m in st.session_state['markets']:
            val = st.number_input(
                f"{p} → {m}", min_value=0.0,
                value=float(st.session_state['route_capacity'].get(p, {}).get(m, 0.0)),
                key=f"rc_{p}_{m}"
            )
            st.session_state['route_capacity'][p][m] = float(val)

if run_solve:
    plants = list(st.session_state['plants'])
    markets = list(st.session_state['markets'])

    if len(plants) == 0 or len(markets) == 0:
        st.session_state['results'] = {'error': 'Please define at least one plant and one market.', 'flows': None, 'objective': None}
    else:
        warnings = validate_inputs_dicts(
            plants, markets,
            st.session_state['plant_capacity'],
            st.session_state['prod_cost'],
            st.session_state['demand'],
            st.session_state['transport_cost'],
            st.session_state['route_capacity']
        )
        
        for w in warnings:
            st.warning(w)
        plant_capacity = {p: float(st.session_state['plant_capacity'].get(p, 0.0)) for p in plants}
        prod_cost = {p: float(st.session_state['prod_cost'].get(p, 0.0)) for p in plants}
        demand = {m: float(st.session_state['demand'].get(m, 0.0)) for m in markets}
        transport_cost = {(p, m): float(st.session_state['transport_cost'].get(p, {}).get(m, 0.0)) for p in plants for m in markets}
        route_capacity = {(p, m): float(st.session_state['route_capacity'].get(p, {}).get(m, 0.0)) for p in plants for m in markets}

        with st.spinner("Solving the optimization model..."):
            # 一次求解（build_and_solve 內部可改為 mdl.solve(log_output=True)；這裡先呼叫既有函式）
            solution, mdl, err = build_and_solve(
                plants, markets, plant_capacity, demand,
                prod_cost, transport_cost, route_capacity, service_level
            )

        # === 統一後處理 ===
        if err:
            st.session_state['results'] = {'error': err, 'flows': None, 'objective': None}
            # 針對常見的 CPLEX Runtime 問題給出指引
            if "no cplex runtime" in str(err).lower():
                st.warning("No CPLEX runtime found on Streamlit Cloud. Try running locally with CPLEX or export LP to solve offline.")
                st.code(f"{sys.executable} -m pip install cplex", language="bash")
                if mdl is not None:
                    with st.expander("Export model as LP (for offline solving)"):
                        try:
                            st.code(mdl.export_as_lp_string()[:5000] + "\n... (truncated)", language="lp")
                        except Exception:
                            st.info("LP export not available.")
        elif (solution is None) or (mdl is None):
            # 嘗試在雲端再跑一次並開啟 log（有時可得到更清楚的訊息）
            retry_msg = None
            try:
                solution = mdl.solve(log_output=True) if mdl is not None else None
            except Exception as e2:
                retry_msg = str(e2)

            if (solution is None):
                # 快速可行性診斷：總產能 vs. 服務水準 × 總需求
                total_capacity = float(sum(plant_capacity.values()))
                required = float(service_level * sum(demand.values()))
                tips = []
                if total_capacity + 1e-9 < required:
                    tips.append(f"Total capacity ({total_capacity:.1f}) < required service ({required:.1f}).")
                # 構造錯誤訊息
                msg = "No feasible solution found."
                if retry_msg:
                    msg += f" Retry error: {retry_msg}"
                st.session_state['results'] = {'error': msg, 'flows': None, 'objective': None}
                if tips:
                    st.info("Feasibility tips:\n- " + "\n- ".join(tips))
                if mdl is not None:
                    with st.expander("Export model as LP (for offline solving)"):
                        try:
                            st.code(mdl.export_as_lp_string()[:5000] + "\n... (truncated)", language="lp")
                        except Exception:
                            st.info("LP export not available.")
            else:
                # 進入成功路徑（與下面相同）
                rows = []
                shipped_by_market = {m: 0.0 for m in markets}
                for v in mdl.iter_variables():
                    qty = v.solution_value
                    if qty and qty > 0:
                        # 安全分割：只切兩次，避免名稱含底線導致錯位
                        parts = v.name.split("_", 2)  # e.g. ["x", "PlantName", "Market_Name_With_Underscore"]
                        p = parts[1] if len(parts) >= 2 else "?"
                        m = parts[2] if len(parts) >= 3 else v.name
                        unit_cost = prod_cost.get(p, 0.0) + transport_cost.get((p, m), 0.0)
                        rows.append({"Plant": p, "Market": m, "Quantity": round(qty, 2), "UnitCost": round(unit_cost, 2)})
                        if m in shipped_by_market:
                            shipped_by_market[m] += float(qty)

                flows_df = pd.DataFrame(rows)
                total_shipped = float(flows_df['Quantity'].sum()) if not flows_df.empty else 0.0
                total_demand = float(sum(demand.values()))
                ratios = []
                for m in markets:
                    dem = demand.get(m, 0.0)
                    ship = shipped_by_market.get(m, 0.0)
                    ratios.append(ship / dem if dem > 0 else 1.0)
                min_service = min(ratios) if ratios else 0.0

                st.session_state['results'] = {
                    'error': None,
                    'flows': flows_df,
                    'objective': mdl.objective_value,
                    'total_shipped': total_shipped,
                    'total_demand': total_demand,
                    'min_service': min_service,
                    'target_service': service_level,
                }
        else:
            # 直接成功的情況
            rows = []
            shipped_by_market = {m: 0.0 for m in markets}
            for v in mdl.iter_variables():
                qty = v.solution_value
                if qty and qty > 0:
                    # 安全分割：只切兩次，避免名稱含底線導致錯位
                    parts = v.name.split("_", 2)
                    p = parts[1] if len(parts) >= 2 else "?"
                    m = parts[2] if len(parts) >= 3 else v.name
                    unit_cost = prod_cost.get(p, 0.0) + transport_cost.get((p, m), 0.0)
                    rows.append({"Plant": p, "Market": m, "Quantity": round(qty, 2), "UnitCost": round(unit_cost, 2)})
                    if m in shipped_by_market:
                        shipped_by_market[m] += float(qty)

            flows_df = pd.DataFrame(rows)
            total_shipped = float(flows_df['Quantity'].sum()) if not flows_df.empty else 0.0
            total_demand = float(sum(demand.values()))
            ratios = []
            for m in markets:
                dem = demand.get(m, 0.0)
                ship = shipped_by_market.get(m, 0.0)
                ratios.append(ship / dem if dem > 0 else 1.0)
            min_service = min(ratios) if ratios else 0.0

            st.session_state['results'] = {
                'error': None,
                'flows': flows_df,
                'objective': mdl.objective_value,
                'total_shipped': total_shipped,
                'total_demand': total_demand,
                'min_service': min_service,
                'target_service': service_level,
            }


st.subheader("Results")


res = st.session_state.get('results')
if not res:
    st.info("Use the sidebar to set service level and click Solve.")
else:
    if res.get('error'):
        st.error("Failed to solve the model.")
        msg = str(res['error'])
        if "no cplex runtime" in msg.lower():
            st.warning("No CPLEX runtime found. Install the CPLEX Python package or configure a solver.")
            st.code(f"{sys.executable} -m pip install cplex", language="bash")
        else:
            st.write(msg)
    else:
        st.success("Solution Found")
        c1, c2, c3 = st.columns(3)
        c1.metric(label="Total Cost", value=f"${res['objective']:,.2f}")
        c2.metric(label="Total Shipped", value=f"{res['total_shipped']:,.1f}")
        c3.metric(label="Total Demand", value=f"{res['total_demand']:,.1f}")
        c4, c5 = st.columns(2)
        c4.metric(label="Min Market Service", value=f"{res['min_service']*100:.1f}%")
        c5.metric(label="Target Service", value=f"{res['target_service']*100:.1f}%")
        st.divider()
        st.subheader("Shipment Plan")
        if res['flows'] is not None and not res['flows'].empty:
            ui_dataframe(res['flows'], width='stretch')
        else:
            st.write("All flows are zero.")
