"""
Optimization solver implementations for the Supply Chain Optimizer.

Provides two solver backends:
- CPLEX (docplex) — primary, requires IBM ILOG CPLEX Optimization Studio
- OR-Tools (GLOP) — fallback, runs on any environment
"""

import pandas as pd
from docplex.mp.model import Model


def build_and_solve(
    plants, markets, plant_capacity, demand,
    prod_cost, transport_cost, route_capacity, service_level: float
):
    """
    Build and solve a supply chain flow optimization model using CPLEX (docplex).

    Returns (solution, model, error_string).
    """
    mdl = Model(name="Simple Supply Chain Optimizer")

    # Decision variables: shipment quantity from plant p to market m
    x = {
        (p, m): mdl.continuous_var(lb=0, name=f"x_{p}_{m}")
        for p in plants for m in markets
    }

    # Objective: minimize total (production + transport) cost
    mdl.minimize(
        mdl.sum(
            (prod_cost[p] + transport_cost[(p, m)]) * x[(p, m)]
            for p in plants for m in markets
        )
    )

    # Constraints: plant capacity
    for p in plants:
        mdl.add_constraint(
            mdl.sum(x[(p, m)] for m in markets) <= plant_capacity[p],
            ctname=f"capacity_{p}",
        )

    # Constraints: route capacity
    for p in plants:
        for m in markets:
            mdl.add_constraint(
                x[(p, m)] <= route_capacity[(p, m)],
                ctname=f"route_{p}_{m}",
            )

    # Constraints: service-level demand satisfaction
    for m in markets:
        mdl.add_constraint(
            mdl.sum(x[(p, m)] for p in plants) >= service_level * demand[m],
            ctname=f"demand_{m}",
        )

    try:
        solution = mdl.solve()
    except Exception as e:
        return None, mdl, str(e)

    return solution, mdl, None


def solve_with_ortools(
    plants, markets, plant_capacity, demand,
    prod_cost, transport_cost, route_capacity, service_level: float
):
    """
    Fallback LP solver using OR-Tools (GLOP).

    Used when CPLEX runtime is unavailable (e.g. on Streamlit Cloud).
    Returns (result_dict, error_string) where result_dict contains:
      - flows_df: DataFrame of non-zero shipments
      - objective: total cost value
      - shipped_by_market: dict mapping market -> shipped quantity
    """
    try:
        from ortools.linear_solver import pywraplp
    except Exception as e:
        return None, f"OR-Tools not available: {e}"

    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        return None, "Failed to create OR-Tools GLOP solver."

    # Decision variables with route capacity as upper bound
    x = {}
    for p in plants:
        for m in markets:
            ub = float(route_capacity[(p, m)])
            x[(p, m)] = solver.NumVar(0.0, ub, f"x_{p}_{m}")

    # Objective
    objective_expr = solver.Sum(
        (float(prod_cost[p]) + float(transport_cost[(p, m)])) * x[(p, m)]
        for p in plants for m in markets
    )
    solver.Minimize(objective_expr)

    # Constraints
    for p in plants:
        solver.Add(
            solver.Sum(x[(p, m)] for m in markets) <= float(plant_capacity[p])
        )
    for m in markets:
        solver.Add(
            solver.Sum(x[(p, m)] for p in plants) >= float(service_level * demand[m])
        )

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return None, "No feasible solution with OR-Tools."

    rows = []
    shipped_by_market = {m: 0.0 for m in markets}
    for (p, m), var in x.items():
        qty = var.solution_value()
        if qty and qty > 1e-8:
            unit_cost = float(prod_cost[p]) + float(transport_cost[(p, m)])
            rows.append({
                "Plant": p,
                "Market": m,
                "Quantity": round(qty, 2),
                "UnitCost": round(unit_cost, 2),
            })
            shipped_by_market[m] += float(qty)

    flows_df = pd.DataFrame(rows)
    objective_value = solver.Objective().Value()

    return {
        "flows_df": flows_df,
        "shipped_by_market": shipped_by_market,
        "objective": objective_value,
    }, None


def extract_shipment_results(mdl, plants, markets, prod_cost, transport_cost):
    """
    Extract shipment flows, shipped-by-market, and total cost from a solved CPLEX model.
    """
    rows = []
    shipped_by_market = {m: 0.0 for m in markets}

    for v in mdl.iter_variables():
        qty = v.solution_value
        if qty and qty > 0:
            parts = v.name.split("_", 2)
            p = parts[1] if len(parts) >= 2 else "?"
            m = parts[2] if len(parts) >= 3 else v.name
            unit_cost = prod_cost.get(p, 0.0) + transport_cost.get((p, m), 0.0)
            rows.append({
                "Plant": p,
                "Market": m,
                "Quantity": round(qty, 2),
                "UnitCost": round(unit_cost, 2),
            })
            if m in shipped_by_market:
                shipped_by_market[m] += float(qty)

    flows_df = pd.DataFrame(rows)
    total_shipped = float(flows_df["Quantity"].sum()) if not flows_df.empty else 0.0
    total_demand = float(sum(demand.values()))
    ratios = []
    for m in markets:
        dem = demand.get(m, 0.0)
        ship = shipped_by_market.get(m, 0.0)
        ratios.append(ship / dem if dem > 0 else 1.0)
    min_service = min(ratios) if ratios else 0.0

    return flows_df, shipped_by_market, total_shipped, total_demand, min_service
