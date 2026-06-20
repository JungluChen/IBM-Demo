# Simple Supply Chain Optimizer

A **Streamlit** application built by **Innovation XLab** for IBM that uses **Decision Optimization** (CPLEX Modeling for Python / OR-Tools) to plan optimal shipments between plants and markets at minimal total cost.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![IBM CPLEX](https://img.shields.io/badge/IBM_CPLEX-D05229?logo=ibm&logoColor=white)
![OR-Tools](https://img.shields.io/badge/OR--Tools-4285F4?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
  - [Mathematical Model](#mathematical-model)
  - [Solvers](#solvers)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the App](#running-the-app)
- [Usage Guide](#usage-guide)
  - [1. Define Entities](#1-define-entities)
  - [2. Set Parameters](#2-set-parameters)
  - [3. Solve & Review](#3-solve--review)
- [Preset Scenarios](#preset-scenarios)
- [Project Structure](#project-structure)
- [Technical Stack](#technical-stack)
- [Development](#development)
- [License](#license)

---

## Overview

This tool solves a **supply chain flow optimization** problem: given a set of plants with limited production capacity and a set of markets with known demand, determine the most cost-efficient shipment quantities from each plant to each market while respecting route capacity limits and a configurable service level.

It is designed as an interactive demonstration of **IBM Decision Optimization** technology, with an automatic fallback to **Google OR-Tools** so it runs on any environment (including Streamlit Cloud) without requiring a CPLEX license.

---

## Features

- **Interactive entity management** — Add or remove plants and markets dynamically via editable tables.
- **Full cost modelling** — Set per-plant production cost, per-route transport cost, plant capacity, route capacity, and market demand.
- **Service level control** — Specify the required fulfilment percentage (0–100%) via a sidebar slider.
- **Dual solver backend**:
  - **CPLEX (docplex)** — Used by default; requires IBM ILOG CPLEX Optimization Studio.
  - **OR-Tools (GLOP)** — Automatic fallback when CPLEX is unavailable.
- **Preset scenarios** — Quickly load "High Demand" or "Low Capacity" configurations.
- **Model export** — Export the mathematical model as LP format for offline solving.
- **Feasibility diagnostics** — Warnings and hints when the problem is infeasible (e.g., total capacity < required service level).
- **Rich results** — Total cost, shipped quantities, market-level service percentages, and a detailed shipment plan table.

---

## How It Works

### Mathematical Model

The app formulates a **linear programming (LP)** problem:

| Component | Description |
|---|---|
| **Decision variables** | `x[p, m]` — quantity shipped from plant `p` to market `m` |
| **Objective** | Minimize total cost = sum of `(production_cost[p] + transport_cost[p, m]) * x[p, m]` |
| **Constraints** | 1. Plant capacity: shipments out of each plant <= its capacity<br>2. Route capacity: each plant–market flow <= its route limit<br>3. Demand fulfilment: shipments into each market >= `service_level * demand[m]` |
| **Service level** | A percentage (0–100%) that scales the demand constraint, allowing partial fulfilment analysis |

### Solvers

- **Primary: IBM CPLEX (docplex)** — Industrial-strength LP solver. Requires a local CPLEX installation.
- **Fallback: Google OR-Tools (GLOP)** — Open-source LP solver. Activated automatically when CPLEX is not found (e.g., on Streamlit Cloud, CI, or fresh environments).

The app transparently selects the available solver; the results display indicates which solver was used.

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/JungluChen/IBM-Demo.git
cd IBM-Demo

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

For CPLEX support (optional), install the IBM CPLEX package:

```bash
pip install cplex
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

---

## Usage Guide

### 1. Define Entities

- Use the **Plant** and **Market** tables at the top of the page to add or remove entities.
- Changes auto-sync to the parameter sections below.

### 2. Set Parameters

For each **plant**:
- **Capacity** — maximum total output
- **ProdCost** — cost per unit produced

For each **market**:
- **Demand** — total quantity required

For each **route** (plant → market pair):
- **Transport Cost** — cost per unit shipped
- **Route Capacity** — maximum flow on this specific lane

### 3. Solve & Review

- Set the **Service level** in the sidebar (default: 100%).
- Click **Solve**.
- Review the results panel:
  - **Total Cost** — objective value of the optimal solution
  - **Total Shipped / Total Demand** — aggregate flow vs. requirements
  - **Min Market Service** — the lowest fulfilment percentage across all markets
  - **Shipment Plan** — detailed table of every non-zero flow with quantities and unit costs

---

## Preset Scenarios

| Preset | Description |
|---|---|
| **Default** | 2 plants (A, B), 3 markets (North, East, West) with baseline values |
| **High Demand** | All market demands increased by 20% |
| **Low Capacity** | All plant capacities decreased by 20% |

Future versions may include more complex scenarios (e.g., seasonal demand shifts, plant outages).

---

## Project Structure

```
IBM-Demo/
├── app.py                 # Streamlit entry point (UI rendering & user interaction)
├── src/
│   ├── __init__.py        # Package marker
│   ├── config.py          # Default data, presets, and constants
│   ├── solver.py          # CPLEX and OR-Tools solver implementations
│   └── utils.py           # UI helpers and input validation
├── requirements.txt       # Python package dependencies
├── .gitignore             # Git ignore rules
└── README.md              # Project documentation (this file)
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit UI layout, session state management, solve orchestration, results display |
| `src/config.py` | Default input data, preset definitions, `load_preset()` |
| `src/solver.py` | `build_and_solve()` (CPLEX), `solve_with_ortools()` (fallback), `extract_shipment_results()` |
| `src/utils.py` | `ui_data_editor()`, `ui_dataframe()`, `validate_inputs_dicts()` |

---

## Technical Stack

| Technology | Role |
|---|---|
| [Streamlit](https://streamlit.io/) | Web UI framework |
| [IBM CPLEX (docplex)](https://www.ibm.com/products/ilog-cplex-optimization-studio) | Primary LP solver |
| [Google OR-Tools](https://developers.google.com/optimization) | Fallback LP solver (GLOP) |
| [pandas](https://pandas.pydata.org/) | Data manipulation and tabular display |

---

## Development

```bash
# Install dev dependencies (optional)
pip install pytest mypy ruff

# Run all checks
ruff check src/ app.py
python -m pytest
```

### Adding a New Preset

1. Add the preset entry in `src/config.py` in the `PRESETS` dictionary.
2. Implement the parameter modifications in the `load_preset()` function.
3. (Optional) Add a UI trigger in `app.py` under the sidebar controls.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

*Built by **Innovation XLab** as an IBM Decision Optimization demonstration project.*
