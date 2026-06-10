## Maritime Energy Carrier Optimisation 

This repository contains the code developed for a master's thesis comparing battery-electric and compressed green hydrogen as energy carriers for short-sea RoRo shipping. A Mixed Integer Linear Programming (MILP) model minimises total annual system cost across five components: onboard energy storage, production plant investment, port charging/refuelling infrastructure, energy production, and distribution, while also penalising lost cargo capacity caused by the physical footprint of the energy system.

The model selects the least-cost carrier for each route in a multi-route fleet and produces cost breakdowns, energy chain (WtW) analysis, WtW lifecycle emission estimates, and sensitivity analyses.

Reference: Harbo, K.B. and Skovdahl, H. (2026). "Electric vs. Green Fuels in Shortsea Transport: System Design and Benchmarking." 
Master's thesis, Norwegian University of Science and Technology (NTNU).

---

## Repository structure

| File | Description |
|---|---|
| `Iteration_1.py` | Single vessel, hardcoded parameters. Starting point for model development. |
| `Iteration_2.py` | Multi-vessel model; reads fleet from `vessels_2.xlsx`. |
| `Iteration_3.py` | Main model. Multi-route, multi-vessel fleet; reads `routes.xlsx` and `vessels_3.xlsx`. Includes sensitivity analyses and Sankey diagrams. |
| `Scenario_tech_and_economy.py` | Scenario comparing technology and economic assumptions on the base fleet. |
| `Scenario_WAPS.py` | Scenario using wind-assisted propulsion system (WAPS) assumptions (`vessels_3_WAPS.xlsx`). |
| `Scenario_offshore_charging.py` | Scenario with offshore charging infrastructure (`routes_offshore_charging.xlsx`, `vessels_3_offshore_charging.xlsx`). |
Output is written to `results/<script-name>/` and includes PNG plots and interactive Sankey HTML files.

---

## Dependencies

Python: 3.10 or later (developed on 3.13)

Solver: Gurobi 11 or later (required): 
The model uses [Gurobi](https://www.gurobi.com/) via `gurobipy`. A free academic licence is available at [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/). Without a valid licence the solver will not run.

Python packages:

```
gurobipy>=11.0
pandas>=2.0
openpyxl>=3.1        # for reading .xlsx files
matplotlib>=3.8
plotly>=5.0
numpy>=1.24
```

Install with:

```bash
pip install gurobipy pandas openpyxl matplotlib plotly numpy
```

---

## How to run

Each script is self-contained. Run from the repository root so that the Excel input files are found at their expected relative paths.

```bash
# Main model (base case + sensitivity analyses)
python Iteration_3.py

# Scenario variants
python Scenario_tech_and_economy.py
python Scenario_WAPS.py
python Scenario_offshore_charging.py

# Earlier iterations (development reference only)
python Iteration_1.py   # single vessel, no Excel input needed
python Iteration_2.py   # reads vessels_2.xlsx
```

Each script prints a structured console report and saves figures to `results/`.  
`Iteration_3.py` also saves interactive Sankey HTML files to `results/iteration_3/`.

> Note: The sensitivity sweeps in `Iteration_3.py` solve the MILP dozens to hundreds of times and may take several minutes depending on hardware.

---

## Key model parameters

All cost, efficiency, and technology parameters are defined in the `DEFAULT_PARAMS` dictionary near the top of each script (e.g. `Iteration_3.py`, lines 75–121). Parameters include:

- Efficiency chain: `eta_production`, `eta_distribution`, `eta_conversion` for both carriers
- Physical storage: `energy_volume_density` [m³/MWh], `energy_weight_density` [t/MWh]
- PEMFC power density: `pemfc_volume_density` [MW/m³], `pemfc_weight_density` [MW/t]
- Investment costs: `cost_storage` [USD/MWh], `cost_power_conversion` [USD/MW], `cost_plant` [USD/MW], `cost_port` [USD/MW]
- Operational costs: `cost_production` [USD/MWh primary energy], `cost_distribution` [USD/MWh]
- Lost opportunity cost: `cost_LO_V` [USD/m³], `cost_LO_W` [USD/tonne]
- Lifetimes and discount rate: `T_system`, `T_port`, `T_plant`, `r_*` (all use 5% discount rate)

Any parameter can be overridden per run by passing a dictionary to `solve_shipping_model(..., params={...})`.

WtW lifecycle emission factors are defined in `WTW_EMISSION_FACTORS_KG_PER_MWH` (battery: 25 kg CO₂e/MWh, hydrogen: 73 kg CO₂e/MWh, both per MWh of useful propulsion energy).

---

## What is not included

| Item | Reason |
| `vessels_3.xlsx`, `vessels_2.xlsx`, `vessels_3_WAPS.xlsx`, `vessels_3_offshore_charging.xlsx` | Vessel particulars (DWT, installed power, lane metres, speed, etc.) were sourced from Sea-web. Redistribution is not permitted under the licence terms. |
|---|---|
| `routes.xlsx`, `routes_offshore_charging.xlsx` | Included in the repository. Route distances are based on publicly available port information. |
| `results/` | Generated output files are not committed; re-run the scripts to reproduce them. |

To replicate the results, access to Sea-web (or an equivalent maritime data source) is required to reconstruct the vessel input files. The required column schema is documented in the comment header at the top of each script.

---

## Model overview

The MILP selects one energy carrier (battery or hydrogen) per route to minimise total annualised system cost. The decision problem covers:

- Carrier selection: binary choice per route
- Onboard energy capacity: sized for one or two sailing legs depending on whether midpoint charging infrastructure is available at both ports
- Port infrastructure: binary build decision per port and carrier
- Energy chain: primary energy → production → distribution → conversion → useful propulsion (WtW)
- Lost cargo opportunity cost: energy storage that displaces cargo space is penalised at a market rate

The model is described in full in the accompanying thesis.
