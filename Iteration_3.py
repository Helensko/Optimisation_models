import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import plotly.graph_objects as go
import os

# =============================================================================
# OUTPUT FOLDER
# =============================================================================

OUT_DIR = 'results/iteration_3/'
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# LOAD DATA FROM EXCEL
# routes.xlsx   : route_id, origin, destination, distance_nm
# vessels_3.xlsx: route_id, vessel_id, voyages_per_year, vessel_type, count, speed, 
#                 dwt, teu, lane_meters, energy_per_nm, ower_kW, volume_available,
#                 weight_available, volume_max, weight_max
#
# Each vessel row is assigned to exactly one route via route_id.
# Multiple vessel types (and multiple physical vessels via count) can share
# the same route - that is the intended regional fleet model.
# =============================================================================

df_routes_raw  = pd.read_excel('routes.xlsx')
df_vessels_raw = pd.read_excel('vessels_3.xlsx')

vessels_expanded_rows = []
for _, row in df_vessels_raw.iterrows():
    for i in range(int(row['count'])):
        new_row = row.copy()
        new_row['vessel_id'] = f"{row['vessel_id']}_{i+1}"
        vessels_expanded_rows.append(new_row)
df_vessels = pd.DataFrame(vessels_expanded_rows).reset_index(drop=True)


# =============================================================================
# TRANSPORT WORK
# =============================================================================
    
def get_transport_work_effective(row, length, delta_V):
    '''
    Calculates the effective transport work based on remaining 
    cargo space after the energy system is installed. 

    Utilizing delta_V to adjust available capacity.
    100% utilization of remaining cargo space is assumed.
    '''
    vtype = str(row['vessel_type']).lower()
    if vtype == 'roro':
        lane_width   = 2.5
        lane_height  = 4.0
        m3_per_lm    = lane_width * lane_height
        lm_lost      = delta_V / m3_per_lm
        remaining_lm = max(0, row['lane_meters'] - lm_lost)
        return remaining_lm * length * 2
    elif vtype == 'container':
        remaining_fraction = max(0, (row['volume_max'] - delta_V) / row['volume_max'])
        return row['teu'] * remaining_fraction * length * 2
    else:
        remaining_fraction = max(0, (row['volume_max'] - delta_V) / row['volume_max'])
        return row['dwt'] * remaining_fraction * length * 2


# =============================================================================
# DEFAULT PARAMETERS
# Override any of these by passing a dict to solve_shipping_model()
# =============================================================================

DEFAULT_PARAMS = {
    # Lost opportunity cost
    'cost_LO_V':            15,         # [USD/m³]
    'cost_LO_W':            20,         # [USD/tonne]

    # Annuity factors
    'r_port':               0.05,
    'T_port':               40,
    'r_plant':              0.05,
    'T_plant':              17,
    'T_system': {'battery': 10,   'hydrogen': 15},
    'r_system': {'battery': 0.05, 'hydrogen': 0.05},
    'T_power_conversion_system': {'battery': 10, 'hydrogen': 6},
    'r_power_conversion_system': {'battery': 0.05, 'hydrogen': 0.05},

    # Big-M
    'M':                    10_000_000,

    # Efficiency factors
    'eta_production':   {'battery': 1,    'hydrogen': 0.6},
    'eta_distribution': {'battery': 0.85, 'hydrogen': 0.98},
    'eta_conversion':   {'battery': 0.95, 'hydrogen': 0.52},

    # Physical storage requirements
    'energy_volume_density': {'battery': 11.7, 'hydrogen': 1.16},  # [m³/MWh]
    'energy_weight_density': {'battery': 15.4, 'hydrogen': 0.58},  # [t/MWh]

    # PEMFC power density (hydrogen system only)
    'pemfc_volume_density': 0.2,    # [MW/m³]
    'pemfc_weight_density': 0.4,    # [MW/t]

    # Usable capacity fraction
    'usable_capacity_fraction': {'battery': 0.8, 'hydrogen': 1.0},

    # Onboard system investment
    'cost_storage':          {'battery': 434_498,   'hydrogen': 12_700},     # [USD/MWh]
    'cost_power_conversion': {'battery': 0,         'hydrogen': 1_014_000},  # [USD/MW]

    # Production plant investment
    'cost_plant':            {'battery': 0,         'hydrogen': 1_424_000},  # [USD/MW]
    'plant_full_load_hours': {'battery': 8760,      'hydrogen': 4000},       # [h/year]

    # Operational costs
    'cost_production':   {'battery': 120,     'hydrogen': 120},     # [USD/MWh primary energy]
    'cost_distribution': {'battery': 0,       'hydrogen': 16.6},    # [USD/MWh entering distribution]
    'cost_port':         {'battery': 255_170, 'hydrogen': 356_000}, # [USD/MW]
}

# WtW lifecycle emission factors (per MWh of useful propulsion energy, E_wake)
# Basis: mechanical energy delivered to propeller shaft (not primary or upstream energy)
# Sources: thesis background chapter
#   Battery-electric (Elec-BE):           0.025 g CO2e/Wh  =  25 kg CO2e/MWh
#   Compressed green H2 + PEMFC (e-CH2):  0.073 g CO2e/Wh  =  73 kg CO2e/MWh
WTW_EMISSION_FACTORS_KG_PER_MWH = {
    'battery':  25.0,
    'hydrogen': 73.0,
}


# =============================================================================
# SHADOW PRICE HELPERS
# =============================================================================

def print_relevant_binding_constraints(model, tol=1e-6):
    relevant_keywords = [
        'physical', 'LO', 'C_LO', 'onboard_energy_capacity',
        'vessel_port_coupling', 'both_ports_indicator', 'production_plant_power',
        'energy_balance', 'port_capacity', 'charging_tank_limit', 'infra_requires',
    ]
    print()
    print('  Relevant binding constraints:')
    found = False
    for c in model.getConstrs():
        if any(k in c.ConstrName for k in relevant_keywords):
            if abs(c.Slack) <= tol:
                print(f'    {c.ConstrName:55s} Slack={c.Slack:.4g}  RHS={c.RHS:.4g}')
                found = True
    if not found:
        print('    (none)')

def print_relevant_lp_shadow_prices(model, tol=1e-6):
    relevant_keywords = [
        'physical', 'LO', 'C_LO', 'onboard_energy_capacity',
        'vessel_port_coupling', 'both_ports_indicator', 'production_plant_power',
        'energy_balance', 'port_capacity', 'charging_tank_limit', 'infra_requires',
    ]
    relaxed = model.relax()
    relaxed.setParam('OutputFlag', 0)
    relaxed.optimize()
    if relaxed.status == GRB.OPTIMAL:
        print()
        print('  Relevant shadow prices (LP relaxation):')
        found = False
        for c in relaxed.getConstrs():
            if any(k in c.ConstrName for k in relevant_keywords):
                if abs(c.Pi) > tol:
                    print(f'    {c.ConstrName:55s} Pi={c.Pi:.4g}  Slack={c.Slack:.4g}')
                    found = True
        if not found:
            print('    (none)')

# =============================================================================
# MODEL FUNCTION
# =============================================================================

def solve_shipping_model(df_routes, df_vessels, params=None, force_carrier=None, verbose=False):
    '''
    Solve the maritime MILP for a given corridor.
 
    Parameters:
    length : float
        One-way route length [nm]. Model computes round trip internally.
    params : dict, optional
        Override any key in DEFAULT_PARAMS.
    force_carrier : str or None
        'battery' or 'hydrogen' - fix that carrier as the only choice.
        Useful for computing the cost of each technology independently
        across all route lengths (for crossover comparison plots).
 
    Returns:
    dict with results, or None if infeasible.
    '''
    p = {**DEFAULT_PARAMS, **(params or {})}

    ports           = list(dict.fromkeys(df_routes['origin'].tolist() + df_routes['destination'].tolist()))
    energy_carriers = ['battery', 'hydrogen']
    routes          = df_routes['route_id'].tolist()
    vessels         = df_vessels['vessel_id'].tolist()

    # Route parameters
    origin_r = dict(zip(df_routes['route_id'], df_routes['origin']))
    dest_r   = dict(zip(df_routes['route_id'], df_routes['destination']))
    length_r = dict(zip(df_routes['route_id'], df_routes['distance_nm']))

    # Vessel parameters
    route_of         = dict(zip(df_vessels['vessel_id'], df_vessels['route_id']))
    N_v              = dict(zip(df_vessels['vessel_id'], df_vessels['voyages_per_year']))
    energy_per_nm    = dict(zip(df_vessels['vessel_id'], df_vessels['energy_per_nm']))
    volume_available = dict(zip(df_vessels['vessel_id'], df_vessels['volume_available']))
    weight_available = dict(zip(df_vessels['vessel_id'], df_vessels['weight_available']))
    volume_max       = dict(zip(df_vessels['vessel_id'], df_vessels['volume_max']))
    weight_max       = dict(zip(df_vessels['vessel_id'], df_vessels['weight_max']))
    power_kW         = dict(zip(df_vessels['vessel_id'], df_vessels['power_kw']))
    T_port_v         = dict(zip(df_vessels['vessel_id'], df_vessels['lay_time_port']))

    # Derived per-vessel route info (v -> route -> port/length)
    origin_of = {v: origin_r[route_of[v]] for v in vessels}
    dest_of   = {v: dest_r[route_of[v]]   for v in vessels}
    length_of = {v: length_r[route_of[v]] for v in vessels}

    # Parameters
    cost_LO_V             = p['cost_LO_V']
    cost_LO_W             = p['cost_LO_W']
    M                     = p['M']
    eta_prod              = p['eta_production']
    eta_dist              = p['eta_distribution']
    eta_conv              = p['eta_conversion']
    energy_volume_density = p['energy_volume_density']
    energy_weight_density = p['energy_weight_density']
    pemfc_volume_density  = p['pemfc_volume_density']
    pemfc_weight_density  = p['pemfc_weight_density']
    cost_storage          = p['cost_storage']
    cost_power_conversion = p['cost_power_conversion']
    cost_plant            = p['cost_plant']
    cost_port             = p['cost_port']
    plant_full_load_hours = p['plant_full_load_hours']
    cost_production       = p['cost_production']
    cost_dist             = p['cost_distribution']
    F                     = p['usable_capacity_fraction']

    # Annuity factors - annual model
    af_port   = p['r_port']  / (1 - (1 + p['r_port']) **(-p['T_port']))
    af_plant  = p['r_plant'] / (1 - (1 + p['r_plant'])**(-p['T_plant']))
    af_system = {e: p['r_system'][e] / (1 - (1 + p['r_system'][e])**(-p['T_system'][e]))
                 for e in energy_carriers}
    af_power_conversion_system = {e: p['r_power_conversion_system'][e] / (1 - (1 + p['r_power_conversion_system'][e])**(-p['T_power_conversion_system'][e]))
                 for e in energy_carriers}

    # Power converter footprint, volume and weight
    phi_volume = {}
    phi_weight = {}
    for e in energy_carriers:
        for v in vessels:
            if e == 'hydrogen':
                p_conv = power_kW[v] / 1000
                phi_volume[e, v] = p_conv / pemfc_volume_density
                phi_weight[e, v] = p_conv / pemfc_weight_density
            else:
                phi_volume[e, v] = 0.0
                phi_weight[e, v] = 0.0

    model = gp.Model('Maritime_Optimization')
    model.setParam('OutputFlag', 0)

    # Decision variables
    x       = model.addVars(energy_carriers, routes,  vtype=GRB.BINARY,     name='x')
    y       = model.addVars(energy_carriers, ports,   vtype=GRB.BINARY,     name='y')
    q       = model.addVars(energy_carriers, vessels, vtype=GRB.CONTINUOUS, name='q')
    w       = model.addVars(energy_carriers, routes,  vtype=GRB.BINARY,     name='w')
    z       = model.addVars(energy_carriers, ports, vessels, vtype=GRB.CONTINUOUS, name='z')
    delta_V = model.addVars(energy_carriers, vessels, vtype=GRB.CONTINUOUS, name='delta_V')
    delta_W = model.addVars(energy_carriers, vessels, vtype=GRB.CONTINUOUS, name='delta_W')
    C_LO    = model.addVars(energy_carriers, vessels, vtype=GRB.CONTINUOUS, name='C_LO')
    S_port  = model.addVars(energy_carriers, ports,   vtype=GRB.CONTINUOUS, name='S_port')
    P_plant = model.addVars(energy_carriers,          vtype=GRB.CONTINUOUS, name='P_plant')

    # Energy chain variables
    E_wake = model.addVars(energy_carriers, vessels, vtype=GRB.CONTINUOUS, name='E_wake')
    E_conv = model.addVars(energy_carriers, vessels, vtype=GRB.CONTINUOUS, name='E_conv')
    E_dist = model.addVars(energy_carriers, ports,   vtype=GRB.CONTINUOUS, name='E_dist')
    E_prod = model.addVars(energy_carriers,          vtype=GRB.CONTINUOUS, name='E_prod')
    L_prod = model.addVars(energy_carriers,          vtype=GRB.CONTINUOUS, name='L_prod')
    L_dist = model.addVars(energy_carriers, ports,   vtype=GRB.CONTINUOUS, name='L_dist')
    L_conv = model.addVars(energy_carriers, vessels, vtype=GRB.CONTINUOUS, name='L_conv')

    # Objective Function (annual costs)
    model.setObjective(
        gp.quicksum(af_system[e] * cost_storage[e] * q[e, v]
                    + af_power_conversion_system[e] * cost_power_conversion[e] * (power_kW[v] / 1000) * x[e, route_of[v]]
                    for e in energy_carriers for v in vessels)
        + gp.quicksum(af_plant * cost_plant[e] * P_plant[e]
                      for e in energy_carriers)
        + gp.quicksum(af_port  * cost_port[e]  * S_port[e, p]
                      for e in energy_carriers for p in ports)
        + gp.quicksum(cost_production[e] * E_prod[e]
                      for e in energy_carriers)
        + gp.quicksum(cost_dist[e] * E_dist[e, p]
                      for e in energy_carriers for p in ports)
        + gp.quicksum(N_v[v] * C_LO[e, v]
                      for e in energy_carriers for v in vessels),
        GRB.MINIMIZE
    )

    # Constraints

    # (1) Exactly one energy carrier per route
    if force_carrier:
        model.addConstrs((x[force_carrier, r] == 1 for r in routes), name='carrier_forced')
        for e in energy_carriers:
            if e != force_carrier:
                model.addConstrs((x[e, r] == 0 for r in routes), name=f'carrier_blocked_{e}')
    else:
        model.addConstrs(
            (gp.quicksum(x[e, r] for e in energy_carriers) == 1 for r in routes),
            name='carrier_selection'
        )

    # (2) Wake energy demand (round trip, per voyage)
    model.addConstrs(
        (E_wake[e, v] == 2 * length_of[v] * energy_per_nm[v] * x[e, route_of[v]]
         for e in energy_carriers for v in vessels),
        name='wake_energy'
    )

    # (3) Conversion: wake -> onboard energy needed (per voyage)
    model.addConstrs(
        (E_conv[e, v] == E_wake[e, v] / eta_conv[e]
         for e in energy_carriers for v in vessels),
        name='conversion_energy'
    )

    # (4) Distribution: annual energy entering distribution to port
    model.addConstrs(
        (E_dist[e, p] == gp.quicksum(z[e, p, v] for v in vessels) / eta_dist[e]
         for e in energy_carriers for p in ports),
        name='distribution_energy'
    )

    # (5) Production: annual primary energy required
    model.addConstrs(
        (E_prod[e] == gp.quicksum(E_dist[e, p] for p in ports) / eta_prod[e]
         for e in energy_carriers),
        name='production_energy'
    )

    # (6) Production plant power capacity
    model.addConstrs(
        (P_plant[e] == E_prod[e] / plant_full_load_hours[e]
         for e in energy_carriers),
        name='production_plant_power'
    )

    # (7) Energy balance: annual deliveries = N_v * per-voyage onboard demand
    model.addConstrs(
        (gp.quicksum(z[e, p, v] for p in ports) == N_v[v] * E_conv[e, v]
         for e in energy_carriers for v in vessels),
        name='energy_balance'
    )

    # (8) Port throughput capacity (per voyage)
    model.addConstrs(
        (S_port[e, p] >= z[e, p, v] / (N_v[v] * T_port_v[v])
         for e in energy_carriers for p in ports for v in vessels),
        name='port_capacity'
    )

    # (9) Conversion loss (per voyage)
    model.addConstrs(
        (L_conv[e, v] == E_conv[e, v] - E_wake[e, v]
         for e in energy_carriers for v in vessels),
        name='conversion_loss'
    )

    # (10) Distribution loss (annual)
    model.addConstrs(
        (L_dist[e, p] == E_dist[e, p] - gp.quicksum(z[e, p, v] for v in vessels)
         for e in energy_carriers for p in ports),
        name='distribution_loss'
    )

    # (11) Production loss (annual)
    model.addConstrs(
        (L_prod[e] == E_prod[e] - gp.quicksum(E_dist[e, p] for p in ports)
         for e in energy_carriers),
        name='production_loss'
    )

    # (12) Port capacity linked to infrastructure
    model.addConstrs(
        (S_port[e, p] <= M * y[e, p]
         for e in energy_carriers for p in ports),
        name='port_capacity_infrastructure'
    )

    # (13) z = 0 for ports not on vessel's route
    model.addConstrs(
        (z[e, p, v] == 0
         for e in energy_carriers for v in vessels for p in ports
         if p not in {origin_of[v], dest_of[v]}),
        name='z_port_restriction'
    )

    # (14) Vessel-port coupling: energy only where infrastructure exists
    model.addConstrs(
        (z[e, p, v] <= M * y[e, p]
         for e in energy_carriers for v in vessels for p in ports
         if p in {origin_of[v], dest_of[v]}),
        name='vessel_port_coupling'
    )

    # Per-voyage charging at any port cannot exceed onboard tank capacity
    model.addConstrs(
        (z[e, p, v] <= N_v[v] * q[e, v]
         for e in energy_carriers for v in vessels for p in ports
         if p in {origin_of[v], dest_of[v]}),
        name='charging_tank_limit'
    )

    # Infrastructure can only be built for carrier e in port p 
    # if e is selected on at least one route calling at port p
    model.addConstrs(
        (y[e, p] <= gp.quicksum(
            x[e, r] for r in routes if origin_r[r] == p or dest_r[r] == p)
        for e in energy_carriers for p in ports),
        name='infra_requires_carrier_on_route'
    )

    # (15-17) w_{e,r} = 1 iff infrastructure in both ports of route r
    model.addConstrs(
        (w[e, r] <= y[e, origin_r[r]]
         for e in energy_carriers for r in routes),
        name='both_ports_indicator_1'
    )
    model.addConstrs(
        (w[e, r] <= y[e, dest_r[r]]
         for e in energy_carriers for r in routes),
        name='both_ports_indicator_2'
    )
    model.addConstrs(
        (w[e, r] >= y[e, origin_r[r]] + y[e, dest_r[r]] - 1
         for e in energy_carriers for r in routes),
        name='both_ports_indicator_3'
    )

    # (18) Onboard energy capacity (one leg if w=1, full round trip if w=0)
    model.addConstrs(
        (q[e, v] >= (2 * x[e, route_of[v]] - w[e, route_of[v]])
                     * (length_of[v] * energy_per_nm[v] / (eta_conv[e] * F[e]))
         for e in energy_carriers for v in vessels),
        name='onboard_energy_capacity'
    )

    # (19) No capacity for unselected carrier
    model.addConstrs(
        (q[e, v] <= M * x[e, route_of[v]]
         for e in energy_carriers for v in vessels),
        name='zero_capacity_unselected'
    )

    # (20-21) Hard physical caps: energy system cannot exceed available space
    model.addConstrs(
        (energy_volume_density[e] * q[e, v] + phi_volume[e, v] * x[e, route_of[v]]
         <= volume_max[v] * x[e, route_of[v]]
         for e in energy_carriers for v in vessels),
        name='physical_volume_cap'
    )
    model.addConstrs(
        (energy_weight_density[e] * q[e, v] + phi_weight[e, v] * x[e, route_of[v]]
         <= weight_max[v] * x[e, route_of[v]]
         for e in energy_carriers for v in vessels),
        name='physical_weight_cap'
    )

    # (22-23) Hard caps: lost cargo cannot exceed total cargo space
    model.addConstrs(
        (delta_V[e, v] <= volume_max[v] * x[e, route_of[v]]
         for e in energy_carriers for v in vessels),
        name='LO_V_limit'
    )
    model.addConstrs(
        (delta_W[e, v] <= weight_max[v] * x[e, route_of[v]]
         for e in energy_carriers for v in vessels),
        name='LO_W_limit'
    )

    # (24-25) Lost cargo = storage footprint minus freed space from removed system
    model.addConstrs(
        (delta_V[e, v] >= energy_volume_density[e] * q[e, v]
                          + phi_volume[e, v] * x[e, route_of[v]]
                          - volume_available[v] * x[e, route_of[v]]
         for e in energy_carriers for v in vessels),
        name='LO_V_kick_in'
    )
    model.addConstrs(
        (delta_W[e, v] >= energy_weight_density[e] * q[e, v]
                          + phi_weight[e, v] * x[e, route_of[v]]
                          - weight_available[v] * x[e, route_of[v]]
         for e in energy_carriers for v in vessels),
        name='LO_W_kick_in'
    )

    # (26-27) C_LO = max(volume penalty, weight penalty)
    model.addConstrs(
        (C_LO[e, v] >= cost_LO_V * delta_V[e, v]
         for e in energy_carriers for v in vessels),
        name='C_LO_volume'
    )
    model.addConstrs(
        (C_LO[e, v] >= cost_LO_W * delta_W[e, v]
         for e in energy_carriers for v in vessels),
        name='C_LO_weight'
    )

    model.optimize()

    if model.status == GRB.OPTIMAL:
        from collections import Counter
        carrier_by_route = {r: next(e for e in energy_carriers if x[e, r].X > 0.5) for r in routes}
        e_sel            = Counter(carrier_by_route.values()).most_common(1)[0][0]
        ports_with_infra = int(sum(y[e_sel, port].X for port in ports))

        # Annual cost decomposition
        cost_system_out = sum(
            af_system[carrier_by_route[route_of[v]]] * cost_storage[carrier_by_route[route_of[v]]] * q[carrier_by_route[route_of[v]], v].X
            + af_power_conversion_system[carrier_by_route[route_of[v]]] * cost_power_conversion[carrier_by_route[route_of[v]]] * (power_kW[v] / 1000)
            for v in vessels)
        cost_plant_out = sum(af_plant * cost_plant[e] * P_plant[e].X for e in energy_carriers)
        cost_port_out  = sum(af_port * cost_port[e] * S_port[e, p].X for e in energy_carriers for p in ports)
        cost_prod_out  = sum(cost_production[e] * E_prod[e].X for e in energy_carriers)
        cost_dist_out  = sum(cost_dist[e] * E_dist[e, p].X for e in energy_carriers for p in ports)
        cost_lo_out    = sum(N_v[v] * C_LO[carrier_by_route[route_of[v]], v].X for v in vessels)

        # Sanity check: cost components must sum to objVal
        _check = cost_system_out + cost_plant_out + cost_port_out + cost_prod_out + cost_dist_out + cost_lo_out
        if abs(_check - model.objVal) / max(model.objVal, 1) > 1e-4:
            print(f"[WARNING] Cost breakdown sum {_check:,.0f} != objVal {model.objVal:,.0f} (diff {_check - model.objVal:+,.0f})")

        # Annual aggregated energy chain
        E_wake_tot = sum(N_v[v] * E_wake[carrier_by_route[route_of[v]], v].X for v in vessels)
        E_conv_tot = sum(N_v[v] * E_conv[carrier_by_route[route_of[v]], v].X for v in vessels)
        L_conv_tot = sum(N_v[v] * L_conv[carrier_by_route[route_of[v]], v].X for v in vessels)
        E_dist_tot = sum(E_dist[e, p].X for e in energy_carriers for p in ports)
        E_prod_tot = sum(E_prod[e].X for e in energy_carriers)
        L_dist_tot = sum(L_dist[e, p].X for e in energy_carriers for p in ports)
        L_prod_tot = sum(L_prod[e].X for e in energy_carriers)
        
        # Efficiencies
        wtw_eff  = E_wake_tot / E_prod_tot if E_prod_tot > 0 else 0
        eff_conv = E_wake_tot / E_conv_tot if E_conv_tot > 0 else 0
        eff_dist = E_conv_tot / E_dist_tot if E_dist_tot > 0 else 0
        eff_prod = E_dist_tot / E_prod_tot if E_prod_tot > 0 else 0

        # Transport work and energy intensity 
        vessel_row_map = df_vessels.set_index('vessel_id')
        total_transport_work = sum(N_v[v] * get_transport_work_effective(vessel_row_map.loc[v], 
                length_of[v], delta_V[carrier_by_route[route_of[v]], v].X) for v in vessels)
        max_transport_work = sum(N_v[v] * get_transport_work_effective(
                vessel_row_map.loc[v], length_of[v], 0) for v in vessels)
        remaining_fraction            = total_transport_work / max_transport_work if max_transport_work > 0 else 0
        cost_per_transport_work       = model.objVal / total_transport_work if total_transport_work > 0 else float('inf')
        cost_per_transport_work_gross = model.objVal / max_transport_work if max_transport_work > 0 else float('inf')
        energy_intensity = E_wake_tot / total_transport_work if total_transport_work > 0 else 0

        # Per vessel results
        onboard_capacity  = {v: q[carrier_by_route[route_of[v]], v].X for v in vessels}
        cargo_loss_volume = {v: delta_V[carrier_by_route[route_of[v]], v].X for v in vessels}
        cargo_loss_weight = {v: delta_W[carrier_by_route[route_of[v]], v].X for v in vessels}
        volume_share = {v: round(100 * (energy_volume_density[carrier_by_route[route_of[v]]] * q[carrier_by_route[route_of[v]], v].X + phi_volume[carrier_by_route[route_of[v]], v]) / volume_max[v], 1) for v in vessels}
        weight_share = {v: round(100 * (energy_weight_density[carrier_by_route[route_of[v]]] * q[carrier_by_route[route_of[v]], v].X + phi_weight[carrier_by_route[route_of[v]], v]) / weight_max[v], 1) for v in vessels}

        # Per-carrier energy chain (one entry per active carrier, used by Sankey)
        energy_chain_by_carrier = {}
        for e in energy_carriers:
            vv = [v for v in vessels if carrier_by_route[route_of[v]] == e]
            if not vv:
                continue
            ec_ew = sum(N_v[v] * E_wake[e, v].X for v in vv)
            ec_ec = sum(N_v[v] * E_conv[e, v].X for v in vv)
            ec_lc = sum(N_v[v] * L_conv[e, v].X for v in vv)
            ec_ed = sum(E_dist[e, p].X for p in ports)
            ec_ep = E_prod[e].X
            ec_ld = sum(L_dist[e, p].X for p in ports)
            ec_lp = L_prod[e].X
            n_vv  = len(vv)
            wtw_e = ec_ew / ec_ep if ec_ep > 0 else 0
            energy_chain_by_carrier[e] = {
                'E_wake': ec_ew, 'E_conv': ec_ec, 'L_conv': ec_lc,
                'E_dist': ec_ed, 'E_prod': ec_ep, 'L_dist': ec_ld, 'L_prod': ec_lp,
                'n_vessels': n_vv, 'wtw_efficiency': round(wtw_e, 3),
            }

        # WtW lifecycle emissions (post-processing, no effect on optimisation)
        # Basis: useful propulsion energy E_wake per carrier (MWh/year)
        emissions_by_carrier_tco2e = {
            e: chain['E_wake'] * WTW_EMISSION_FACTORS_KG_PER_MWH[e] / 1000
            for e, chain in energy_chain_by_carrier.items()
        }
        wtw_emissions_tco2e = sum(emissions_by_carrier_tco2e.values())

        if verbose:
            print_relevant_binding_constraints(model)
            print_relevant_lp_shadow_prices(model)

        # Per-route cost breakdown (system-optimal, shared costs allocated by energy flow)
        cost_by_route = {}
        for r in routes:
            c   = carrier_by_route[r]
            V_r = [v for v in vessels if route_of[v] == r]

            # Direct per-vessel costs
            cs_r  = sum(
                af_system[c] * cost_storage[c] * q[c, v].X
                + af_power_conversion_system[c] * cost_power_conversion[c] * (power_kW[v] / 1000)
                for v in V_r)
            clo_r = sum(N_v[v] * C_LO[c, v].X for v in V_r)

            # Shared plant + production: proportional to route share of carrier energy
            E_conv_r = sum(N_v[v] * E_conv[c, v].X for v in V_r)
            E_conv_c = sum(N_v[v] * E_conv[c, v].X for v in vessels if carrier_by_route[route_of[v]] == c)
            frac_c   = E_conv_r / E_conv_c if E_conv_c > 0 else 0.0
            cp_r    = af_plant * cost_plant[c] * P_plant[c].X * frac_c
            cprod_r = cost_production[c] * E_prod[c].X * frac_c

            # Port costs: allocate by each port's energy flow attributable to this route
            cport_r = 0.0
            cdist_r = 0.0
            for p in {origin_r[r], dest_r[r]}:
                z_total_p = sum(z[e, p, v].X for e in energy_carriers for v in vessels)
                z_r_p     = sum(z[c, p, v].X for v in V_r)
                frac_p    = z_r_p / z_total_p if z_total_p > 0 else 0.0
                cport_r  += af_port * cost_port[c] * S_port[c, p].X * frac_p
                cdist_r  += cost_dist[c] * E_dist[c, p].X * frac_p

            cost_by_route[r] = {
                'carrier':           c,
                'cost_system':       cs_r,
                'cost_plant':        cp_r,
                'cost_port':         cport_r,
                'cost_production':   cprod_r,
                'cost_distribution': cdist_r,
                'cost_lo':           clo_r,
                'total_cost':        cs_r + cp_r + cport_r + cprod_r + cdist_r + clo_r,
            }

        return {
                # General
                'carrier':              e_sel,
                'carrier_by_route':     carrier_by_route,
                'total_cost':           model.objVal,
                'charging_ports':       ports_with_infra,
                'n_routes':             len(routes),
                'n_vessels':            len(vessels),

                # Aggregated energy chain
                'E_wake_MWh':           E_wake_tot,
                'E_conv_MWh':           E_conv_tot,
                'E_dist_MWh':           E_dist_tot,
                'E_prod_MWh':           E_prod_tot,
                'L_conv_MWh':           L_conv_tot,
                'L_dist_MWh':           L_dist_tot,
                'L_prod_MWh':           L_prod_tot,
                'port_capacity_MW':    {p: {e: S_port[e, p].X for e in energy_carriers} for p in ports},
                'port_infra':          {p: {e: y[e, p].X > 0.5 for e in energy_carriers} for p in ports},

                # Efficiencies
                'eff_conversion':       round(eff_conv, 3),
                'eff_distribution':     round(eff_dist, 3),
                'eff_production':       round(eff_prod, 3),
                'wtw_efficiency':       round(wtw_eff, 3),

                # Transport work and energy intensity
                'total_transport_work':          total_transport_work,
                'max_transport_work':            max_transport_work,
                'remaining_fraction':            remaining_fraction,
                'cost_per_transport_work':       cost_per_transport_work,
                'cost_per_transport_work_gross': cost_per_transport_work_gross,
                'energy_intensity':              round(energy_intensity, 6),

                # Plant capacity
                'P_plant_MW':           {e: P_plant[e].X for e in energy_carriers if cost_plant[e] > 0},

                # Cost breakdown
                'cost_system':          cost_system_out,
                'cost_plant':           cost_plant_out,
                'cost_port':            cost_port_out,
                'cost_production':      cost_prod_out,
                'cost_distribution':    cost_dist_out,
                'cost_lo':              cost_lo_out,

                # Per-carrier energy chains (for Sankey)
                'energy_chain_by_carrier': energy_chain_by_carrier,

                # Per-route cost breakdown (system-optimal allocation)
                'cost_by_route':        cost_by_route,

                # WtW lifecycle emissions (post-processing)
                'wtw_emissions_tco2e':            wtw_emissions_tco2e,
                'wtw_emissions_by_carrier_tco2e': emissions_by_carrier_tco2e,

                # Per vessel
                'onboard_capacity':     onboard_capacity,
                'cargo_loss_volume':    cargo_loss_volume,
                'cargo_loss_weight':    cargo_loss_weight,
                'pct_volume_by_vessel': volume_share,
                'pct_weight_by_vessel': weight_share,
            }
    else:
        if verbose:
                print(f"[!] No optimal solution found (status={model.status})")
        return None

# =============================================================================
# SANKEY DIAGRAM - annual energy chain visualisation for a single model result
# =============================================================================

def plot_sankey(result, carrier=None, chain=None, title=None):
    e    = carrier or result['carrier']
    ch   = chain   or result['energy_chain_by_carrier'][e]
    E_p  = ch['E_prod']
    L_pr = ch['L_prod']
    E_d  = ch['E_dist']
    L_di = ch['L_dist']
    E_c  = ch['E_conv']
    L_co = ch['L_conv']
    E_w  = ch['E_wake']

    node_labels = [
        f'Primary energy<br>{E_p:.1f} MWh/year',
        f'Production loss<br>{L_pr:.1f} MWh/year',
        f'Distributed energy<br>{E_d:.1f} MWh/year',
        f'Distribution loss<br>{L_di:.1f} MWh/year',
        f'Onboard energy<br>{E_c:.1f} MWh/year',
        f'Conversion loss<br>{L_co:.1f} MWh/year',
        f'Useful propulsion<br>{E_w:.1f} MWh/year',
    ]

    node_colors = ['#2196F3', '#FF7043', '#5C6BC0', '#7E57C2', '#26A69A', '#FFA726', '#66BB6A']
    sources     = [0, 0, 2, 2, 4, 4]
    targets     = [1, 2, 3, 4, 5, 6]
    values      = [L_pr, E_d, L_di, E_c, L_co, E_w]
    link_colors = ['rgba(255,112,67,0.5)', 'rgba(92,107,192,0.5)', 'rgba(126,87,194,0.5)',
                   'rgba(38,166,154,0.5)', 'rgba(255,167,38,0.5)',  'rgba(102,187,106,0.5)',]
    link_labels = [
        f'Production loss: {L_pr:.1f} MWh ({100*L_pr/E_p:.1f}% of primary)',
        f'To distribution: {E_d:.1f} MWh',
        f'Distribution loss: {L_di:.1f} MWh ({100*L_di/E_p:.1f}% of primary)',
        f'To onboard: {E_c:.1f} MWh',
        f'Conversion loss: {L_co:.1f} MWh ({100*L_co/E_p:.1f}% of primary)',
        f'Useful propulsion: {E_w:.1f} MWh ({100*E_w/E_p:.1f}% of primary)',
    ]

    fig = go.Figure(go.Sankey(arrangement='snap',
        node=dict(pad=20, thickness=20, line=dict(color='white', width=0.5), label=node_labels, color=node_colors),
        link=dict(source=sources, target=targets, value=values, color=link_colors, label=link_labels)))
    wtw   = ch['wtw_efficiency']
    n_v   = ch['n_vessels']
    label = title or f"Annual energy chain - {e.capitalize()} | {n_v} vessels | WtW: {wtw:.1%}"
    fig.update_layout(title_text=label, title_x=0.5, font_size=13, height=450)
    fname = f"{OUT_DIR}sankey_{e}_{n_v}vessels.html"
    fig.write_html(fname)
    fig.show()
    print(f"  Sankey saved: {fname}")


# =============================================================================
# COLOURS
# =============================================================================

COLORS = {
    'battery':           '#2196F3',
    'hydrogen':          '#FF7043',
    'cost_system':       '#5C6BC0',
    'cost_plant':        '#7E57C2',
    'cost_port':         '#26A69A',
    'cost_production':   '#FFA726',
    'cost_distribution': '#66BB6A',
    'cost_lo':           '#EF5350',
}
plt.rcParams['axes.xmargin'] = 0
fmt_usd = mticker.FuncFormatter(lambda x, _: f'{x:,.0f}')

# =============================================================================
# BASE CASE
# =============================================================================

print('=' * 60)
print('BASE CASE')
print('=' * 60)

case     = solve_shipping_model(df_routes_raw, df_vessels)
case_bat = solve_shipping_model(df_routes_raw, df_vessels, force_carrier="battery",  verbose=False)
case_hyd = solve_shipping_model(df_routes_raw, df_vessels, force_carrier="hydrogen", verbose=False)
if case:
    _carriers_used = set(case['carrier_by_route'].values())
    if len(_carriers_used) == 1:
        _carrier_str = case['carrier'].capitalize()
    else:
        from collections import Counter
        _counts = Counter(case['carrier_by_route'].values())
        _carrier_str = 'Mixed (' + ', '.join(f'{e.capitalize()}: {n} route{"s" if n > 1 else ""}' for e, n in _counts.items()) + ')'
    print(f'  Carrier selected : {_carrier_str}')
    print(f'  Total cost       : {case["total_cost"]:,.0f} USD/year')
    print(f'  Plant capacity   : {case["P_plant_MW"]}')
    print(f'  Charging ports   : {case["charging_ports"]}')

    print(f'{"-"*60}')
    print(f'  Route carrier selection')
    print(f'  {"-"*60}')
    print(f'  {"Route":<6} {"Origin":<14} {"Destination":<14} {"Dist (nm)":>10} {"Carrier":>10}')
    print(f'  {"-"*60}')
    origin_r = dict(zip(df_routes_raw["route_id"], df_routes_raw["origin"]))
    dest_r   = dict(zip(df_routes_raw["route_id"], df_routes_raw["destination"]))
    length_r = dict(zip(df_routes_raw["route_id"], df_routes_raw["distance_nm"]))
    for r, carrier in case["carrier_by_route"].items():
        print(f'  {r:<6} {origin_r[r]:<14} {dest_r[r]:<14} {length_r[r]:>10.0f} {carrier.capitalize():>10}')

    print(f'\n  {"-"*60}')
    print(f'  Annual energy chain')
    print(f'  {"-"*60}')
    print(f'  {"Stage":<22} {"Energy in (MWh/yr)":>18} {"Loss (MWh/yr)":>14} {"Loss %":>8}')
    print(f'  {"-"*60}')
    E_p = case['E_prod_MWh']
    print(f'  {"Production":<22} {E_p:>18.0f} {case["L_prod_MWh"]:>14.0f} {100*case["L_prod_MWh"]/E_p:>7.1f}%')
    print(f'  {"Distribution":<22} {case["E_dist_MWh"]:>18.0f} {case["L_dist_MWh"]:>14.0f} {100*case["L_dist_MWh"]/E_p:>7.1f}%')
    print(f'  {"Conversion":<22} {case["E_conv_MWh"]:>18.0f} {case["L_conv_MWh"]:>14.0f} {100*case["L_conv_MWh"]/E_p:>7.1f}%')
    print(f'  {"Useful propulsion":<22} {case["E_wake_MWh"]:>18.0f} {"--":>14}')
    print(f'  {"-"*60}')
    print(f'  {"Well-to-wake eff.":<22} {case["wtw_efficiency"]:>41.1%}')

    print(f'\n  {"-"*60}')
    print(f'  Per vessel onboard capacity')
    print(f'  {"-"*60}')
    for v, cap in case['onboard_capacity'].items():
        pv = case['pct_volume_by_vessel'][v]
        pw = case['pct_weight_by_vessel'][v]
        print(f'  {v:<20} {cap:>8.1f} MWh   vol:{pv:>5.1f}%  wgt:{pw:>5.1f}%')

    print(f'\n  {"-"*60}')
    print(f'  Port throughput capacity (MW)')
    print(f'  {"-"*60}')
    for port, caps in case['port_capacity_MW'].items():
        for e, cap in caps.items():
            if cap > 0.001:
                print(f'  Port {port:<18} [{e:<9}] {cap:>8.1f} MW')

    print(f'{"-"*60}')
    print(f'  Port infrastructure')
    print(f'  {"-"*60}')
    for port, carriers in case['port_infra'].items():
        built = [e.capitalize() for e, is_built in carriers.items() if is_built]
        status = ', '.join(built) if built else '--'
        print(f'  Port {port:<18} {status}')

    print(f'\n  {"-"*60}')
    print(f'  Transport work')
    print(f'  {"-"*60}')
    print(f'  Remaining cargo fraction : {case["remaining_fraction"]:.2%}')
    print(f'  Transport work (actual)  : {case["total_transport_work"]:,.0f} unit·nm/year')
    print(f'  Transport work (max)     : {case["max_transport_work"]:,.0f} unit·nm/year')
    print(f'  Cost per transport work  : {case["cost_per_transport_work"]:.4f} USD/unit·nm')

    print(f'{"-"*60}')
    print(f'  Annual cost breakdown (USD/year)')
    print(f'  {"-"*60}')
    for lbl_key, lbl in [
        ('cost_system',       'Onboard system'),
        ('cost_plant',        'Plant investment'),
        ('cost_port',         'Port infrastructure'),
        ('cost_production',   'Production'),
        ('cost_distribution', 'Distribution'),
        ('cost_lo',           'Lost opportunity'),
    ]:
        share = 100 * case[lbl_key] / case['total_cost']
        print(f'  {lbl:<22} {case[lbl_key]:>12,.0f}   ({share:.1f}%)')
    print(f'  {"-"*60}')
    print(f'  {"TOTAL":<22} {case["total_cost"]:>12,.0f}')

    print(f'{"-"*60}')
    print(f'  WtW lifecycle emissions  (basis: useful propulsion energy, E_wake)')
    print(f'  {"-"*60}')
    print(f'  Emission factors:  Battery {WTW_EMISSION_FACTORS_KG_PER_MWH["battery"]:.0f} kg CO2e/MWh  |  '
          f'Hydrogen {WTW_EMISSION_FACTORS_KG_PER_MWH["hydrogen"]:.0f} kg CO2e/MWh  (per MWh of E_wake)')
    print(f'  {"Scenario":<30} {"tCO2e/year":>14}')
    print(f'  {"-"*60}')
    for _lbl, _res in [
        ('Optimal solution',  case),
        ('Forced battery',    case_bat),
        ('Forced hydrogen',   case_hyd),
    ]:
        if _res:
            _em = _res['wtw_emissions_tco2e']
            print(f'  {_lbl:<30} {_em:>14,.1f}')
            for _e, _em_e in _res['wtw_emissions_by_carrier_tco2e'].items():
                print(f'    {_e.capitalize():<28} {_em_e:>14,.1f}')
    print(f'  {"-"*60}')
    print(f'  WtW emissions (tCO2e/year) : {case["wtw_emissions_tco2e"]:,.1f}')

    for e, chain in case['energy_chain_by_carrier'].items():
        plot_sankey(case, carrier=e, chain=chain)

    # -- R3: Per-route cost structure (100% stacked) ---------------------------
    route_cost_rows = []
    _rdist_map = dict(zip(df_routes_raw['route_id'], df_routes_raw['distance_nm']))
    _rorig_map = dict(zip(df_routes_raw['route_id'], df_routes_raw['origin'].astype(str).str.strip()))
    _rdest_map = dict(zip(df_routes_raw['route_id'], df_routes_raw['destination'].astype(str).str.strip()))
    for rid, rc in case['cost_by_route'].items():
        rdist = _rdist_map.get(rid, 0)
        rorig = _rorig_map.get(rid, rid)
        rdest = _rdest_map.get(rid, rid)
        c_sel = rc['carrier']
        total = rc['total_cost']
        if total <= 0:
            continue
        route_cost_rows.append({
            'label':             f"{rid.replace('R_', 'Route ')}, {rdist:.0f} nm: {c_sel.capitalize()}",
            'distance':          rdist,
            'cost_system':       100 * rc['cost_system']       / total,
            'cost_plant':        100 * rc['cost_plant']        / total,
            'cost_port':         100 * rc['cost_port']         / total,
            'cost_production':   100 * rc['cost_production']   / total,
            'cost_distribution': 100 * rc['cost_distribution'] / total,
            'cost_lo':           100 * rc['cost_lo']           / total,
        })
    route_cost_rows.sort(key=lambda d: d['distance'])
    cost_components_r3 = [
        ('cost_system',       'Onboard system'),
        ('cost_plant',        'Plant invest.'),
        ('cost_port',         'Port infra.'),
        ('cost_production',   'Production'),
        ('cost_distribution', 'Distribution'),
        ('cost_lo',           'Lost opportunity'),
    ]
    active_r3 = [(k, lbl) for k, lbl in cost_components_r3
                 if any(row[k] > 0.5 for row in route_cost_rows)]
    bar_labels = [row['label'] for row in route_cost_rows]
    fig_r3, ax_r3 = plt.subplots(figsize=(10, max(4, len(route_cost_rows) * 1.0)))
    lefts = [0.0] * len(route_cost_rows)
    for key, lbl in active_r3:
        max_pct  = max(row[key] for row in route_cost_rows)
        bar_lbl  = lbl if max_pct >= 2.0 else '_nolegend_'
        vals = [row[key] for row in route_cost_rows]
        ax_r3.barh(bar_labels, vals, left=lefts, color=COLORS[key], label=bar_lbl)
        for i, (val, lft) in enumerate(zip(vals, lefts)):
            if val > 5:
                ax_r3.text(lft + val / 2, i, f'{val:.0f}%', ha='center', va='center',
                           fontsize=8, color='white', fontweight='bold')
        lefts = [l + v for l, v in zip(lefts, vals)]
    ax_r3.set_xlabel('Share of total annual route cost (%)')
    ax_r3.set_xlim(0, 100)
    ax_r3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                 ncol=3, fontsize=8, framealpha=0.8)
    plt.tight_layout(rect=[0, 0.14, 1, 1])
    r3_png = f'{OUT_DIR}cost_structure_by_route.png'
    plt.savefig(r3_png, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Per-route cost structure saved: {r3_png}')

# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

# ─── S1: ELECTRICITY PRICE SWEEP ─────────────────────────────────────────────

print()
print('=' * 60)
print('SENSITIVITY 1: ELECTRICITY PRICE')
print('=' * 60)

elec_prices = np.linspace(20, 250, 46)
s1_bat, s1_hyd, s1_free = [], [], []

for _P in elec_prices:
    _params = {'cost_production': {'battery': _P, 'hydrogen': _P}}
    _rb = solve_shipping_model(df_routes_raw, df_vessels, params=_params, force_carrier='battery',  verbose=False)
    _rh = solve_shipping_model(df_routes_raw, df_vessels, params=_params, force_carrier='hydrogen', verbose=False)
    _rf = solve_shipping_model(df_routes_raw, df_vessels, params=_params, verbose=False)
    s1_bat.append(_rb['total_cost']  if _rb  else None)
    s1_hyd.append(_rh['total_cost']  if _rh  else None)
    s1_free.append(_rf['total_cost'] if _rf  else None)

_diffs1 = [h - b if (h and b) else None for b, h in zip(s1_bat, s1_hyd)]
_cross1 = None
for _i in range(len(_diffs1) - 1):
    if _diffs1[_i] is not None and _diffs1[_i + 1] is not None:
        if _diffs1[_i] * _diffs1[_i + 1] < 0:
            _cross1 = elec_prices[_i] + (elec_prices[_i + 1] - elec_prices[_i]) * abs(_diffs1[_i]) / (abs(_diffs1[_i]) + abs(_diffs1[_i + 1]))
            break
if _cross1:
    print(f'  Fleet crossover electricity price: {_cross1:.1f} USD/MWh')

fig_s1, ax_s1 = plt.subplots(figsize=(9, 5))
ax_s1.plot(elec_prices, [v / 1e6 if v else None for v in s1_bat],  color=COLORS['battery'],  lw=2, label='Battery (forced)')
ax_s1.plot(elec_prices, [v / 1e6 if v else None for v in s1_hyd],  color=COLORS['hydrogen'], lw=2, label='Hydrogen (forced)')
ax_s1.plot(elec_prices, [v / 1e6 if v else None for v in s1_free], color='#4CAF50', lw=2, ls='--', label='Optimal mix (free)')
ax_s1.axvline(DEFAULT_PARAMS['cost_production']['battery'], color='grey', ls=':', lw=1.5, label='Base case (120 USD/MWh)')
if _cross1:
    ax_s1.axvline(_cross1, color='black', ls='--', lw=1, alpha=0.7, label=f'Crossover ({_cross1:.0f} USD/MWh)')
    _cross1_y = np.interp(_cross1, elec_prices, [v / 1e6 if v else 0 for v in s1_bat])
    ax_s1.scatter(_cross1, _cross1_y, color="black", s=30, zorder=5)
    ax_s1.text(_cross1 + 2, _cross1_y * 1.05, f"{_cross1:.0f} USD/MWh", va="bottom", ha="left", fontsize=8, color="gray")
ax_s1.set_xlabel('Electricity price (USD/MWh)')
ax_s1.set_ylabel('Total annual system cost (M USD/year)')
ax_s1.legend(fontsize=9)
ax_s1.grid(linestyle="--", alpha=0.5)
ax_s1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
plt.tight_layout()
_s1_path = f'{OUT_DIR}sensitivity_electricity_price.png'
plt.savefig(_s1_path, dpi=150, bbox_inches='tight')
print(f'  Saved: {_s1_path}')


# ─── S2: INFRASTRUCTURE COST SCALE ───────────────────────────────────────────

print()
print('=' * 60)
print('SENSITIVITY 2: INFRASTRUCTURE COST SCALE')
print('=' * 60)

infra_scales = np.linspace(0.25, 4.0, 46)
s2_bat, s2_hyd, s2_free = [], [], []

for _s in infra_scales:
    _params = {
        'cost_port':  {'battery': DEFAULT_PARAMS['cost_port']['battery']  * _s,
                        'hydrogen': DEFAULT_PARAMS['cost_port']['hydrogen'] * _s},
        'cost_plant': {'battery': DEFAULT_PARAMS['cost_plant']['battery'],
                        'hydrogen': DEFAULT_PARAMS['cost_plant']['hydrogen'] * _s},
    }
    _rb = solve_shipping_model(df_routes_raw, df_vessels, params=_params, force_carrier='battery',  verbose=False)
    _rh = solve_shipping_model(df_routes_raw, df_vessels, params=_params, force_carrier='hydrogen', verbose=False)
    _rf = solve_shipping_model(df_routes_raw, df_vessels, params=_params, verbose=False)
    s2_bat.append(_rb['total_cost']  if _rb  else None)
    s2_hyd.append(_rh['total_cost']  if _rh  else None)
    s2_free.append(_rf['total_cost'] if _rf  else None)

_diffs2 = [h - b if (h and b) else None for b, h in zip(s2_bat, s2_hyd)]
_cross2 = None
for _i in range(len(_diffs2) - 1):
    if _diffs2[_i] is not None and _diffs2[_i + 1] is not None:
        if _diffs2[_i] * _diffs2[_i + 1] < 0:
            _cross2 = infra_scales[_i] + (infra_scales[_i + 1] - infra_scales[_i]) * abs(_diffs2[_i]) / (abs(_diffs2[_i]) + abs(_diffs2[_i + 1]))
            break
if _cross2:
    print(f'  Crossover infrastructure scale: {_cross2:.2f}x')

fig_s2, ax_s2 = plt.subplots(figsize=(9, 5))
ax_s2.plot(infra_scales, [v / 1e6 if v else None for v in s2_bat],  color=COLORS['battery'],  lw=2, label='Battery (forced)')
ax_s2.plot(infra_scales, [v / 1e6 if v else None for v in s2_hyd],  color=COLORS['hydrogen'], lw=2, label='Hydrogen (forced)')
ax_s2.plot(infra_scales, [v / 1e6 if v else None for v in s2_free], color='#4CAF50', lw=2, ls='--', label='Optimal mix (free)')
ax_s2.axvline(1.0, color='grey', ls=':', lw=1.5, label='Base case (scale = 1.0)')
if _cross2:
    ax_s2.axvline(_cross2, color='black', ls='--', lw=1, alpha=0.7, label=f'Crossover ({_cross2:.2f}x)')
    _cross2_y = np.interp(_cross2, infra_scales, [v / 1e6 if v else 0 for v in s2_bat])
    ax_s2.scatter(_cross2, _cross2_y, color="black", s=30, zorder=5)
    ax_s2.text(_cross2 + 0.05, _cross2_y * 1.05, f"{_cross2:.2f}x", va="bottom", ha="left", fontsize=8, color="gray")
ax_s2.set_xlabel('Infrastructure cost scale factor')
ax_s2.set_ylabel('Total annual system cost (M USD/year)')
ax_s2.legend(fontsize=9)
ax_s2.grid(linestyle="--", alpha=0.5)
ax_s2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
plt.tight_layout()
_s2_path = f'{OUT_DIR}sensitivity_infrastructure_cost.png'
plt.savefig(_s2_path, dpi=150, bbox_inches='tight')
print(f'  Saved: {_s2_path}')


# ─── S3: PER-ROUTE LENGTH SENSITIVITY ────────────────────────────────────────

print()
print('=' * 60)
print('SENSITIVITY 3: PER-ROUTE LENGTH')
print('=' * 60)

_length_nm   = np.arange(50, 801, 10)
_route_ids   = df_routes_raw['route_id'].tolist()
_origins_map = dict(zip(df_routes_raw['route_id'], df_routes_raw['origin']))
_dests_map   = dict(zip(df_routes_raw['route_id'], df_routes_raw['destination']))
_base_len    = dict(zip(df_routes_raw['route_id'], df_routes_raw['distance_nm']))
s3 = {}

for _rid in _route_ids:
    _df_r = df_routes_raw[df_routes_raw['route_id'] == _rid].copy()
    _df_v = df_vessels[df_vessels['route_id'] == _rid].copy()
    _cb, _ch = [], []
    for _L in _length_nm:
        _df_rm = _df_r.copy()
        _df_rm['distance_nm'] = _L
        _rb = solve_shipping_model(_df_rm, _df_v, force_carrier='battery',  verbose=False)
        _rh = solve_shipping_model(_df_rm, _df_v, force_carrier='hydrogen', verbose=False)
        _cb.append(_rb['total_cost'] if _rb else None)
        _ch.append(_rh['total_cost'] if _rh else None)
    _diffs = [h - b if (h and b) else None for b, h in zip(_cb, _ch)]
    _cross = None
    for _i in range(len(_diffs) - 1):
        if _diffs[_i] is not None and _diffs[_i + 1] is not None:
            if _diffs[_i] * _diffs[_i + 1] < 0:
                _cross = _length_nm[_i] + (_length_nm[_i + 1] - _length_nm[_i]) * abs(_diffs[_i]) / (abs(_diffs[_i]) + abs(_diffs[_i + 1]))
                break
    s3[_rid] = {'diff': _diffs, 'crossover_nm': _cross}
    _xstr = f'{_cross:.0f} nm' if _cross else '> 800 nm'
    print(f'  {_rid:6s}  {str(_origins_map[_rid]).strip():<14} - {str(_dests_map[_rid]).strip():<14}  base: {_base_len[_rid]:>6.0f} nm   crossover: {_xstr}')

# Colour palette
_palette = ['#0077BB', '#CC3311', '#009988', '#EE7733', '#AA3377']
_linestyles = ['-', '--', '-', '-', '-']
fig_s3, ax_s3 = plt.subplots(figsize=(10, 5))
ax_s3.axhline(0, color='black', lw=1.5, zorder=3)

# Plot all curves first so ylim is data-driven (R_2 last so dashes stay on top)
_plot_order = [_i for _i, _rid in enumerate(_route_ids) if _rid != "R_2"] + [_i for _i, _rid in enumerate(_route_ids) if _rid == "R_2"]
for _i in _plot_order:
    _rid = _route_ids[_i]
    _label = f'{_rid}: {str(_origins_map[_rid]).strip()}-{str(_dests_map[_rid]).strip()}'
    _dk_arr = np.array([v / 1e3 if v else np.nan for v in s3[_rid]["diff"]])
    ax_s3.plot(_length_nm, _dk_arr, color=_palette[_i], lw=2, ls=_linestyles[_i], label=_label)

# After curves are drawn, get ylim and add crossover drop-lines
ylo, yhi = ax_s3.get_ylim()
for _i, _rid in enumerate(_route_ids):
    _cx = s3[_rid]['crossover_nm']
    if _cx and _length_nm[0] <= _cx <= _length_nm[-1]:
        ax_s3.vlines(_cx, ymin=ylo, ymax=0,
                     color=_palette[_i], linestyle=':', linewidth=1.0, alpha=0.9, zorder=2)

_cx_labels = []
for _i, _rid in enumerate(_route_ids):
    _cx = s3[_rid]['crossover_nm']
    if _cx and _length_nm[0] <= _cx <= _length_nm[-1]:
        _cx_labels.append((_cx, _i, _rid))
_cx_labels.sort(key=lambda a: a[0])
_y0 = ylo + (0 - ylo) * 0.03
_dy = (0 - ylo) * 0.09
_placed = {0: [], 1: [], 2: []}
for _cx, _i, _rid in _cx_labels:
    for _lvl in range(3):
        if all(abs(_cx - _px) >= 30 for _px in _placed[_lvl]):
            _placed[_lvl].append(_cx)
            _y = _y0 + _lvl * _dy
            break
    else:
        _y = _y0
    ax_s3.text(_cx, _y, f'{_cx:.0f} nm', ha='center', va='bottom', fontsize=7, color=_palette[_i])
ax_s3.set_ylim(ylo, yhi)
ax_s3.text(57, (0 - ylo) * 0.06,  'Battery cheaper ↑', va='bottom', ha='left', fontsize=7, color='gray')
ax_s3.text(57, -(0 - ylo) * 0.06, 'Hydrogen cheaper ↓', va='top',   ha='left', fontsize=7, color='gray')
ax_s3.set_xlabel('Route length (nm)')
ax_s3.set_ylabel('Cost difference: $C_{H_2} - C_{BE}$ (kUSD/year)')
ax_s3.legend(fontsize=8, loc='upper right')
ax_s3.grid(linestyle="--", alpha=0.5)
plt.tight_layout()
_s3_path = f'{OUT_DIR}sensitivity_per_route_length.png'
plt.savefig(_s3_path, dpi=150, bbox_inches='tight')
print(f'  Saved: {_s3_path}')