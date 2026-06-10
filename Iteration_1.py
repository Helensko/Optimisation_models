import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import plotly.graph_objects as go
import os

# =============================================================================
# OUTPUT FOLDER
# =============================================================================

OUT_DIR = 'results/iteration_1/'
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# DEFAULT PARAMETERS
# Override any of these by passing a dict to solve_shipping_model()
# =============================================================================

DEFAULT_PARAMS = {
    # Route
    'energy_per_nm':        0.17,      # [MWh/nm]
    'power_kw':             2700,      # [kW] - from the energy model using MariTeam
    'lane_meters':          500,       # [lm]
    'vessel_type':          'roro',    # Used for transport work calculations
    'lay_time_port':        6,         # [h]

    # Lost opportunity cost - max(volume penalty, weight penalty) per vessel
    'cost_LO_V':            15,        # [USD/m³]
    'cost_LO_W':            20,        # [USD/tonne]

    # Vessel capacity
    'volume_available':     1_261,     # [m³]  – freed from conventional system (0.70 × Watson engine room)
    'weight_available':     328,       # [t]   – freed from conventional system (Watson machinery weight)
    'volume_max':           5_000,     # [m³]  – total cargo volume (100% hard cap)
    'weight_max':           4_894,     # [t]   – total cargo weight (100% hard cap)

    # Annuity / allocation
    'N':                    104,       # voyages per year
    'r_port':               0.05,
    'T_port':               40,

    'r_plant':              0.05,
    'T_plant':              17,

    'T_system': {'battery': 10,   'hydrogen': 15},
    'r_system': {'battery': 0.05, 'hydrogen': 0.05},

    'T_power_conversion_system': {'battery': 10, 'hydrogen': 6},        # Dummy value for battery to avoid division by 0 issues (cost_power_conversion = 0)
    'r_power_conversion_system': {'battery': 0.05, 'hydrogen': 0.05},

    # Big-M
    'M':                    10_000_000,

    # Efficiency factors along the energy chain
    'eta_production':       {'battery': 1, 'hydrogen': 0.6},
    'eta_distribution':     {'battery': 0.85, 'hydrogen': 0.98},
    'eta_conversion':       {'battery': 0.95, 'hydrogen': 0.52},     

    # Investment costs
    'plant_full_load_hours': {'battery': 8760, 'hydrogen': 4000},

    'cost_plant_power': {
    'battery':  0,
    'hydrogen': 1_424_000  # USD/MW 
    },

    'cost_storage': {
    'battery':  434_498,   # USD/MWh
    'hydrogen': 12_700,    # USD/MWh
    },

    # PEMFC (power related) onboard system cost
    'cost_power_conversion': {
    'battery': 0,          # USD/MW, or include battery converters later
    'hydrogen': 1_014_000, # USD/MW
    },

    'usable_capacity_fraction': {
    'battery': 0.8,
    'hydrogen': 1.0,
    },

    # Physical storage requirements (tanks/packs only - PEMFC contribution added inside model)
    'energy_volume_density':  {'battery': 11.7, 'hydrogen': 1.16},  # [m³/MWh]
    'energy_weight_density':  {'battery': 15.4, 'hydrogen': 0.58},  # [tonne/MWh]

    # PEMFC power density (hydrogen system only)
    'pemfc_volume_density': 0.2,   # [MW/m³]
    'pemfc_weight_density': 0.4,   # [MW/t]

    # Production cost (per MWh of primary energy produced)
    'cost_production':  {'battery': 120,  'hydrogen': 120},     # [USD/MWh]

    # Distribution cost per MWh delivered to each port
    'cost_distribution': {
        'battery':  0,    # [USD/MWh entering distribution]
        'hydrogen': 16.6,   # [USD/MWh entering distribution] 
    },
 
    # Port infrastructure investment cost
    'cost_port':        {'battery': 255_170, 'hydrogen': 356_000},      # [USD/MW]
}

# =============================================================================
# HELP FUNCTION
# =============================================================================

def print_relevant_binding_constraints(model, tol=1e-6):
    relevant_keywords = [
        'physical',
        'LO',
        'C_LO',
        'onboard_energy_capacity',
        'vessel_port',
        'both_ports',
        'zero_capacity',
        'production_plant_power'
    ]

    print("\nRelevant binding constraints:")
    for c in model.getConstrs():
        if any(k in c.ConstrName for k in relevant_keywords):
            if abs(c.Slack) <= tol:
                print(f"{c.ConstrName:45s} Slack={c.Slack:.4g} RHS={c.RHS:.4g}")

def print_relevant_lp_shadow_prices(model, tol=1e-6):
    relevant_keywords = [
        'physical',
        'LO',
        'C_LO',
        'onboard_energy_capacity',
        'vessel_port',
        'both_ports',
        'production_plant_power'
    ]

    relaxed = model.relax()
    relaxed.setParam('OutputFlag', 0)
    relaxed.optimize()

    if relaxed.status == GRB.OPTIMAL:
        print("\nRelevant shadow prices from LP relaxation:")
        for c in relaxed.getConstrs():
            if any(k in c.ConstrName for k in relevant_keywords):
                if abs(c.Pi) > tol:
                    print(f"{c.ConstrName:45s} Pi={c.Pi:.4g} Slack={c.Slack:.4g}")

def calculate_converter_footprint(power_kw, pemfc_volume_density, pemfc_weight_density):
    p_conv = power_kw / 1000

    phi_volume = p_conv / pemfc_volume_density
    phi_weight = p_conv / pemfc_weight_density

    return phi_volume, phi_weight

# =============================================================================
# MODEL FUNCTION
# =============================================================================

def solve_shipping_model(length, params=None, force_carrier=None, verbose=False):
    """
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
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    ports                 = ['A', 'B']
    energy_carriers       = ['battery', 'hydrogen']
    energy_per_nm         = p['energy_per_nm']
    cost_LO_V             = p['cost_LO_V']
    cost_LO_W             = p['cost_LO_W']
    volume_available      = p['volume_available']
    weight_available      = p['weight_available']
    volume_max            = p['volume_max']
    weight_max            = p['weight_max']
    N                     = p['N']
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
    plant_full_load_hours = p['plant_full_load_hours']
    cost_plant            = p['cost_plant_power']
    cost_production       = p['cost_production']
    cost_distribution     = p['cost_distribution']
    cost_port             = p['cost_port']
    F                     = p['usable_capacity_fraction']
    lay_time_port         = p['lay_time_port']

    af_port   = p['r_port']  / (1 - (1 + p['r_port'])**(-p['T_port']))
    af_plant  = p['r_plant'] / (1 - (1 + p['r_plant'])**(-p['T_plant']))
    af_system = {e: p['r_system'][e] / (1 - (1 + p['r_system'][e])**(-p['T_system'][e]))
             for e in energy_carriers}
    af_power_conversion_system = {e: p['r_power_conversion_system'][e] / (1 - (1 + p['r_power_conversion_system'][e])**(-p['T_power_conversion_system'][e]))
             for e in energy_carriers}

    phi_volume = {}
    phi_weight = {}

    for e in energy_carriers:

        if e == 'hydrogen':
            phi_volume[e], phi_weight[e] = calculate_converter_footprint(
                p['power_kw'], pemfc_volume_density, pemfc_weight_density)
        else:
            phi_volume[e] = 0.0
            phi_weight[e] = 0.0

    model = gp.Model('Maritime_Optimization')
    model.setParam('OutputFlag', 0)

    # Decision variables
    x       = model.addVars(energy_carriers,        vtype=GRB.BINARY,     name='x')
    y       = model.addVars(energy_carriers, ports, vtype=GRB.BINARY,     name='y')
    q       = model.addVars(energy_carriers,        vtype=GRB.CONTINUOUS, name='q')
    z       = model.addVars(energy_carriers, ports, vtype=GRB.CONTINUOUS, name='z')
    w       = model.addVars(energy_carriers,        vtype=GRB.BINARY,     name='w')
    delta_V = model.addVars(energy_carriers,        vtype=GRB.CONTINUOUS, name='delta_V')
    delta_W = model.addVars(energy_carriers,        vtype=GRB.CONTINUOUS, name='delta_W')
    C_LO    = model.addVars(energy_carriers,        vtype=GRB.CONTINUOUS, name='C_LO')
    P_plant = model.addVars(energy_carriers,        vtype=GRB.CONTINUOUS, name='P_plant')
    S_port  = model.addVars(energy_carriers, ports, vtype=GRB.CONTINUOUS, name='S_port')

    # Energy chain variables
    E_wake  = model.addVars(energy_carriers,        vtype=GRB.CONTINUOUS, name='E_wake')
    E_conv  = model.addVars(energy_carriers,        vtype=GRB.CONTINUOUS, name='E_conv')
    E_dist  = model.addVars(energy_carriers, ports, vtype=GRB.CONTINUOUS, name='E_dist')
    E_prod  = model.addVars(energy_carriers,        vtype=GRB.CONTINUOUS, name='E_prod')
    L_prod  = model.addVars(energy_carriers,        vtype=GRB.CONTINUOUS, name='L_prod')
    L_dist  = model.addVars(energy_carriers, ports, vtype=GRB.CONTINUOUS, name='L_dist')
    L_conv  = model.addVars(energy_carriers,        vtype=GRB.CONTINUOUS, name='L_conv')

    # Objective function
    
    model.setObjective(
        gp.quicksum(af_system[e] * cost_storage[e] * q[e]
                    + af_power_conversion_system[e] * cost_power_conversion[e] * (p['power_kw'] / 1000) * x[e]
                      for e in energy_carriers)
        + gp.quicksum(af_plant * cost_plant[e] * P_plant[e]
                      for e in energy_carriers)
        + gp.quicksum(af_port * cost_port[e] * S_port[e, p]
                      for e in energy_carriers for p in ports)
        + gp.quicksum(cost_production[e] * E_prod[e]
                      for e in energy_carriers)
        + gp.quicksum(cost_distribution[e] * E_dist[e, p]
                      for e in energy_carriers for p in ports)
        + gp.quicksum(C_LO[e] * N for e in energy_carriers),
        GRB.MINIMIZE
    )

    # (1) Exactly one energy carrier
    # model.addConstr(x.sum() == 1, name='carrier_selection')
    if force_carrier:
        model.addConstr(x[force_carrier] == 1, name='carrier_forced')
        for e in energy_carriers:
            if e != force_carrier:
                model.addConstr(x[e] == 0, name=f'carrier_blocked_{e}')
    else:
        model.addConstr(x.sum() == 1, name='carrier_selection')
    
    # (2) Wake energy demand (round trip)
    model.addConstrs(
        (E_wake[e] == 2 * length * energy_per_nm * x[e]
         for e in energy_carriers),
        name='wake_energy'
    )

    # (3) Conversion: wake -> onboard energy needed (accounts for drivetrain loss)
    model.addConstrs(
        (E_conv[e] == E_wake[e] / eta_conv[e]
         for e in energy_carriers),
         name='conversion_energy'
    )

    # (4) Distribution: energy needed at port after distribution losses
    model.addConstrs(
    (E_dist[e, p] == z[e, p] / eta_dist[e] 
     for e in energy_carriers for p in ports), 
     name='distribution_energy'
     )

    # (5) Production: primary energy required including production losses
    model.addConstrs(
    (E_prod[e] == gp.quicksum(E_dist[e, p] for p in ports) / eta_prod[e]
        for e in energy_carriers),
        name='production_energy'
    )

    # (6) Power for production plant
    model.addConstrs(
    (P_plant[e] == E_prod[e] / plant_full_load_hours[e]
     for e in energy_carriers),
    name='production_plant_power'
)

    # (7) Total energy delivered at ports = energy needed onboard (after conv) (annual)
    model.addConstrs(
        (gp.quicksum(z[e, p] for p in ports) == N * E_conv[e]
         for e in energy_carriers),
        name='energy_balance'
    )
    
    # (8) Port capacity
    model.addConstrs(
        (S_port[e, p] >= z[e, p] / (N * lay_time_port)
         for e in energy_carriers for p in ports),
         name='port_capacity'
    )

    # (9–11) Energy losses at each stage
    model.addConstrs(
        (L_conv[e] == E_conv[e] - E_wake[e]
         for e in energy_carriers),
         name='conversion_loss'
    )

    model.addConstrs(
        (L_dist[e, p] == E_dist[e, p] - z[e, p]
         for e in energy_carriers for p in ports),
        name='distribution_loss'
    )
    model.addConstrs(
        (L_prod[e] == E_prod[e] - gp.quicksum(E_dist[e, p] for p in ports)
         for e in energy_carriers),
        name='production_loss'
    )

    # (12) Port capacity linked to infrastructure
    model.addConstrs(
        (S_port[e, p] <= M * y[e, p] for e in energy_carriers for p in ports),
        name='port_capacity_infrastructure'
    )

    # (13–14) Vessel–port coupling
    model.addConstrs(
        (z[e, p] <= M * y[e, p] for e in energy_carriers for p in ports),
        name='vessel_port_coupling_1'
    )
    model.addConstrs(
        (y[e, p] <= x[e] for e in energy_carriers for p in ports),
        name='vessel_port_coupling_2'
    )

    # Per-voyage charging at any port cannot exceed onboard tank capacity
    model.addConstrs(
        (z[e, p] <= N * q[e] for e in energy_carriers for p in ports),
        name='charging_tank_limit'
    )

    # (15–16) w_e = 1 iff infrastructure in both ports
    model.addConstrs(
        (w[e] <= y[e, p] for e in energy_carriers for p in ports),
        name='both_ports_indicator_1'
    )
    
    model.addConstrs(
        (w[e] >= y.sum(e, '*') - 1 for e in energy_carriers),
        name='both_ports_indicator_2'
    )

    # (17) Onboard energy capacity (one leg if w=1, full round trip if w=0)
    model.addConstrs(
        (q[e] == (2 * x[e] - w[e]) * (length * energy_per_nm / (eta_conv[e] * F[e]))
         for e in energy_carriers),
        name='onboard_energy_capacity'
    )

    # (18) No capacity for unselected carrier
    model.addConstrs(
        (q[e] <= M * x[e] for e in energy_carriers),
        name='zero_capacity_unselected'
    )

    # (19-20) Hard cap: energy storage cannot physically exceed available space
    model.addConstrs(
        (energy_volume_density[e] * q[e] + phi_volume[e] * x[e] <= volume_max * x[e]
        for e in energy_carriers),
        name='physical_volume_cap'
    )
    model.addConstrs(
        (energy_weight_density[e] * q[e] + phi_weight[e] * x[e] <= weight_max * x[e]
        for e in energy_carriers),
        name='physical_weight_cap'
    )
    # (21–22) Hard physical caps: delta cannot exceed total cargo space
    model.addConstrs(
        (delta_V[e] <= volume_max * x[e] for e in energy_carriers),
        name='LO_V_limit'
    )
    model.addConstrs(
        (delta_W[e] <= weight_max * x[e] for e in energy_carriers),
        name='LO_W_limit'
    )

    # (23-24) Lost cargo = storage footprint minus freed space from removed system
    model.addConstrs(
        (delta_V[e] >= energy_volume_density[e] * q[e] 
                    + phi_volume[e] * x[e] 
                    - volume_available * x[e]
        for e in energy_carriers),
        name='LO_V_kick_in'
    )
    model.addConstrs(
        (delta_W[e] >= energy_weight_density[e] * q[e]
                    + phi_weight[e] * x[e]
                    - weight_available * x[e]
        for e in energy_carriers),
        name='LO_W_kick_in'
    )

    # (25-26) C_LO = max(volume penalty, weight penalty)
    model.addConstrs(
        (C_LO[e] >= cost_LO_V * delta_V[e] for e in energy_carriers),
        name='C_LO_volume'
    )
    model.addConstrs(
        (C_LO[e] >= cost_LO_W * delta_W[e] for e in energy_carriers),
        name='C_LO_weight'
    )

    model.optimize()

    if model.status == GRB.OPTIMAL:
        e_sel = [e for e in energy_carriers if x[e].X > 0.5][0]
        ports_with_infra = int(sum(y[e_sel, p].X for p in ports))

        # Cost decomposition
        cost_system_out = (af_system[e_sel] * cost_storage[e_sel] * q[e_sel].X
                        + af_power_conversion_system[e_sel] * cost_power_conversion[e_sel] * (p['power_kw'] / 1000))
        cost_plant_out  = af_plant * (cost_plant[e_sel] * P_plant[e_sel].X)
        cost_port_out   = sum(af_port * (cost_port[e_sel] * S_port[e_sel, port].X) for port in ports)
        cost_prod_out   = cost_production[e_sel] * E_prod[e_sel].X
        cost_dist_out   = sum(cost_distribution[e_sel] * E_dist[e_sel, port].X
                            for port in ports)
        cost_lo_out     = N * C_LO[e_sel].X

        total_V_used = energy_volume_density[e_sel] * q[e_sel].X + phi_volume[e_sel] * x[e_sel].X
        total_W_used = energy_weight_density[e_sel] * q[e_sel].X + phi_weight[e_sel] * x[e_sel].X

        # Well-to-wake efficiency (useful propulsion energy / primary energy)
        wtw_eff = (E_wake[e_sel].X * N / E_prod[e_sel].X) if E_prod[e_sel].X > 0 else 0

        # Efficiencies (all stages)
        E_dist_tot = sum(E_dist[e_sel, p].X for p in ports)

        eff_conv = E_wake[e_sel].X / E_conv[e_sel].X if E_conv[e_sel].X > 0 else 0
        eff_dist = (E_conv[e_sel].X * N) / E_dist_tot if E_dist_tot > 0 else 0
        eff_prod = E_dist_tot / E_prod[e_sel].X if E_prod[e_sel].X > 0 else 0

        # Transport work and energy intensity (annual)
        m3_per_lm = 2.5 * 4.0
        lm_lost = delta_V[e_sel].X / m3_per_lm
        remaining_lm = max(0, p['lane_meters'] - lm_lost)
        remaining_fraction = remaining_lm / p['lane_meters']

        transport_work = remaining_lm * length * 2 * N
        max_transport_work = p['lane_meters'] * length * 2 * N

        energy_intensity = (E_wake[e_sel].X * N) / transport_work if transport_work > 0 else np.inf
        energy_intensity_gross = (E_wake[e_sel].X * N) / max_transport_work

        cost_per_transport_work = model.objVal / transport_work if transport_work > 0 else np.inf
        cost_per_transport_work_gross = model.objVal / max_transport_work

        sum_breakdown = (
            cost_system_out + cost_plant_out + cost_port_out
            + cost_prod_out + cost_dist_out + cost_lo_out
        )

        if verbose:
            print("Sum breakdown:", sum_breakdown)
            print("Model objective:", model.objVal)
            print("Difference:", sum_breakdown - model.objVal)
            print_relevant_binding_constraints(model)
            print_relevant_lp_shadow_prices(model)
            for lo_v, lo_w in [(800,80), (400,40), (100,10), (10,1)]:
                rb = solve_shipping_model(270, params={'cost_LO_V': lo_v, 'cost_LO_W': lo_w}, force_carrier='battery')
                rh = solve_shipping_model(270, params={'cost_LO_V': lo_v, 'cost_LO_W': lo_w}, force_carrier='hydrogen')
                if rb and rh:
                    print(lo_v, lo_w, rb['total_cost'], rh['total_cost'], rb['cost_lo'], rh['cost_lo'])
            for hp in [90, 135, 180, 225, 270]:
                rh = solve_shipping_model(270, params={'cost_production': {'battery': 50, 'hydrogen': hp}}, force_carrier='hydrogen')
                rb = solve_shipping_model(270, force_carrier='battery')
                if rb and rh:
                    print(hp, rb['total_cost'], rh['total_cost'])

    
        return {
            'length':                  length,
            'total_cost':              model.objVal,
            'carrier':                 e_sel,
            'onboard_cap_MWh':         q[e_sel].X,
            'remaining_fraction':      remaining_fraction,
            'total_transport_work':          transport_work,
            'max_transport_work':            max_transport_work,
            'energy_intensity':              round(energy_intensity, 6),
            'energy_intensity_gross':        round(energy_intensity_gross, 6),
            'cost_per_transport_work':       cost_per_transport_work,
            'cost_per_transport_work_gross': cost_per_transport_work_gross,
            'E_wake_MWh':              E_wake[e_sel].X * N,
            'E_conv_MWh':              E_conv[e_sel].X * N,
            'E_dist_MWh':              sum(E_dist[e_sel, p].X for p in ports),
            'E_prod_MWh':              E_prod[e_sel].X, 
            'L_conv_MWh':              L_conv[e_sel].X * N,
            'L_dist_MWh':              sum(L_dist[e_sel, p].X for p in ports),
            'L_prod_MWh':              L_prod[e_sel].X,
            'port_capacity_MW':       {port: S_port[e_sel, port].X for port in ports},
            'eff_conversion':          round(eff_conv, 3),
            'eff_distribution':        round(eff_dist, 3),
            'eff_production':          round(eff_prod, 3),
            'wtw_efficiency':          round(wtw_eff, 3),
            'delta_V_m3':              delta_V[e_sel].X,
            'delta_W_t':               delta_W[e_sel].X,
            'pct_volume_used':         round(100 * total_V_used / volume_max, 1),
            'pct_weight_used':         round(100 * total_W_used / weight_max, 1),
            'charging_ports':          ports_with_infra,
            'cost_system':             cost_system_out,
            'cost_plant':              cost_plant_out,
            'cost_port':               cost_port_out,
            'cost_production':         cost_prod_out,
            'cost_distribution':       cost_dist_out,
            'cost_lo':                 cost_lo_out,
        }
    else:
        run_type = force_carrier if force_carrier else "optimal_choice"
        if verbose:
            print(f"[!] No optimal solution found for length={length} nm"
                  f"(run={run_type}, status={model.status})")
    return None

# =============================================================================
# SANKEY DIAGRAM - energy chain for a single run
# =============================================================================
 
def plot_sankey(result, title=None):
    """
    Plot a Plotly Sankey diagram of the energy chain for one model result.
    """
    e     = result['carrier']
    E_p   = result['E_prod_MWh']
    L_pr  = result['L_prod_MWh']
    E_d   = result['E_dist_MWh']
    L_di  = result['L_dist_MWh']
    E_c   = result['E_conv_MWh']
    L_co  = result['L_conv_MWh']
    E_w   = result['E_wake_MWh']
 
    # Node labels include MWh values
    node_labels = [
        f'Primary energy<br>{E_p:.1f} MWh/year',
        f'Production loss<br>{L_pr:.1f} MWh/year',
        f'Distributed energy<br>{E_d:.1f} MWh/year',
        f'Distribution loss<br>{L_di:.1f} MWh/year',
        f'Onboard energy<br>{E_c:.1f} MWh/year',
        f'Conversion loss<br>{L_co:.1f} MWh/year',
        f'Useful propulsion<br>{E_w:.1f} MWh/year',
    ]

    node_colors = [
        '#2196F3',  # 0 primary
        '#FF7043',  # 1 prod loss
        '#5C6BC0',  # 2 distributed
        '#7E57C2',  # 3 dist loss
        '#26A69A',  # 4 onboard
        '#FFA726',  # 5 conv loss
        '#66BB6A',  # 6 wake
    ]
 
    sources = [0, 0, 2, 2, 4, 4]
    targets = [1, 2, 3, 4, 5, 6]
    values  = [L_pr, E_d, L_di, E_c, L_co, E_w]

    link_colors = [
    'rgba(255,112,67,0.5)',   # prod loss     
    'rgba(92,107,192,0.5)',   # distributed   
    'rgba(126,87,194,0.5)',   # dist loss    
    'rgba(38,166,154,0.5)',   # onboard       
    'rgba(255,167,38,0.5)',   # conv loss     
    'rgba(102,187,106,0.5)',  # wake   
    ]      

    link_labels = [
        f'Production loss: {L_pr:.1f} MWh ({100*L_pr/E_p:.1f}% of primary)',
        f'To distribution: {E_d:.1f} MWh',
        f'Distribution loss: {L_di:.1f} MWh ({100*L_di/E_p:.1f}% of primary)',
        f'To onboard: {E_c:.1f} MWh',
        f'Conversion loss: {L_co:.1f} MWh ({100*L_co/E_p:.1f}% of primary)',
        f'Useful propulsion: {E_w:.1f} MWh ({100*E_w/E_p:.1f}% of primary)',
    ]
 
    fig = go.Figure(go.Sankey(
        arrangement='snap',
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color='white', width=0.5),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            label=link_labels,
        )
    ))

    wtw = result['wtw_efficiency']
    label = title or f"Energy chain - {e.capitalize()} - {result['length']} nm  |  WtW efficiency: {wtw:.1%}"
    fig.update_layout(
        title_text=label,
        title_x=0.5,
        font_size=13,
        height=450,
    )
    fig.write_html(f"{OUT_DIR}sankey_{e}_{result['length']}nm.html")
    fig.show()
    print(f"  Sankey saved: {OUT_DIR}sankey_{e}_{result['length']}nm.html")

# =============================================================================
# CASE: Tananger - Aberdeen (270 nm)
# =============================================================================

print("=" * 60)
print("CASE: Tananger - Aberdeen (270 nm)")
print("=" * 60)

phi_v, phi_w = calculate_converter_footprint(
    DEFAULT_PARAMS['power_kw'],
    DEFAULT_PARAMS['pemfc_volume_density'],
    DEFAULT_PARAMS['pemfc_weight_density']
)
print(f"Hydrogen converter volume footprint: {phi_v:.2f} m3")
print(f"Hydrogen converter weight footprint: {phi_w:.2f} tonne")

case = solve_shipping_model(270)

if case:
    print(f"\n Carrier selected: {case['carrier'].capitalize()}")
    print(f"  Total cost       : {case['total_cost']:,.0f} USD/year")
    print(f"  Onboard capacity : {case['onboard_cap_MWh']:.1f} MWh")
    print(f"  Charging ports   : {case['charging_ports']}")
 
    print(f"\n  {'-'*60}")
    print(f"  Energy chain")
    print(f"  {'-'*60}")
    print(f"  {'Stage':<22} {'Energy in (MWh)':>16} {'Loss (MWh)':>12} {'Loss %':>8}")
    print(f"  {'-'*60}")
    E_p = case['E_prod_MWh']
    print(f"  {'Production':<22} {E_p:>16.1f} "
          f"{case['L_prod_MWh']:>12.1f} {100*case['L_prod_MWh']/E_p:>7.1f}%")
    print(f"  {'Distribution':<22} {case['E_dist_MWh']:>16.1f} "
          f"{case['L_dist_MWh']:>12.1f} {100*case['L_dist_MWh']/E_p:>7.1f}%")
    print(f"  {'Conversion':<22} {case['E_conv_MWh']:>16.1f} "
          f"{case['L_conv_MWh']:>12.1f} {100*case['L_conv_MWh']/E_p:>7.1f}%")
    print(f"  {'Useful propulsion':<22} {case['E_wake_MWh']:>16.1f} "
          f"{'—':>12} {'':>8}")
    print(f"  {'-'*60}")
    print(f"  {'Well-to-wake eff.':<22} {'':>16} {'':>12} "
          f"{case['wtw_efficiency']:>7.1%}")
 
    print(f"\n  {'-'*60}")
    print(f"  Physical footprint")
    print(f"  {'-'*60}")
    print(f"  Volume used : {case['pct_volume_used']:.1f}% of total cargo volume")
    print(f"  Weight used : {case['pct_weight_used']:.1f}% of total cargo weight")
    print(f"  Lost cargo  : {case['delta_V_m3']:.0f} m³  |  {case['delta_W_t']:.0f} t")
 
    print(f"\n  {'-'*60}")
    print(f"  Cost breakdown (USD/year)")
    print(f"  {'-'*60}")
    print(f"Remaining cargo fraction:         {case['remaining_fraction']:.2%}")
    print(f"Transport work (actual):          {case['total_transport_work']:,.0f} lm·nm/year")
    print(f"Transport work (max):             {case['max_transport_work']:,.0f} lm·nm/year")
    print(f"Capacity loss from system:        {100*(1-case['remaining_fraction']):.1f}%")
    print(f"Energy intensity (actual cargo):  {case['energy_intensity']:.6f} MWh/lm·nm")
    print(f"Energy intensity (max cargo):     {case['energy_intensity_gross']:.6f} MWh/lm·nm")
    print(f"Cost per transport work (actual): {case['cost_per_transport_work']:.4f} USD/lm·nm")
    print(f"Cost per transport work (max):    {case['cost_per_transport_work_gross']:.4f} USD/lm·nm")
    for k, label in [
        ('cost_system',       'Onboard system'),
        ('cost_plant',        'Plant investment'),
        ('cost_port',         'Port infrastructure'),
        ('cost_production',   'Production'),
        ('cost_distribution', 'Distribution'),
        ('cost_lo',           'Lost opportunity'),
    ]:
        share = 100 * case[k] / case['total_cost']
        print(f"  {label:<22} {case[k]:>12,.0f}   ({share:.1f}%)")
    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<22} {case['total_cost']:>12,.0f}")
 
    plot_sankey(case) 

# =============================================================================
# SENSITIVITY ANALYSIS 
# =============================================================================

route_lengths = sorted(list(range(50, 801, 10)))

optimal_results  = []
battery_results  = []
hydrogen_results = []

for route_length in route_lengths:
    optimal_result  = solve_shipping_model(route_length)
    battery_result  = solve_shipping_model(route_length, force_carrier='battery')
    hydrogen_result = solve_shipping_model(route_length, force_carrier='hydrogen')

    if optimal_result:
        optimal_results.append(optimal_result)
    if battery_result:
        battery_results.append(battery_result) 
    if hydrogen_result:
        hydrogen_results.append(hydrogen_result)

df_optimal  = pd.DataFrame(optimal_results)
df_battery  = pd.DataFrame(battery_results)
df_hydrogen = pd.DataFrame(hydrogen_results)

# =============================================================================
# SENSITIVITY - available onboard volume
# =============================================================================

vol_range = range(500, 15_001, 500)
vol_results = []
for vol in vol_range:
    r = solve_shipping_model(270, params={'volume_available': vol})
    if r:
        r['volume_available'] = vol
        vol_results.append(r)
df_vol = pd.DataFrame(vol_results)

# =============================================================================
# SENSITIVITY - energy prices (battery vs hydrogen)
# =============================================================================

bat_prices = range(20, 201, 20)
hyd_prices = range(20, 301, 20)
price_results = []
for bp in bat_prices:
    for hp in hyd_prices:
        r = solve_shipping_model(340, params={
            'cost_production':   {'battery': bp, 'hydrogen': hp},
            'cost_distribution': {'battery': bp//5, 'hydrogen': hp//4},
        })
        if r:
            price_results.append({**r, 'bat_price': bp, 'hyd_price': hp})
df_price = pd.DataFrame(price_results)

# =============================================================================
# SENSITIVITY - infrastructure cost scale (forced battery and hydrogen)
# =============================================================================

base_port  = DEFAULT_PARAMS['cost_port']
base_plant = DEFAULT_PARAMS['cost_plant_power']
infra_bat_results = []
infra_hyd_results = []
for scale in np.linspace(0.25, 3.0, 20):
    _p = {
        'cost_port':  {e: base_port[e] * scale for e in ['battery', 'hydrogen']},
        'cost_plant_power': {e: base_plant[e] * scale for e in ['battery', 'hydrogen']},
    }
    r_bat = solve_shipping_model(270, params=_p, force_carrier='battery')
    r_hyd = solve_shipping_model(270, params=_p, force_carrier='hydrogen')
    if r_bat: infra_bat_results.append({**r_bat, 'infra_scale': round(scale, 2)})
    if r_hyd: infra_hyd_results.append({**r_hyd, 'infra_scale': round(scale, 2)})
df_infra_bat = pd.DataFrame(infra_bat_results)
df_infra_hyd = pd.DataFrame(infra_hyd_results)


# =============================================================================
# COLOURS
# =============================================================================

COLORS = {
    'battery':          '#2196F3',
    'hydrogen':         '#FF7043',
    'cost_system':      '#5C6BC0',
    'cost_plant':       '#7E57C2',
    'cost_port':        '#26A69A',
    'cost_production':  '#FFA726',
    'cost_distribution':'#66BB6A',
    'cost_lo':          '#EF5350',
}
plt.rcParams['axes.xmargin'] = 0
fmt_usd = mticker.FuncFormatter(lambda x, _: f'{x:,.0f}')

# =============================================================================
# PLOT - Cost share at 270 nm: Battery vs Hydrogen
# =============================================================================

pie_costs  = ['cost_system', 'cost_plant', 'cost_port', 'cost_production', 'cost_distribution', 'cost_lo',]
pie_labels = ['Onboard system', 'Plant investment', 'Port infrastructure', 'Production', 'Distribution', 'Lost opportunity',]
pie_colors = [COLORS[cost_type] for cost_type in pie_costs]

battery_270  = solve_shipping_model(270, force_carrier='battery')
hydrogen_270 = solve_shipping_model(270, force_carrier='hydrogen')

fig_cost_share, (ax_battery, ax_hydrogen) = plt.subplots(1, 2, figsize=(11, 5))

for ax, result, carrier, plot_title in [
    (ax_battery, battery_270, 'battery', 'Battery'),
    (ax_hydrogen, hydrogen_270, 'hydrogen', 'Hydrogen'),]:
    values = [result[cost_type] for cost_type in pie_costs]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        colors=pie_colors,
        autopct=lambda percentage: f'{percentage:.1f}%' if percentage > 3 else '',
        startangle=90,
        pctdistance=0.75,
    )

    ax.set_title( f'{plot_title}\nTotal: {result["total_cost"]:,.0f} USD/year', 
                 fontsize=10, color=COLORS[carrier],)

fig_cost_share.legend(wedges, pie_labels, loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.01),)
fig_cost_share.tight_layout(rect=[0, 0.08, 1, 1])
fig_cost_share.savefig(f'{OUT_DIR}cost_shares_270nm.png', dpi=300)

# =============================================================================
# TECH CROSSOVER SENSITIVITY - Technology crossover
# =============================================================================

fig_crossover, ax_crossover = plt.subplots(figsize=(9, 5))

ax_crossover.plot(df_battery['length'], df_battery['total_cost'], color=COLORS['battery'], linewidth=2, label='Battery',)
ax_crossover.plot(df_hydrogen['length'], df_hydrogen['total_cost'], color=COLORS['hydrogen'], linewidth=2, label='Hydrogen',)

crossover_length = None

df_crossover = pd.merge(df_battery[['length', 'total_cost']], df_hydrogen[['length', 'total_cost']],
    on='length', suffixes=('_battery', '_hydrogen'),)

battery_costs  = df_crossover['total_cost_battery'].values
hydrogen_costs = df_crossover['total_cost_hydrogen'].values
route_lengths  = df_crossover['length'].values

for index in range(1, len(battery_costs)):
    previous_difference = (battery_costs[index - 1] - hydrogen_costs[index - 1])
    current_difference  = (battery_costs[index] - hydrogen_costs[index])
    if previous_difference * current_difference < 0:
        interpolation_fraction = (previous_difference / (previous_difference - current_difference))
        crossover_length = (route_lengths[index - 1] + interpolation_fraction * (route_lengths[index] - route_lengths[index - 1]))
        break

minimum_cost = min(battery_costs.min(), hydrogen_costs.min(),)
maximum_cost = max(battery_costs.max(), hydrogen_costs.max(),)

if crossover_length is not None:
    crossover_cost = np.interp(crossover_length, df_battery['length'], df_battery['total_cost'])
    ax_crossover.axvline(crossover_length, color='gray', linestyle='--', linewidth=1.0,)
    ax_crossover.scatter(crossover_length, crossover_cost, color='black', s=30, zorder=5)
    ax_crossover.text(crossover_length - 10, maximum_cost * 0.95, '← Battery optimal',
        ha='right', va='top', fontsize=8, color=COLORS['battery'],)
    ax_crossover.text(crossover_length + 10, maximum_cost * 0.95, 'Hydrogen optimal →',
        ha='left', va='top', fontsize=8, color=COLORS['hydrogen'],)
    ax_crossover.text(crossover_length, minimum_cost + 0.05 * (maximum_cost - minimum_cost),
        f'{crossover_length:.0f} nm', ha='center', va='bottom', fontsize=8, color='gray',)

ax_crossover.set_xlabel('One-way route length (nm)')
ax_crossover.set_ylabel('Total cost per year (USD)')
ax_crossover.yaxis.set_major_formatter(fmt_usd)
ax_crossover.legend(fontsize=8)
ax_crossover.grid(linestyle='--', alpha=0.5,)
fig_crossover.tight_layout()
fig_crossover.savefig(OUT_DIR + 'technology_crossover.png', dpi=300,)


# =============================================================================
# COMBINED CROSSOVER SENSITIVITY - Combined crossover + cargo displacement 
# =============================================================================

# Find where battery first displaces cargo weight and volume separately
cargo_displacement_W = df_battery.loc[df_battery['delta_W_t'] > 0.01, 'length'].min()
cargo_displacement_V = df_battery.loc[df_battery['delta_V_m3'] > 0.01, 'length'].min()

fig_combined, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
    gridspec_kw={'hspace': 0.08})

# Top panel: total cost crossover
ax_top.plot(df_battery['length'], df_battery['total_cost'],
    color=COLORS['battery'], linewidth=2, label='Battery')
ax_top.plot(df_hydrogen['length'], df_hydrogen['total_cost'],
    color=COLORS['hydrogen'], linewidth=2, label='Hydrogen')

# Shaded zone starts at volume displacement (500 nm) not weight
if crossover_length is not None:
    ax_top.axvspan(cargo_displacement_V, crossover_length,
        alpha=0.07, color='gray', label='Battery penalised but cheaper')
    ax_bot.axvspan(cargo_displacement_V, crossover_length,
        alpha=0.07, color='gray')

# Weight displacement line - salmon
ax_top.axvline(cargo_displacement_W, color='salmon', linestyle=':', linewidth=1.2)
ax_bot.axvline(cargo_displacement_W, color='salmon', linestyle=':', linewidth=1.2)

# Volume displacement line - darker orange/red
ax_top.axvline(cargo_displacement_V, color='#C0392B', linestyle=':', linewidth=1.2)
ax_bot.axvline(cargo_displacement_V, color='#C0392B', linestyle=':', linewidth=1.2)

# Tech crossover line
if crossover_length is not None:
    crossover_cost_combined = np.interp(crossover_length, df_battery['length'], df_battery['total_cost'])
    ax_top.axvline(crossover_length, color='gray', linestyle='--', linewidth=1.0)
    ax_top.scatter(crossover_length, crossover_cost_combined, color='black', s=30, zorder=5)
    ax_top.text(crossover_length + 10, crossover_cost_combined * 1.2, f'{crossover_length:.0f} nm', va='bottom', ha='left', fontsize=8, color='gray')
    ax_bot.axvline(crossover_length, color='gray', linestyle='--', linewidth=1.0)

ax_top.set_ylabel('Total cost per year (USD)')
ax_top.yaxis.set_major_formatter(fmt_usd)
ax_top.legend(fontsize=8)
ax_top.grid(linestyle='--', alpha=0.5)

# Bottom panel: cargo displaced (weight and volume)
ax_bot.fill_between(df_battery['length'], df_battery['delta_W_t'],
    color=COLORS['battery'], alpha=0.25,
    label=f'Weight displaced (t), starts {cargo_displacement_W:.0f} nm')
ax_bot.plot(df_battery['length'], df_battery['delta_W_t'],
    color=COLORS['battery'], linewidth=1.5)

ax_bot.fill_between(df_battery['length'], df_battery['delta_V_m3'],
    color='#42A5F5', alpha=0.5,
    label=f'Volume displaced (m³), starts {cargo_displacement_V:.0f} nm')
ax_bot.plot(df_battery['length'], df_battery['delta_V_m3'],
    color='#42A5F5', linewidth=1.5)


ax_bot.set_ylabel('Cargo displaced (t or m³)*')
ax_bot.set_xlabel('One-way route length (nm)')
ax_bot.legend(fontsize=8, loc='upper right')
ax_bot.grid(linestyle='--', alpha=0.5)

fig_combined.subplots_adjust(hspace=0.08)
fig_combined.savefig(OUT_DIR + 'combined_crossover_cargo.png', dpi=300)

# =============================================================================
# COST BREAKDOWN SENSITIVITY - Stacked area: Battery (top) vs Hydrogen (bottom)
# =============================================================================

cost_components = ['cost_system', 'cost_plant', 'cost_port', 'cost_production', 'cost_distribution', 'cost_lo',]
cost_component_labels = ['Onboard system', 'Plant investment', 'Port infrastructure', 'Production', 'Distribution', 'Lost opportunity', ]

fig_cost_breakdown, (ax_battery, ax_hydrogen) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, sharey=True,)

for ax, df_results, carrier in [(ax_battery, df_battery, 'battery'), (ax_hydrogen, df_hydrogen, 'hydrogen'),]:
    component_values = [df_results[cost_component].values for cost_component in cost_components]
    component_colors = [COLORS[cost_component] for cost_component in cost_components]

    ax.stackplot(df_results['length'], component_values, labels=cost_component_labels, colors=component_colors, alpha=0.85,)
    ax.set_ylabel('Cost per year (USD)')
    ax.yaxis.set_major_formatter(fmt_usd)
    ax.grid(axis='y', linestyle='--', alpha=0.5,)
    ax.set_facecolor('white')
    ax.text(0.02, 0.97, carrier.capitalize(), transform=ax.transAxes, va='top', fontsize=10, color=COLORS[carrier],)

ax_battery.legend(fontsize=8,)
ax_hydrogen.set_xlabel('One-way route length (nm)')
fig_cost_breakdown.tight_layout()
fig_cost_breakdown.savefig(OUT_DIR + 'cost_breakdown.png', dpi=300,)

# =============================================================================
# WtW SENSITIVITY - Well-to-wake energy stacked area: Battery vs Hydrogen
# =============================================================================

energy_components       = ['E_wake_MWh', 'L_conv_MWh', 'L_dist_MWh', 'L_prod_MWh']
energy_component_labels = ['Useful propulsion', 'Conversion loss', 'Distribution loss', 'Production loss']
energy_component_colors = ['#42A5F5', '#FF7043', '#FFA726', '#AB47BC']

fig_energy_breakdown, (ax_battery, ax_hydrogen) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, sharey=True)
for ax, df_results, carrier in [(ax_battery, df_battery, 'battery'),(ax_hydrogen, df_hydrogen, 'hydrogen'),]:
    energy_values = [df_results[energy_component].values for energy_component in energy_components]
    ax.stackplot(df_results['length'], energy_values, labels=energy_component_labels, colors=energy_component_colors, alpha=0.85)
    ax.set_ylabel('Primary energy per year (MWh)')
    ax.grid( axis='y', linestyle='--', alpha=0.5,)
    ax.text(0.02, 0.97, carrier.capitalize(), transform=ax.transAxes, va='top', fontsize=10, color=COLORS[carrier])

ax_battery.legend(fontsize=8)
ax_hydrogen.set_xlabel('One-way route length (nm)')
fig_energy_breakdown.tight_layout()
fig_energy_breakdown.savefig(OUT_DIR + 'WtW_breakdown.png', dpi=300)

# =============================================================================
# TORNADO SENSITIVITY PLOT - ±50% on all cost parameters, 270 nm base case
# =============================================================================

base_route_length = 270

# Base case results for both energy carriers
base_battery_result  = solve_shipping_model(base_route_length, force_carrier='battery', verbose=False)
base_hydrogen_result = solve_shipping_model(base_route_length, force_carrier='hydrogen', verbose=False)
base_battery_cost  = base_battery_result['total_cost']
base_hydrogen_cost = base_hydrogen_result['total_cost']


# Parameters to test: label, parameter key, and whether the parameter is carrier-specific
sensitivity_parameters = [
    ('Production cost',      'cost_production',    True),
    ('Onboard system cost',  'cost_storage',       True),
    ('Port cost',            'cost_port',          True),
    ('Plant investment',     'cost_plant_power',   True),
    ('Distribution cost',    'cost_distribution',  True),
    ('LO volume penalty',    'cost_LO_V',          False),
    ('LO weight penalty',    'cost_LO_W',          False),
]

def scale_parameter(parameter_key, scale_factor, is_carrier_specific):
    base_parameter_value = DEFAULT_PARAMS[parameter_key]
    if is_carrier_specific:
        return {parameter_key: {carrier: value * scale_factor for carrier, value in base_parameter_value.items()}}
    return {parameter_key: base_parameter_value * scale_factor}


sensitivity_results = []

for parameter_label, parameter_key, is_carrier_specific in sensitivity_parameters:
    for scale_factor, sensitivity_case in [(0.5, 'low'), (1.5, 'high')]:
        modified_parameters = scale_parameter(parameter_key, scale_factor, is_carrier_specific)
        battery_result = solve_shipping_model(base_route_length, params=modified_parameters, force_carrier='battery', verbose=False)
        hydrogen_result = solve_shipping_model(base_route_length, params=modified_parameters, force_carrier='hydrogen', verbose=False)
        sensitivity_results.append({
            'parameter_label':    parameter_label,
            'sensitivity_case':   sensitivity_case,
            'delta_battery_pct':  (100 * (battery_result['total_cost'] - base_battery_cost) / base_battery_cost if battery_result else np.nan),
            'delta_hydrogen_pct': (100 * (hydrogen_result['total_cost'] - base_hydrogen_cost) / base_hydrogen_cost if hydrogen_result else np.nan),})

df_tornado = pd.DataFrame(sensitivity_results)

# Sort parameters by battery cost swing for tornado shape
battery_cost_swing = (df_tornado.groupby('parameter_label')['delta_battery_pct'].apply(lambda values: values.max() - values.min()).sort_values(ascending=True))
sorted_parameter_labels = battery_cost_swing.index.tolist()

fig_tornado, (ax_battery, ax_hydrogen) = plt.subplots(2, 1, figsize=(10, 10), sharey=True, sharex=True,)

for ax, carrier, delta_column in [
    (ax_battery, 'battery', 'delta_battery_pct'),
    (ax_hydrogen, 'hydrogen', 'delta_hydrogen_pct'),
    ]:

    for parameter_index, parameter_label in enumerate(sorted_parameter_labels):
        parameter_results = df_tornado[df_tornado['parameter_label'] == parameter_label]
        low_delta = parameter_results.loc[parameter_results['sensitivity_case'] == 'low', delta_column].values[0]
        high_delta = parameter_results.loc[parameter_results['sensitivity_case'] == 'high', delta_column].values[0]

        ax.barh(parameter_index, low_delta, color=COLORS[carrier], alpha=0.5, height=0.5)
        ax.barh(parameter_index, high_delta, color=COLORS[carrier], alpha=0.9, height=0.5)

    ax.axvline(0, color='black', linewidth=0.8)
    ax.text(0.02, 0.97, carrier.capitalize(), transform=ax.transAxes,
            va='top', fontsize=10, color=COLORS[carrier])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:+.0f}%'))
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)

ax_hydrogen.set_xlabel('Change in total cost relative to base case (%)')
ax_battery.set_yticks(range(len(sorted_parameter_labels)))
ax_battery.set_yticklabels(sorted_parameter_labels)
ax_hydrogen.set_yticks(range(len(sorted_parameter_labels)))
ax_hydrogen.set_yticklabels(sorted_parameter_labels)

fig_tornado.tight_layout()
fig_tornado.savefig(f'{OUT_DIR}tornado_sensitivity.png', dpi=300)

# =============================================================================
# TRANSPORT WORK SENSITIVITY PLOT
# =============================================================================

fig_transport_work_cost, ax_transport_work_cost = plt.subplots(figsize=(8, 5))
ax_transport_work_cost.plot(df_battery['length'], df_battery['cost_per_transport_work'],
                            color=COLORS['battery'], linewidth=2, label='Battery (forced)')
ax_transport_work_cost.plot(df_hydrogen['length'], df_hydrogen['cost_per_transport_work'],
                            color=COLORS['hydrogen'], linewidth=2, label='Hydrogen (forced)')

# Find crossover where battery becomes more expensive per transport work than hydrogen
df_transport_work = df_battery[['length', 'cost_per_transport_work']].merge(
    df_hydrogen[['length', 'cost_per_transport_work']], on='length', suffixes=('_battery', '_hydrogen'))

diff = df_transport_work['cost_per_transport_work_battery'] - df_transport_work['cost_per_transport_work_hydrogen']
sign_changes = (diff.values[:-1] * diff.values[1:]) < 0
if sign_changes.any():
    idx = sign_changes.argmax()
    d0, d1 = diff.values[idx], diff.values[idx + 1]
    l0, l1 = df_transport_work['length'].values[idx], df_transport_work['length'].values[idx + 1]
    transport_work_crossover = l0 + d0 / (d0 - d1) * (l1 - l0)
else:
    transport_work_crossover = float('nan')

if not np.isnan(transport_work_crossover):
    crossover_cost = np.interp(transport_work_crossover, df_battery['length'], df_battery['cost_per_transport_work'])
    ax_transport_work_cost.scatter(transport_work_crossover, crossover_cost, color='black', s=30, zorder=5)
    ax_transport_work_cost.text(transport_work_crossover + 10, crossover_cost * 1.02, f'{transport_work_crossover:.0f} nm', va='bottom', ha='left', fontsize=8, color='gray')

ax_transport_work_cost.axvline(270, color='gray', linewidth=0.9, linestyle='--', label='Base case (270 nm)')
ax_transport_work_cost.set_xlim(left=50, right=800)
ax_transport_work_cost.set_xlabel('One-way route length (nm)')
ax_transport_work_cost.set_ylabel('Cost per transport work (USD/lm·nm)')
ax_transport_work_cost.grid(True, linestyle='--', alpha=0.4)
ax_transport_work_cost.legend(fontsize=8)

fig_transport_work_cost.tight_layout()
fig_transport_work_cost.savefig(f'{OUT_DIR}plot_cost_per_transport_work.png', dpi=150)

# =============================================================================
# SENSITIVITY - Hydrogen storage density
# Varies hydrogen volume and weight density together (same source uncertainty)
# Shows how crossover length shifts as storage density changes
# =============================================================================

base_vol_density = DEFAULT_PARAMS['energy_volume_density']['hydrogen']  # 1.16 m³/MWh
base_wgt_density = DEFAULT_PARAMS['energy_weight_density']['hydrogen']  # 0.58 t/MWh

# Test range: 50% to 200% of base (pessimistic to optimistic)
density_scales = np.linspace(0.5, 2.0, 10)
crossover_lengths = []
base_cost_at_270 = []

for scale in density_scales:
    pars = {
        'energy_volume_density': {
            'battery':  DEFAULT_PARAMS['energy_volume_density']['battery'],
            'hydrogen': base_vol_density * scale
        },
        'energy_weight_density': {
            'battery':  DEFAULT_PARAMS['energy_weight_density']['battery'],
            'hydrogen': base_wgt_density * scale
        }
    }

    # Sweep lengths to find crossover
    bat_costs, hyd_costs, lengths_sweep = [], [], list(range(50, 801, 10))
    for L in lengths_sweep:
        r_b = solve_shipping_model(L, params=pars, force_carrier='battery', verbose=False)
        r_h = solve_shipping_model(L, params=pars, force_carrier='hydrogen', verbose=False)
        bat_costs.append(r_b['total_cost'] if r_b else np.nan)
        hyd_costs.append(r_h['total_cost'] if r_h else np.nan)

    # Find crossover
    crossover = np.nan
    for k in range(1, len(lengths_sweep)):
        if (not np.isnan(bat_costs[k]) and not np.isnan(hyd_costs[k])
                and hyd_costs[k] < bat_costs[k]):
            crossover = lengths_sweep[k]
            break
    crossover_lengths.append(crossover)

    # Cost at 270 nm base case
    r_h_270 = solve_shipping_model(270, params=pars, force_carrier='hydrogen', verbose=False)
    base_cost_at_270.append(r_h_270['total_cost'] if r_h_270 else np.nan)

# --- Plot ---
fig_dens, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                           gridspec_kw={'hspace': 0.08})

x_labels = [f'{s:.1f}x\n({base_vol_density*s:.2f} m³/MWh)' for s in density_scales]

# Top: crossover length vs density scale
ax_top.plot(density_scales, crossover_lengths,
            color=COLORS['hydrogen'], linewidth=2, marker='o', ms=5)
ax_top.axvline(1.0, color='gray', linestyle='--', linewidth=0.9,
               label='Base case (DOE target)')
ax_top.axhline(793, color='gray', linestyle=':', linewidth=0.9,
               label='Base case crossover (793 nm)')
ax_top.set_ylabel('Technology crossover length (nm)')
ax_top.legend(fontsize=8)
ax_top.grid(True, linestyle='--', alpha=0.4)

# Bottom: hydrogen total cost at 270 nm vs density scale
ax_bot.plot(density_scales, base_cost_at_270,
            color=COLORS['hydrogen'], linewidth=2, marker='o', ms=5)
ax_bot.axvline(1.0, color='gray', linestyle='--', linewidth=0.9,
               label='Base case (DOE target)')
ax_bot.set_ylabel('Hydrogen total cost at 270 nm (USD/year)')
ax_bot.set_xlabel('Hydrogen storage density scale factor\n(1.0 = DOE base assumption)')
ax_bot.yaxis.set_major_formatter(fmt_usd)
ax_bot.legend(fontsize=8)
ax_bot.grid(True, linestyle='--', alpha=0.4)

fig_dens.tight_layout()
fig_dens.savefig(f'{OUT_DIR}sens_hydrogen_density.png', dpi=150)

# =============================================================================
# Show all plots
# =============================================================================
#plt.show()
print('\nAll plots saved.')