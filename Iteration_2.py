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

OUT_DIR = 'results/iteration_2/'
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# LOAD VESSEL DATA FROM EXCEL
# vessels_2.xlsx must contain columns:
#   vessel_id, vessel_type, count, dwt, teu, lane_meters,
#   energy_per_nm, volume_available, weight_available, volume_max, weight_max,
#   voyages_per_year, power_kw
# =============================================================================

df_vessels_raw = pd.read_excel('vessels_2.xlsx')
df_vessels_raw = df_vessels_raw.dropna(subset=['count'])

vessels_expanded_rows = []
for _, row in df_vessels_raw.iterrows():
    for i in range(int(row['count'])):
        new_row = row.copy()
        new_row['vessel_id'] = f"{row['vessel_id']}_{i+1}"
        vessels_expanded_rows.append(new_row)
df_vessels = pd.DataFrame(vessels_expanded_rows).reset_index(drop=True)

# =============================================================================
# DEFAULT PARAMETERS  (identical to Iteration_1)
# =============================================================================

DEFAULT_PARAMS = {
    # Route
    'lay_time_port':        6,          # [h]

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
    'cost_plant_power':      {'battery': 0,         'hydrogen': 1_424_000},  # [USD/MW]
    'plant_full_load_hours': {'battery': 8760,      'hydrogen': 4000},       # [h/year]

    # Operational costs
    'cost_production':   {'battery': 120,     'hydrogen': 120},     # [USD/MWh primary energy]
    'cost_distribution': {'battery': 0,       'hydrogen': 16.6},    # [USD/MWh entering distribution]
    'cost_port':         {'battery': 255_170, 'hydrogen': 356_000}, # [USD/MW]
}

# =============================================================================
# POST-PROCESSING HELPERS
# =============================================================================

def print_relevant_binding_constraints(model, tol=1e-6):
    relevant_keywords = [
        'physical', 'LO', 'C_LO', 'onboard_energy_capacity',
        'vessel_port_coupling', 'both_ports_indicator', 'production_plant_power',
        'energy_balance', 'port_capacity',
    ]
    print('Relevant binding constraints:')
    for c in model.getConstrs():
        if any(k in c.ConstrName for k in relevant_keywords):
            if abs(c.Slack) <= tol:
                print(f'  {c.ConstrName:50s} Slack={c.Slack:.4g} RHS={c.RHS:.4g}')

def print_relevant_lp_shadow_prices(model, tol=1e-6):
    relevant_keywords = [
        'physical', 'LO', 'C_LO', 'onboard_energy_capacity',
        'vessel_port_coupling', 'both_ports_indicator', 'production_plant_power',
        'energy_balance', 'port_capacity',
    ]
    relaxed = model.relax()
    relaxed.setParam('OutputFlag', 0)
    relaxed.optimize()
    if relaxed.status == GRB.OPTIMAL:
        print('Relevant shadow prices from LP relaxation:')
        for c in relaxed.getConstrs():
            if any(k in c.ConstrName for k in relevant_keywords):
                if abs(c.Pi) > tol:
                    print(f'  {c.ConstrName:50s} Pi={c.Pi:.4g} Slack={c.Slack:.4g}')

# =============================================================================
# MODEL FUNCTION
# =============================================================================

def solve_shipping_model(length, df_vessels, params=None, force_carrier=None, verbose=False):
    p = {**DEFAULT_PARAMS, **(params or {})}

    ports           = ['A', 'B']
    energy_carriers = ['battery', 'hydrogen']
    vessels         = df_vessels['vessel_id'].tolist()

    energy_per_nm    = dict(zip(df_vessels['vessel_id'], df_vessels['energy_per_nm']))
    volume_available = dict(zip(df_vessels['vessel_id'], df_vessels['volume_available']))
    weight_available = dict(zip(df_vessels['vessel_id'], df_vessels['weight_available']))
    volume_max       = dict(zip(df_vessels['vessel_id'], df_vessels['volume_max']))
    weight_max       = dict(zip(df_vessels['vessel_id'], df_vessels['weight_max']))
    N_v              = dict(zip(df_vessels['vessel_id'], df_vessels['voyages_per_year']))
    power_kW         = dict(zip(df_vessels['vessel_id'], df_vessels['power_kw']))
    T_port_v         = dict(zip(df_vessels['vessel_id'], df_vessels['lay_time_port']))

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
    cost_plant            = p['cost_plant_power']
    plant_full_load_hours = p['plant_full_load_hours']
    cost_production       = p['cost_production']
    cost_distribution     = p['cost_distribution']
    cost_port             = p['cost_port']
    F                     = p['usable_capacity_fraction']

    af_port  = p['r_port']  / (1 - (1 + p['r_port'])**(-p['T_port']))
    af_plant = p['r_plant'] / (1 - (1 + p['r_plant'])**(-p['T_plant']))
    af_system = {e: p['r_system'][e] / (1 - (1 + p['r_system'][e])**(-p['T_system'][e]))
                for e in energy_carriers}
    af_pcs    = {e: p['r_power_conversion_system'][e] / (1 - (1 + p['r_power_conversion_system'][e])**(-p['T_power_conversion_system'][e]))
                for e in energy_carriers}

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

    x       = model.addVars(energy_carriers,                 vtype=GRB.BINARY,     name='x')
    y       = model.addVars(energy_carriers, ports,          vtype=GRB.BINARY,     name='y')
    q       = model.addVars(energy_carriers, vessels,        vtype=GRB.CONTINUOUS, name='q')
    z       = model.addVars(energy_carriers, ports, vessels, vtype=GRB.CONTINUOUS, name='z')
    w       = model.addVars(energy_carriers,                 vtype=GRB.BINARY,     name='w')
    delta_V = model.addVars(energy_carriers, vessels,        vtype=GRB.CONTINUOUS, name='delta_V')
    delta_W = model.addVars(energy_carriers, vessels,        vtype=GRB.CONTINUOUS, name='delta_W')
    C_LO    = model.addVars(energy_carriers, vessels,        vtype=GRB.CONTINUOUS, name='C_LO')
    S_port  = model.addVars(energy_carriers, ports,          vtype=GRB.CONTINUOUS, name='S_port')
    P_plant = model.addVars(energy_carriers,                 vtype=GRB.CONTINUOUS, name='P_plant')

    E_wake = model.addVars(energy_carriers, vessels, vtype=GRB.CONTINUOUS, name='E_wake')
    E_conv = model.addVars(energy_carriers, vessels, vtype=GRB.CONTINUOUS, name='E_conv')
    E_dist = model.addVars(energy_carriers, ports,   vtype=GRB.CONTINUOUS, name='E_dist')
    E_prod = model.addVars(energy_carriers,          vtype=GRB.CONTINUOUS, name='E_prod')
    L_prod = model.addVars(energy_carriers,          vtype=GRB.CONTINUOUS, name='L_prod')
    L_dist = model.addVars(energy_carriers, ports,   vtype=GRB.CONTINUOUS, name='L_dist')
    L_conv = model.addVars(energy_carriers, vessels, vtype=GRB.CONTINUOUS, name='L_conv')

    model.setObjective(
        gp.quicksum(af_system[e] * cost_storage[e] * q[e, v]
                    + af_pcs[e] * cost_power_conversion[e] * (power_kW[v] / 1000) * x[e]
                    for e in energy_carriers for v in vessels)
        + gp.quicksum(af_plant * cost_plant[e] * P_plant[e]
                    for e in energy_carriers)
        + gp.quicksum(af_port * cost_port[e] * S_port[e, port]
                    for e in energy_carriers for port in ports)
        + gp.quicksum(cost_production[e] * E_prod[e]
                    for e in energy_carriers)
        + gp.quicksum(cost_distribution[e] * E_dist[e, port]
                    for e in energy_carriers for port in ports)
        + gp.quicksum(N_v[v] * C_LO[e, v]
                    for e in energy_carriers for v in vessels),
        GRB.MINIMIZE
    )

    # (1) Carrier selection
    if force_carrier:
        model.addConstr(x[force_carrier] == 1, name='carrier_forced')
        for e in energy_carriers:
            if e != force_carrier:
                model.addConstr(x[e] == 0, name=f'carrier_blocked_{e}')
    else:
        model.addConstr(x.sum() == 1, name='carrier_selection')

    # (2) Wake energy demand (round trip, per voyage)
    model.addConstrs(
        (E_wake[e, v] == 2 * length * energy_per_nm[v] * x[e]
        for e in energy_carriers for v in vessels),
        name='wake_energy'
    )
    # (3) Conversion
    model.addConstrs(
        (E_conv[e, v] == E_wake[e, v] / eta_conv[e]
        for e in energy_carriers for v in vessels),
        name='conversion_energy'
    )
    # (4) Distribution: annual energy at port after losses
    model.addConstrs(
        (E_dist[e, port] == gp.quicksum(z[e, port, v] for v in vessels) / eta_dist[e]
        for e in energy_carriers for port in ports),
        name='distribution_energy'
    )
    # (5) Production: primary energy required
    model.addConstrs(
        (E_prod[e] == gp.quicksum(E_dist[e, port] for port in ports) / eta_prod[e]
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
        (gp.quicksum(z[e, port, v] for port in ports) == N_v[v] * E_conv[e, v]
        for e in energy_carriers for v in vessels),
        name='energy_balance'
    )
    # (8) Port throughput capacity (sized for peak single-vessel charge per voyage)
    model.addConstrs(
        (S_port[e, port] >= z[e, port, v] / (N_v[v] * T_port_v[v])
        for e in energy_carriers for port in ports for v in vessels),
        name='port_capacity'
    )
    # (9-11) Energy losses
    model.addConstrs(
        (L_conv[e, v] == E_conv[e, v] - E_wake[e, v]
        for e in energy_carriers for v in vessels),
        name='conversion_loss'
    )
    model.addConstrs(
        (L_dist[e, port] == E_dist[e, port] - gp.quicksum(z[e, port, v] for v in vessels)
        for e in energy_carriers for port in ports),
        name='distribution_loss'
    )
    model.addConstrs(
        (L_prod[e] == E_prod[e] - gp.quicksum(E_dist[e, port] for port in ports)
        for e in energy_carriers),
        name='production_loss'
    )
    # (12) Port capacity linked to infrastructure
    model.addConstrs(
        (S_port[e, port] <= M * y[e, port]
        for e in energy_carriers for port in ports),
        name='port_capacity_infrastructure'
    )
    # (13-14) Vessel-port coupling
    model.addConstrs(
        (z[e, port, v] <= M * y[e, port]
        for e in energy_carriers for port in ports for v in vessels),
        name='vessel_port_coupling_1'
    )
    model.addConstrs(
        (y[e, port] <= x[e] for e in energy_carriers for port in ports),
        name='vessel_port_coupling_2'
    )

    # Per-voyage charging at any port cannot exceed onboard tank capacity
    model.addConstrs(
        (z[e, port, v] <= N_v[v] * q[e, v]
        for e in energy_carriers for port in ports for v in vessels),
        name='charging_tank_limit'
    )
    # (15-16) w[e] = 1 iff infrastructure in both ports
    model.addConstrs(
        (w[e] <= y[e, port] for e in energy_carriers for port in ports),
        name='both_ports_indicator_1'
    )
    model.addConstrs(
        (w[e] >= y.sum(e, '*') - 1 for e in energy_carriers),
        name='both_ports_indicator_2'
    )
    # (17) Onboard energy capacity
    model.addConstrs(
        (q[e, v] == (2 * x[e] - w[e]) * (length * energy_per_nm[v] / (eta_conv[e] * F[e]))
        for e in energy_carriers for v in vessels),
        name='onboard_energy_capacity'
    )
    # (18) No capacity for unselected carrier
    model.addConstrs(
        (q[e, v] <= M * x[e] for e in energy_carriers for v in vessels),
        name='zero_capacity_unselected'
    )
    # (19-20) Hard physical caps
    model.addConstrs(
        (energy_volume_density[e] * q[e, v] + phi_volume[e, v] * x[e] <= volume_max[v] * x[e]
        for e in energy_carriers for v in vessels),
        name='physical_volume_cap'
    )
    model.addConstrs(
        (energy_weight_density[e] * q[e, v] + phi_weight[e, v] * x[e] <= weight_max[v] * x[e]
        for e in energy_carriers for v in vessels),
        name='physical_weight_cap'
    )
    # (21-22) Lost cargo caps
    model.addConstrs(
        (delta_V[e, v] <= volume_max[v] * x[e] for e in energy_carriers for v in vessels),
        name='LO_V_limit'
    )
    model.addConstrs(
        (delta_W[e, v] <= weight_max[v] * x[e] for e in energy_carriers for v in vessels),
        name='LO_W_limit'
    )
    # (23-24) Lost cargo = storage footprint minus freed space
    model.addConstrs(
        (delta_V[e, v] >= energy_volume_density[e] * q[e, v]
                        + phi_volume[e, v] * x[e]
                        - volume_available[v] * x[e]
        for e in energy_carriers for v in vessels),
        name='LO_V_kick_in'
    )
    model.addConstrs(
        (delta_W[e, v] >= energy_weight_density[e] * q[e, v]
                        + phi_weight[e, v] * x[e]
                        - weight_available[v] * x[e]
        for e in energy_carriers for v in vessels),
        name='LO_W_kick_in'
    )
    # (25-26) C_LO = max(volume penalty, weight penalty)
    model.addConstrs(
        (C_LO[e, v] >= cost_LO_V * delta_V[e, v] for e in energy_carriers for v in vessels),
        name='C_LO_volume'
    )
    model.addConstrs(
        (C_LO[e, v] >= cost_LO_W * delta_W[e, v] for e in energy_carriers for v in vessels),
        name='C_LO_weight'
    )

    model.optimize()

    if model.status == GRB.OPTIMAL:
        e_sel            = [e for e in energy_carriers if x[e].X > 0.5][0]
        ports_with_infra = int(sum(y[e_sel, port].X for port in ports))

        cost_system_out = sum(
            af_system[e_sel] * cost_storage[e_sel] * q[e_sel, v].X
            + af_pcs[e_sel] * cost_power_conversion[e_sel] * (power_kW[v] / 1000)
            for v in vessels)
        cost_plant_out = af_plant * cost_plant[e_sel] * P_plant[e_sel].X
        cost_port_out  = sum(af_port * cost_port[e_sel] * S_port[e_sel, port].X
                            for port in ports)
        cost_prod_out  = cost_production[e_sel] * E_prod[e_sel].X
        cost_dist_out  = sum(cost_distribution[e_sel] * E_dist[e_sel, port].X for port in ports)
        cost_lo_out    = sum(N_v[v] * C_LO[e_sel, v].X for v in vessels)

        E_wake_tot = sum(N_v[v] * E_wake[e_sel, v].X for v in vessels)
        E_conv_tot = sum(N_v[v] * E_conv[e_sel, v].X for v in vessels)
        E_dist_tot = sum(E_dist[e_sel, port].X for port in ports)
        E_prod_tot = E_prod[e_sel].X
        L_conv_tot = sum(N_v[v] * L_conv[e_sel, v].X for v in vessels)
        L_dist_tot = sum(L_dist[e_sel, port].X for port in ports)
        L_prod_tot = L_prod[e_sel].X

        wtw_eff  = E_wake_tot / E_prod_tot if E_prod_tot > 0 else 0
        eff_conv = E_wake_tot / E_conv_tot if E_conv_tot > 0 else 0
        eff_dist = E_conv_tot / E_dist_tot if E_dist_tot > 0 else 0
        eff_prod = E_dist_tot / E_prod_tot if E_prod_tot > 0 else 0

        vessel_row_map       = df_vessels.set_index('vessel_id')
        total_transport_work = 0
        max_transport_work   = 0
        total_V_by_vessel    = {}
        total_W_by_vessel    = {}

        m3_per_lm  = 2.5 * 4.0
        m3_per_teu = 33.0

        for v in vessels:
            row       = vessel_row_map.loc[v]
            total_V_v = energy_volume_density[e_sel] * q[e_sel, v].X + phi_volume[e_sel, v] * x[e_sel].X
            total_W_v = energy_weight_density[e_sel] * q[e_sel, v].X + phi_weight[e_sel, v] * x[e_sel].X
            total_V_by_vessel[v] = total_V_v
            total_W_by_vessel[v] = total_W_v
            vtype = str(row['vessel_type']).lower()
            if vtype == 'roro':
                remaining_cap = max(0, row['lane_meters'] - delta_V[e_sel, v].X / m3_per_lm)
                cap_unit      = row['lane_meters']
            elif vtype == 'container':
                remaining_cap = max(0, row['teu'] - delta_V[e_sel, v].X / m3_per_teu)
                cap_unit      = row['teu']
            else:
                remaining_cap = max(0, row['dwt'] - delta_W[e_sel, v].X)
                cap_unit      = row['dwt']
            total_transport_work += N_v[v] * remaining_cap * length * 2
            max_transport_work   += N_v[v] * cap_unit * length * 2

        remaining_fraction            = total_transport_work / max_transport_work if max_transport_work > 0 else 0
        cost_per_transport_work       = model.objVal / total_transport_work if total_transport_work > 0 else float('inf')
        cost_per_transport_work_gross = model.objVal / max_transport_work

        onboard_capacity  = {v: q[e_sel, v].X for v in vessels}
        cargo_loss_volume = {v: delta_V[e_sel, v].X for v in vessels}
        cargo_loss_weight = {v: delta_W[e_sel, v].X for v in vessels}
        volume_share = {v: round(100 * total_V_by_vessel[v] / volume_max[v], 1) for v in vessels}
        weight_share = {v: round(100 * total_W_by_vessel[v] / weight_max[v], 1) for v in vessels}

        if verbose:
            print_relevant_binding_constraints(model)
            print_relevant_lp_shadow_prices(model)

        return {
            'length':                        length,
            'total_cost':                    model.objVal,
            'carrier':                       e_sel,
            'charging_ports':                ports_with_infra,
            'n_vessels':                     len(vessels),
            'remaining_fraction':            remaining_fraction,
            'total_transport_work':          total_transport_work,
            'max_transport_work':            max_transport_work,
            'cost_per_transport_work':       cost_per_transport_work,
            'cost_per_transport_work_gross': cost_per_transport_work_gross,
            'E_wake_MWh':                    E_wake_tot,
            'E_conv_MWh':                    E_conv_tot,
            'E_dist_MWh':                    E_dist_tot,
            'E_prod_MWh':                    E_prod_tot,
            'L_conv_MWh':                    L_conv_tot,
            'L_dist_MWh':                    L_dist_tot,
            'L_prod_MWh':                    L_prod_tot,
            'port_capacity_MW':              {port: S_port[e_sel, port].X for port in ports},
            'eff_conversion':                round(eff_conv, 3),
            'eff_distribution':              round(eff_dist, 3),
            'eff_production':                round(eff_prod, 3),
            'wtw_efficiency':                round(wtw_eff, 3),
            'delta_V_m3':                    sum(delta_V[e_sel, v].X for v in vessels),
            'delta_W_t':                     sum(delta_W[e_sel, v].X for v in vessels),
            'pct_volume_used':               round(100 * sum(total_V_by_vessel[v] for v in vessels) / sum(volume_max.values()), 1),
            'pct_weight_used':               round(100 * sum(total_W_by_vessel[v] for v in vessels) / sum(weight_max.values()), 1),
            'cost_system':                   cost_system_out,
            'cost_plant':                    cost_plant_out,
            'cost_port':                     cost_port_out,
            'cost_production':               cost_prod_out,
            'cost_distribution':             cost_dist_out,
            'cost_lo':                       cost_lo_out,
            'onboard_capacity':              onboard_capacity,
            'cargo_loss_volume':             cargo_loss_volume,
            'cargo_loss_weight':             cargo_loss_weight,
            'pct_volume_by_vessel':          volume_share,
            'pct_weight_by_vessel':          weight_share,
        }
    else:
        if verbose:
            print(f"[!] No optimal solution found (status={model.status})")
        return None

# =============================================================================
# SANKEY DIAGRAM
# =============================================================================

def plot_sankey(result, title=None):
    e    = result['carrier']
    E_p  = result['E_prod_MWh']
    L_pr = result['L_prod_MWh']
    E_d  = result['E_dist_MWh']
    L_di = result['L_dist_MWh']
    E_c  = result['E_conv_MWh']
    L_co = result['L_conv_MWh']
    E_w  = result['E_wake_MWh']

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
    link_colors = [
        'rgba(255,112,67,0.5)', 'rgba(92,107,192,0.5)', 'rgba(126,87,194,0.5)',
        'rgba(38,166,154,0.5)', 'rgba(255,167,38,0.5)',  'rgba(102,187,106,0.5)',
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
        node=dict(pad=20, thickness=20, line=dict(color='white', width=0.5),
                label=node_labels, color=node_colors),
        link=dict(source=sources, target=targets, value=values,
                color=link_colors, label=link_labels)
    ))
    wtw   = result['wtw_efficiency']
    n_ves = result['n_vessels']
    label = title or (f"Energy chain - {e.capitalize()} - {result['length']} nm"
                    f" | {n_ves} vessels | WtW: {wtw:.1%}")
    fig.update_layout(title_text=label, title_x=0.5, font_size=13, height=450)
    fname = f"{OUT_DIR}sankey_{e}_{result['length']}nm.html"
    fig.write_html(fname)
    fig.show()
    print(f"  Sankey saved: {fname}")

# =============================================================================
# CASE: Tananger - Aberdeen (270 nm)
# =============================================================================

print("=" * 60)
print("CASE: Tananger - Aberdeen (270 nm)")
print("=" * 60)

case = solve_shipping_model(270, df_vessels)
if case:
    print(f"\n  Carrier selected : {case['carrier'].capitalize()}")
    print(f"  Total cost       : {case['total_cost']:,.0f} USD/year")
    print(f"  Charging ports   : {case['charging_ports']}")
    print(f"  Vessels          : {case['n_vessels']}")

    print(f"\n  {'-'*60}")
    print(f"  Energy chain (fleet total)")
    print(f"  {'-'*60}")
    print(f"  {'Stage':<22} {'Energy in (MWh)':>16} {'Loss (MWh)':>12} {'Loss %':>8}")
    print(f"  {'-'*60}")
    E_p = case['E_prod_MWh']
    print(f"  {'Production':<22} {E_p:>16.1f} {case['L_prod_MWh']:>12.1f} {100*case['L_prod_MWh']/E_p:>7.1f}%")
    print(f"  {'Distribution':<22} {case['E_dist_MWh']:>16.1f} {case['L_dist_MWh']:>12.1f} {100*case['L_dist_MWh']/E_p:>7.1f}%")
    print(f"  {'Conversion':<22} {case['E_conv_MWh']:>16.1f} {case['L_conv_MWh']:>12.1f} {100*case['L_conv_MWh']/E_p:>7.1f}%")
    print(f"  {'Useful propulsion':<22} {case['E_wake_MWh']:>16.1f} {'—':>12} {'':>8}")
    print(f"  {'-'*60}")
    print(f"  {'Well-to-wake eff.':<22} {'':>16} {'':>12} {case['wtw_efficiency']:>7.1%}")

    print(f"\n  {'-'*60}")
    print(f"  Physical footprint")
    print(f"  {'-'*60}")
    print(f"  Volume used (fleet): {case['pct_volume_used']:.1f}%")
    print(f"  Weight used (fleet): {case['pct_weight_used']:.1f}%")
    print(f"  Lost cargo (fleet) : {case['delta_V_m3']:.0f} m³  |  {case['delta_W_t']:.0f} t")

    print(f"\n  {'-'*60}")
    print(f"  Per vessel onboard capacity")
    print(f"  {'-'*60}")
    for v, cap in case['onboard_capacity'].items():
        pv = case['pct_volume_by_vessel'][v]
        pw = case['pct_weight_by_vessel'][v]
        print(f"  {v:<22} {cap:>8.1f} MWh   vol:{pv:>5.1f}%  wgt:{pw:>5.1f}%")

    print(f"\n  {'-'*60}")
    print(f"  Transport work")
    print(f"  {'-'*60}")
    print(f"  Remaining cargo fraction : {case['remaining_fraction']:.2%}")
    print(f"  Transport work (actual)  : {case['total_transport_work']:,.0f} unit·nm/year")
    print(f"  Transport work (max)     : {case['max_transport_work']:,.0f} unit·nm/year")
    print(f"  Cost per transport work  : {case['cost_per_transport_work']:.4f} USD/unit·nm")

    print(f"\n  {'-'*60}")
    print(f"  Cost breakdown (USD/year)")
    print(f"  {'-'*60}")
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
    optimal_result  = solve_shipping_model(route_length, df_vessels)
    battery_result  = solve_shipping_model(route_length, df_vessels, force_carrier='battery')
    hydrogen_result = solve_shipping_model(route_length, df_vessels, force_carrier='hydrogen')
    if optimal_result:  optimal_results.append(optimal_result)
    if battery_result:  battery_results.append(battery_result)
    if hydrogen_result: hydrogen_results.append(hydrogen_result)

df_optimal  = pd.DataFrame(optimal_results)
df_battery  = pd.DataFrame(battery_results)
df_hydrogen = pd.DataFrame(hydrogen_results)

# =============================================================================
# SENSITIVITY - energy prices
# =============================================================================

bat_prices = range(20, 201, 20)
hyd_prices = range(20, 301, 20)
price_results = []
for bp in bat_prices:
    for hp in hyd_prices:
        r = solve_shipping_model(340, df_vessels, params={
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
        'cost_port':        {e: base_port[e]  * scale for e in ['battery', 'hydrogen']},
        'cost_plant_power': {e: base_plant[e] * scale for e in ['battery', 'hydrogen']},
    }
    r_bat = solve_shipping_model(270, df_vessels, params=_p, force_carrier='battery')
    r_hyd = solve_shipping_model(270, df_vessels, params=_p, force_carrier='hydrogen')
    if r_bat: infra_bat_results.append({**r_bat, 'infra_scale': round(scale, 2)})
    if r_hyd: infra_hyd_results.append({**r_hyd, 'infra_scale': round(scale, 2)})
df_infra_bat = pd.DataFrame(infra_bat_results)
df_infra_hyd = pd.DataFrame(infra_hyd_results)

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
# PLOT - Cost share at 270 nm: Battery vs Hydrogen
# =============================================================================

pie_costs  = ['cost_system', 'cost_plant', 'cost_port', 'cost_production', 'cost_distribution', 'cost_lo']
pie_labels = ['Onboard system', 'Plant investment', 'Port infrastructure', 'Production', 'Distribution', 'Lost opportunity']
pie_colors = [COLORS[c] for c in pie_costs]

battery_270  = solve_shipping_model(270, df_vessels, force_carrier='battery')
hydrogen_270 = solve_shipping_model(270, df_vessels, force_carrier='hydrogen')

fig_cost_share, (ax_battery, ax_hydrogen) = plt.subplots(1, 2, figsize=(11, 5))
for ax, result, carrier, plot_title in [
    (ax_battery,  battery_270,  'battery',  'Battery'),
    (ax_hydrogen, hydrogen_270, 'hydrogen', 'Hydrogen'),
]:
    values = [result[c] for c in pie_costs]
    wedges, _, _ = ax.pie(
        values, labels=None, colors=pie_colors,
        autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
        startangle=90, pctdistance=0.75,
    )
    ax.set_title(f'{plot_title}\nTotal: {result["total_cost"]:,.0f} USD/year',
                fontsize=10, color=COLORS[carrier])
fig_cost_share.legend(wedges, pie_labels, loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.01))
fig_cost_share.tight_layout(rect=[0, 0.08, 1, 1])
fig_cost_share.savefig(f'{OUT_DIR}cost_shares_270nm.png', dpi=300)

# =============================================================================
# PLOT - Technology crossover
# =============================================================================

fig_crossover, ax_crossover = plt.subplots(figsize=(9, 5))
ax_crossover.plot(df_battery['length'],  df_battery['total_cost'],  color=COLORS['battery'],  linewidth=2, label='Battery')
ax_crossover.plot(df_hydrogen['length'], df_hydrogen['total_cost'], color=COLORS['hydrogen'], linewidth=2, label='Hydrogen')

crossover_length = None
df_co = pd.merge(df_battery[['length', 'total_cost']], df_hydrogen[['length', 'total_cost']],
                on='length', suffixes=('_battery', '_hydrogen'))
bat_costs = df_co['total_cost_battery'].values
hyd_costs = df_co['total_cost_hydrogen'].values
len_arr   = df_co['length'].values

for i in range(1, len(bat_costs)):
    prev = bat_costs[i-1] - hyd_costs[i-1]
    curr = bat_costs[i]   - hyd_costs[i]
    if prev * curr < 0:
        frac = prev / (prev - curr)
        crossover_length = len_arr[i-1] + frac * (len_arr[i] - len_arr[i-1])
        break

min_cost = min(bat_costs.min(), hyd_costs.min())
max_cost = max(bat_costs.max(), hyd_costs.max())

if crossover_length is not None:
    crossover_cost = np.interp(crossover_length, df_battery['length'], df_battery['total_cost'])
    ax_crossover.axvline(crossover_length, color='gray', linestyle='--', linewidth=1.0)
    ax_crossover.scatter(crossover_length, crossover_cost, color='black', s=30, zorder=5)
    ax_crossover.text(crossover_length + 10, crossover_cost * 1.08, f'{crossover_length:.0f} nm', ha='left', va='bottom', fontsize=8, color='gray')

ax_crossover.set_xlabel('One-way route length (nm)')
ax_crossover.set_ylabel('Total cost per year (USD)')
ax_crossover.yaxis.set_major_formatter(fmt_usd)
ax_crossover.legend(fontsize=8)
ax_crossover.grid(linestyle='--', alpha=0.5)
fig_crossover.tight_layout()
fig_crossover.savefig(f'{OUT_DIR}technology_crossover.png', dpi=300)

# =============================================================================
# PLOT - Cost breakdown stacked area: Battery (top) vs Hydrogen (bottom)
# =============================================================================

cost_components       = ['cost_system', 'cost_plant', 'cost_port', 'cost_production', 'cost_distribution', 'cost_lo']
cost_component_labels = ['Onboard system', 'Plant investment', 'Port infrastructure', 'Production', 'Distribution', 'Lost opportunity']

fig_cost_breakdown, (ax_battery, ax_hydrogen) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, sharey=True)
for ax, df_results, carrier in [(ax_battery, df_battery, 'battery'), (ax_hydrogen, df_hydrogen, 'hydrogen')]:
    component_values = [df_results[c].values for c in cost_components]
    component_colors = [COLORS[c] for c in cost_components]
    ax.stackplot(df_results['length'], component_values, labels=cost_component_labels,
                colors=component_colors, alpha=0.85)
    ax.set_ylabel('Cost per year (USD)')
    ax.yaxis.set_major_formatter(fmt_usd)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_facecolor('white')
    ax.text(0.02, 0.97, carrier.capitalize(), transform=ax.transAxes,
            va='top', fontsize=10, color=COLORS[carrier])
ax_battery.legend(fontsize=8, loc='upper right')
ax_hydrogen.set_xlabel('One-way route length (nm)')
fig_cost_breakdown.tight_layout()
fig_cost_breakdown.savefig(f'{OUT_DIR}cost_breakdown.png', dpi=300)

# =============================================================================
# PLOT - Well-to-wake energy stacked area: Battery (top) vs Hydrogen (bottom)
# =============================================================================

energy_components       = ['E_wake_MWh', 'L_conv_MWh', 'L_dist_MWh', 'L_prod_MWh']
energy_component_labels = ['Useful propulsion', 'Conversion loss', 'Distribution loss', 'Production loss']
energy_component_colors = ['#42A5F5', '#FF7043', '#FFA726', '#AB47BC']

fig_energy_breakdown, (ax_battery, ax_hydrogen) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, sharey=True)
for ax, df_results, carrier in [(ax_battery, df_battery, 'battery'), (ax_hydrogen, df_hydrogen, 'hydrogen')]:
    energy_values = [df_results[c].values for c in energy_components]
    ax.stackplot(df_results['length'], energy_values, labels=energy_component_labels,
                colors=energy_component_colors, alpha=0.85)
    ax.set_ylabel('Primary energy per year (MWh)')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.text(0.02, 0.97, carrier.capitalize(), transform=ax.transAxes,
            va='top', fontsize=10, color=COLORS[carrier])
ax_battery.legend(fontsize=8, loc='upper right')
ax_hydrogen.set_xlabel('One-way route length (nm)')
fig_energy_breakdown.tight_layout()
fig_energy_breakdown.savefig(f'{OUT_DIR}WtW_breakdown.png', dpi=300)

# =============================================================================
# TORNADO SENSITIVITY PLOT - ±50% on all cost parameters, 270 nm base case
# =============================================================================

base_route_length    = 270
base_battery_result  = solve_shipping_model(base_route_length, df_vessels, force_carrier='battery')
base_hydrogen_result = solve_shipping_model(base_route_length, df_vessels, force_carrier='hydrogen')
base_battery_cost    = base_battery_result['total_cost']
base_hydrogen_cost   = base_hydrogen_result['total_cost']

sensitivity_parameters = [
    ('Production cost',     'cost_production',   True),
    ('Onboard system cost', 'cost_storage',      True),
    ('Port cost',           'cost_port',         True),
    ('Plant investment',    'cost_plant_power',  True),
    ('Distribution cost',   'cost_distribution', True),
    ('LO volume penalty',   'cost_LO_V',         False),
    ('LO weight penalty',   'cost_LO_W',         False),
]

def scale_parameter(parameter_key, scale_factor, is_carrier_specific):
    base = DEFAULT_PARAMS[parameter_key]
    if is_carrier_specific:
        return {parameter_key: {c: v * scale_factor for c, v in base.items()}}
    return {parameter_key: base * scale_factor}

sensitivity_results = []
for parameter_label, parameter_key, is_carrier_specific in sensitivity_parameters:
    for scale_factor, sensitivity_case in [(0.5, 'low'), (1.5, 'high')]:
        mp   = scale_parameter(parameter_key, scale_factor, is_carrier_specific)
        bat_r = solve_shipping_model(base_route_length, df_vessels, params=mp, force_carrier='battery')
        hyd_r = solve_shipping_model(base_route_length, df_vessels, params=mp, force_carrier='hydrogen')
        sensitivity_results.append({
            'parameter_label':    parameter_label,
            'sensitivity_case':   sensitivity_case,
            'delta_battery_pct':  (100 * (bat_r['total_cost'] - base_battery_cost) / base_battery_cost  if bat_r else float('nan')),
            'delta_hydrogen_pct': (100 * (hyd_r['total_cost'] - base_hydrogen_cost) / base_hydrogen_cost if hyd_r else float('nan')),
        })

df_tornado = pd.DataFrame(sensitivity_results)
battery_cost_swing = (df_tornado.groupby('parameter_label')['delta_battery_pct']
                    .apply(lambda v: v.max() - v.min()).sort_values(ascending=True))
sorted_parameter_labels = battery_cost_swing.index.tolist()

fig_tornado, (ax_battery, ax_hydrogen) = plt.subplots(2, 1, figsize=(10, 10), sharey=True, sharex=True)
for ax, carrier, delta_col in [
    (ax_battery,  'battery',  'delta_battery_pct'),
    (ax_hydrogen, 'hydrogen', 'delta_hydrogen_pct'),
]:
    for pi, pl in enumerate(sorted_parameter_labels):
        pr   = df_tornado[df_tornado['parameter_label'] == pl]
        low  = pr.loc[pr['sensitivity_case'] == 'low',  delta_col].values[0]
        high = pr.loc[pr['sensitivity_case'] == 'high', delta_col].values[0]
        ax.barh(pi, low,  color=COLORS[carrier], alpha=0.5, height=0.5)
        ax.barh(pi, high, color=COLORS[carrier], alpha=0.9, height=0.5)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.text(0.02, 0.97, carrier.capitalize(), transform=ax.transAxes,
            va='top', fontsize=10, color=COLORS[carrier])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:+.0f}%'))
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
ax_hydrogen.set_xlabel('Change in total cost relative to base case (%)')
for ax in (ax_battery, ax_hydrogen):
    ax.set_yticks(range(len(sorted_parameter_labels)))
    ax.set_yticklabels(sorted_parameter_labels)
fig_tornado.tight_layout()
fig_tornado.savefig(f'{OUT_DIR}tornado_sensitivity.png', dpi=300)

# =============================================================================
# PLOT - Transport work cost sensitivity
# =============================================================================

fig_transport_work_cost, ax_transport_work_cost = plt.subplots(figsize=(8, 5))

ax_transport_work_cost.plot(df_battery['length'], df_battery['cost_per_transport_work'],
                            color=COLORS['battery'], linewidth=2, label='Battery (forced)')
ax_transport_work_cost.plot(df_hydrogen['length'], df_hydrogen['cost_per_transport_work'],
                            color=COLORS['hydrogen'], linewidth=2, label='Hydrogen (forced)')

# Find crossover where battery becomes more expensive per transport work than hydrogen
df_transport_work = df_battery[['length', 'cost_per_transport_work']].merge(
    df_hydrogen[['length', 'cost_per_transport_work']],
    on='length', suffixes=('_battery', '_hydrogen'))

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
    crossover_cost = np.interp(transport_work_crossover, df_battery['length'],
                               df_battery['cost_per_transport_work'])
    ax_transport_work_cost.axvline(transport_work_crossover, color='gray', linestyle='--', linewidth=1.0)
    ax_transport_work_cost.scatter(transport_work_crossover, crossover_cost,
                                   color='black', s=30, zorder=5)
    ax_transport_work_cost.text(transport_work_crossover + 10, crossover_cost * 1.02,
                                f'{transport_work_crossover:.0f} nm',
                                va='bottom', ha='left', fontsize=8, color='gray')

ax_transport_work_cost.set_xlim(left=50, right=800)
ax_transport_work_cost.set_xlabel('One-way route length (nm)')
ax_transport_work_cost.set_ylabel('Cost per transport work (USD/unit·nm)')
ax_transport_work_cost.grid(True, linestyle='--', alpha=0.4)
ax_transport_work_cost.legend(fontsize=8)

fig_transport_work_cost.tight_layout()
fig_transport_work_cost.savefig(f'{OUT_DIR}plot_cost_per_transport_work.png', dpi=300)

print('\nAll plots saved.')