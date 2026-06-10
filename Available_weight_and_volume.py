vessels = {
    "sea_cargo_express": {
        "MCR": 4500,
        "RPM_MAIN": 750,
        "P_AE": 1560,
        "RPM_AE": 1500,
        "K": 0.69,
        "RHO_ER": 0.13,
    },
        "trans_sol": {
        "MCR": 15000,      # 2 x 7500 kW Wärtsilä 6L46F
        "RPM_MAIN": 600,   # MCR at 600 rpm
        "P_AE": 1035,      # 3 x 345 kW auxiliary engines
        "RPM_AE": 1500,    
        "K": 0.69,
        "RHO_ER": 0.13,
    },

        "humbria_seaway": {
        "MCR": 23600,       # 2 x 11800 kW
        "RPM_MAIN": 117,    # main engine at MCR
        "P_AE": 3960,       # 2 x 1980 kW auxiliary engines
        "RPM_AE": 1000,     # assumed if AE rpm is not specified
        "K": 0.69,
        "RHO_ER": 0.16,     # slow-speed diesel
    }
}

def calculate_available_capacity(vessel):

    W_d = 12 * (vessel["MCR"] / vessel["RPM_MAIN"]) ** 0.84         # dry weight of the main engine
    W_r = vessel["K"] * (vessel["MCR"] ** 0.70)                     # remaining machinery weight
    W_AE = 24.141 * (vessel["P_AE"] / vessel["RPM_AE"]) ** 0.6901   # auxiliary engine weight
    

    W_m = W_d + W_r + W_AE                                          # total machinery weight
    W_A = W_m                                                       # available weight capacity after removing conventional machinery
    V_ER = W_m / vessel["RHO_ER"]                                   # estimated engine room volume
    V_A = 0.5 * V_ER                                                # available machinery-space volume usable for alternative energy systems

    return {
        "W_d_ton": W_d,
        "W_r_ton": W_r,
        "W_AE_ton": W_AE,
        "W_m_ton": W_m,
        "W_A_ton": W_A,
        "V_ER_m3": V_ER,
        "V_A_m3": V_A
    }

for name, vessel in vessels.items():

    available_capacity = calculate_available_capacity(vessel)

    print("\n" + "="*60)
    print(name.upper())
    print("="*60)

    for key, value in available_capacity.items():
        print(f"{key:<15}: {value:>10.2f}")