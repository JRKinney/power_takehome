import pandas as pd
import numpy as np
import cvxpy as cp
import pandera as pa
from loguru import logger

from schemas import BatteryScheduleSchema, InputDataframe


@pa.check_types
def optimizer(
    battery_config: dict,
    policy_config: dict,
    data: InputDataframe,
    dryrun: bool = False,
    verbose: bool = True,
    debug: bool = False,
) -> tuple[float, float]:
    """
    Optimizes the economic operation and carbon abatement of a battery system and solar PPA

    Args:
        battery_config (dict): Dictionary containing battery parameters.
        policy_config (dict): Dictionary containing policy parameters.
        data: InputDataframe: DataFrame containing the input data.

    Returns:
        None
    """
    logger.info("Starting optimization")
    num_intervals = len(data)
    interval_length_hours = 5 / 60

    # Unpack battery config
    battery_power_max_mw = battery_config["battery_power_max_mw"]
    battery_energy_max_mwh = battery_config["battery_energy_max_mwh"]
    η_c = battery_config["charge_eff"]
    η_d = battery_config["discharge_eff"]

    # Unpack policy config
    starting_soc = policy_config["starting_soc_percentage"] * battery_energy_max_mwh
    regup_throughput = policy_config["regraise_throughput"]
    regdown_throughput = policy_config["reglower_throughput"]
    cycle_limit = policy_config["cycle_limit"]

    time = data["start_datetime"]
    energy_prices = data["price_energy"].values
    regup_prices = data["price_raisereg"].values
    regdown_prices = data["price_lowerreg"].values

    # Big-M value for constraints (max power value we can see)
    M = battery_power_max_mw / η_d
    epsilon = 1e-6

    # Create the Decision variables
    # Battery awards (MW) per 5 minute interval
    discharge_mw_award = cp.Variable(num_intervals, nonneg=True, name="discharge_mw_award")
    charge_mw_award = cp.Variable(num_intervals, nonneg=True, name="charge_mw_award")
    net_power_award = cp.Variable(num_intervals, nonneg=True, name="net_power_award")
    regdown_mw = cp.Variable(num_intervals, nonneg=True, name="regdown_mw")
    regup_mw = cp.Variable(num_intervals, nonneg=True, name="regup_mw")
    # State of charge (MWh) at each time step (we index 0..t_hours)
    soc = cp.Variable(num_intervals + 1, name="soc")
    energy_soc_impact = cp.Variable(num_intervals, name="energy_soc_impact")
    reg_up_soc_impact = cp.Variable(num_intervals, name="reg_up_soc_impact")
    reg_down_soc_impact = cp.Variable(num_intervals, name="reg_down_soc_impact")
    # For tracking and limiting cycles
    e_discharge = cp.Variable(num_intervals, nonneg=True, name="e_discharge")

    # Variables for ramping calculations
    # 1 if power crosses zero
    y = cp.Variable(num_intervals, boolean=True, name="y")

    # Create contraints
    constraints = []

    # Initial SOC
    constraints.append(soc[0] == starting_soc)

    # We can only charge or discharge at any given time. Gen = True. Note this makes this a MILP
    gen_or_load = cp.Variable(num_intervals, boolean=True, name="gen_or_load")
    constraints.append(discharge_mw_award <= battery_power_max_mw * gen_or_load)
    constraints.append(charge_mw_award <= battery_power_max_mw * (1 - gen_or_load))

    # Even though we know from the simplified problem what the regup throughput is, we should
    # not oversubscribe to the regulation market. Also limit the charge/discharge power
    # to the battery power limits
    constraints.append(discharge_mw_award + regup_mw <= battery_power_max_mw)
    constraints.append(charge_mw_award + regdown_mw <= battery_power_max_mw)

    # SoC changes for charging/discharging in the energy market based on linear ramping rule
    # This works by lagging the charge/discharge power by one time step, calculating the change in power
    # This is complicated because of the discontinuity around 0 where the effiency losses go from
    # causing a gain in SOC to a loss in SOC.
    constraints.append(net_power_award == discharge_mw_award - charge_mw_award)
    prev_net_power_award = cp.hstack([0, net_power_award[:-1]])
    net_power_output = (net_power_award - prev_net_power_award) / 2
    net_energy = net_power_output * interval_length_hours

    # Detect Sign Change in Interval
    # y = 1 if gen_or_load_prev ≠ gen_or_load (power crosses zero)
    gen_or_load_prev = cp.hstack([True, gen_or_load[:-1]])
    discharge_mw_award_prev = cp.hstack([0, discharge_mw_award[:-1]])
    charge_mw_award_prev = cp.hstack([epsilon, charge_mw_award[:-1]])
    constraints.append(y >= gen_or_load - gen_or_load_prev)
    constraints.append(y >= gen_or_load_prev - gen_or_load)
    constraints.append(y <= 2 - gen_or_load_prev - gen_or_load + 2 * y)
    # Cases for last constraint
    # 1 <= 2 - 1 - 0 + 2 = 3
    # 1 <= 2 - 0 - 1 + 2 = 3
    # 0 <= 2 - 1 - 1 + 0 = 0
    # 0 <= 2 - 0 - 0 + 0 = 2

    # LINEARIZE ALPHA CALCULATION
    # We want: alpha = |prev_net_power_award| / (|prev_net_power_award| + |net_power_award) when y=1
    # First, create variables for absolute values
    abs_prev_net_power_award = cp.Variable(num_intervals, nonneg=True, name="abs_prev_net_power_award")
    abs_net_power_award = cp.Variable(num_intervals, nonneg=True, name="abs_net_power_award")

    # Define absolute values
    constraints.append(abs_prev_net_power_award >= -1 * prev_net_power_award)
    constraints.append(abs_prev_net_power_award >= prev_net_power_award)
    constraints.append(abs_net_power_award >= -1 * net_power_award)
    constraints.append(abs_net_power_award >= net_power_award)

    # Now we need to linearize: alpha = abs_prev_net_power_award / (abs_prev_net_power_award + abs_net_power_award)
    # This only applies when y=1 (power crosses zero)
    # When y=1, we need the ratio calculation
    # Use the constraint: alpha * (abs_prev_net_power_award + abs_net_power_award) = abs_prev_net_power_award
    # Linearize using big-M method and discretization because otherwise we are multiplying two variables and thus nonlinear
    # (more points = better approximation)
    num_points = 10
    alpha_points = np.linspace(0, 1, num_points)
    # Create binary variables for selecting alpha points
    # z[i,j] = 1 means interval i selects alpha point j
    z = cp.Variable((num_intervals, num_points), boolean=True, name="z")
    # Each interval must select exactly one point when y=1
    constraints.append(cp.sum(z, axis=1) == y)
    # Add the relationship constraints for each alpha point
    for j, a_val in enumerate(alpha_points):
        # When z[i,j] = 1, enforce: a_val * sum_abs[i] - abs_prev[i] ≈ 0
        # Upper bound (vectorized across all intervals)
        constraints.append(
            a_val * (abs_prev_net_power_award + abs_net_power_award) - abs_prev_net_power_award <= M * (1 - z[:, j])
        )

        # Lower bound (vectorized across all intervals)
        constraints.append(
            a_val * (abs_prev_net_power_award + abs_net_power_award) - abs_prev_net_power_award >= -M * (1 - z[:, j])
        )

    # Add safety constraint to avoid division by zero
    constraints.append(abs_prev_net_power_award + abs_net_power_award >= epsilon)

    # ENERGY CALCULATION BASED ON CHARGING/DISCHARGING STATES
    # Case 1: No sign change (y=0)
    # Using auxiliary variables for each binary*continuous term
    # Note: We calculate both cases, but only one will contribute due to the gen_or_load values

    # Case 1a: Both charging (gen_or_load_prev=0, gen_or_load=0)
    # First, create E_charge_both as a standard expression
    E_charge_both = interval_length_hours * (charge_mw_award_prev + charge_mw_award) / 2 * η_c

    # Create an auxiliary variable for the product (1-gen_or_load) * E_charge_both
    load_times_charge = cp.Variable(num_intervals, nonneg=True, name="load_times_charge")

    # Find a reasonable upper bound on E_charge_both
    charge_bound = interval_length_hours * battery_power_max_mw * η_c

    # The constraints for the product ((1-gen_or_load) * E_charge_both)
    constraints.append(load_times_charge <= charge_bound * (1 - gen_or_load))
    constraints.append(load_times_charge <= E_charge_both)
    constraints.append(load_times_charge >= E_charge_both - charge_bound * gen_or_load)

    # Now use this linearized variable in the energy calculation
    charge_scenario1 = cp.Variable(num_intervals, nonneg=True, name="charge_scenario1")

    # When y=0, enforce: charge_scenario1 = notgen_times_charge
    constraints.append(charge_scenario1 <= charge_bound * (1 - y))
    constraints.append(charge_scenario1 <= load_times_charge)
    constraints.append(charge_scenario1 >= load_times_charge - charge_bound * y)

    # Case 1b: Both discharging (gen_or_load_prev=1, gen_or_load=1)
    # E_discharge_both = -interval_length_hours * (discharge_mw_award_prev + discharge_mw_award) / 2 / η_d
    E_discharge_both = -interval_length_hours * (discharge_mw_award_prev + discharge_mw_award) / 2 / η_d

    # Create an auxiliary variable for the product gen_or_load * (-1) * E_discharge_both
    discharge_scenario1 = cp.Variable(num_intervals, nonneg=True, name="gen_times_discharge")

    # Linearize the product using standard techniques (assuming E_discharge_both is bounded)
    # upper bound on E_discharge_both
    discharge_bound = interval_length_hours * battery_power_max_mw / η_d

    # The constraints for the product (gen_or_load * (-1) * E_discharge_both)
    constraints.append(discharge_scenario1 <= discharge_bound * gen_or_load)
    constraints.append(discharge_scenario1 <= -1 * E_discharge_both)
    constraints.append(discharge_scenario1 >= -1 * E_discharge_both - discharge_bound * (1 - gen_or_load))

    # Case 2: Sign change (y=1)

    # Case 2a: Charging to discharging (gen_or_load_prev=0, gen_or_load=1)
    # First portion (alpha): charging at average rate charge_mw_award_prev/2
    # Second portion (1-alpha): discharging at average rate discharge_mw_award/2
    c_to_d_impacts = []
    # E_c_to_d = interval_length_hours * (
    #     alpha * (charge_mw_award_prev / 2) * η_c
    #     + (1 - alpha) * (-1 * discharge_mw_award / 2) / η_d  # Charging portion  # Discharging portion
    # )
    for j, a_val in enumerate(alpha_points):
        c_to_d_j = interval_length_hours * (
            a_val * (charge_mw_award_prev / 2) * η_c + (1 - a_val) * (-1 * discharge_mw_award / 2) / η_d
        )

        c_to_d_impacts.append(c_to_d_j)

    # Create charging-to-discharging impact variables
    ctd_impact = cp.Variable(num_intervals, name="ctd_impact")  # gen_or_load * E_c_to_d

    # For each discretized alpha value
    for j in range(num_points):
        # When z[i,j]=1, enforce the corresponding impact
        constraints.append(ctd_impact <= battery_power_max_mw * interval_length_hours * gen_or_load + M * (1 - z[:, j]))
        constraints.append(ctd_impact <= c_to_d_impacts[j] + M * (1 - z[:, j]))
        constraints.append(ctd_impact >= c_to_d_impacts[j] - M * (1 - z[:, j]) - M * (1 - gen_or_load))

    # Case 2b: Discharging to charging (gen_or_load_prev=1, gen_or_load=0)
    # First portion (alpha): discharging at average rate discharge_mw_award_prev/2
    # Second portion (1-alpha): charging at average rate charge_mw_award/2
    d_to_c_impacts = []
    # E_d_to_c = interval_length_hours * (
    #     alpha * (-1 * discharge_mw_award_prev / 2) / η_d
    #     + (1 - alpha) * (charge_mw_award / 2) * η_c  # Discharging portion  # Charging portion
    # )
    for j, a_val in enumerate(alpha_points):
        d_to_c_j = interval_length_hours * (
            a_val * (-1 * discharge_mw_award_prev / 2) / η_d + (1 - a_val) * (charge_mw_award / 2) * η_c
        )

        d_to_c_impacts.append(d_to_c_j)

    # Create discharging-to-charging impact variables
    dtc_impact = cp.Variable(num_intervals, name="dtc_impact")  # (1-gen_or_load) * E_d_to_c

    # For each discretized alpha value
    for j in range(num_points):
        # When z[i,j]=1, enforce the corresponding impact
        constraints.append(dtc_impact <= battery_power_max_mw * interval_length_hours * (1 - gen_or_load) + M * (1 - z[:, j]))
        constraints.append(dtc_impact <= d_to_c_impacts[j] + M * (1 - z[:, j]))
        constraints.append(dtc_impact >= d_to_c_impacts[j] - M * (1 - z[:, j]) - M * gen_or_load)

    # Combine all cases - only one will apply based on binary variables
    # No sign change (y=0)
    no_sign_change_impact = cp.Variable(num_intervals, name="no_sign_change_impact")
    constraints.append(no_sign_change_impact == charge_scenario1 + discharge_scenario1)

    # Sign change (y=1)
    sign_change_impact = cp.Variable(num_intervals, nonneg=True, name="sign_change_impact")
    constraints.append(sign_change_impact <= battery_power_max_mw * interval_length_hours * y)
    constraints.append(sign_change_impact <= ctd_impact + dtc_impact)
    constraints.append(sign_change_impact >= ctd_impact + dtc_impact - battery_power_max_mw * interval_length_hours * (1 - y))

    # Final energy_soc_impact
    constraints.append(energy_soc_impact == no_sign_change_impact + sign_change_impact)

    # Regulation SoC impact
    constraints.append(reg_up_soc_impact == -1 * interval_length_hours * regup_throughput * regup_mw / η_d)
    constraints.append(reg_down_soc_impact == -1 * interval_length_hours * regdown_throughput * regdown_mw * η_c)

    constraints.append(soc[1:] == soc[:-1] + energy_soc_impact + reg_up_soc_impact + reg_down_soc_impact)

    # State of charge limits
    constraints.append(soc[:num_intervals] >= 0)
    constraints.append(soc[:num_intervals] <= battery_energy_max_mwh)

    # Limit the battery cycles
    # Calculate discharge amount from the cases 1b, 2a, and 2b

    # For the y=1 scenarios, create expressions based on alpha discretization
    discharge_scenario2a = cp.Variable(num_intervals, nonneg=True, name="discharge_scenario2a")
    discharge_scenario2b = cp.Variable(num_intervals, nonneg=True, name="discharge_scenario2b")

    discharge_scenario2a_values = []
    discharge_scenario2b_values = []

    for j, a_val in enumerate(alpha_points):
        # For scenario 2a: y=1, (1-gen_or_load)=1, using alpha value j
        # Discharging to charging: alpha portion is discharging
        scenario2a_j = interval_length_hours * (a_val * (discharge_mw_award_prev / 2) / η_d)

        # For scenario 2b: y=1, gen_or_load=1, using alpha value j
        # Charging to discharging: (1-alpha) portion is discharging
        scenario2b_j = interval_length_hours * ((1 - a_val) * (discharge_mw_award / 2) / η_d)

        discharge_scenario2a_values.append(scenario2a_j)
        discharge_scenario2b_values.append(scenario2b_j)

    # Now create the constraints for scenario 2a
    # (y=1, (1-gen_or_load)=1, using z to select alpha)
    for j in range(num_points):
        # When y=1, (1-gen_or_load)=1, and z[i,j]=1, enforce this discharge value

        # Upper bound when all conditions are met
        constraints.append(
            discharge_scenario2a
            <= battery_power_max_mw * interval_length_hours * y
            + M * gen_or_load
            + M * (1 - z[:, j])  # Big-M for (1-gen_or_load)=1  # Big-M for z[i,j]=1
        )

        constraints.append(
            discharge_scenario2a
            <= discharge_scenario2a_values[j]
            + M * (1 - y)
            + M * gen_or_load  # Big-M for y=1
            + M * (1 - z[:, j])  # Big-M for (1-gen_or_load)=1  # Big-M for z[i,j]=1
        )

        # Lower bound when all conditions are met
        constraints.append(
            discharge_scenario2a
            >= discharge_scenario2a_values[j]
            - M * (1 - y)
            - M * gen_or_load  # Big-M for y=1
            - M * (1 - z[:, j])  # Big-M for (1-gen_or_load)=1  # Big-M for z[i,j]=1
        )

    # Now create the constraints for scenario 2b
    # (y=1, gen_or_load=1, using z to select alpha)
    for j in range(num_points):
        # When y=1, gen_or_load=1, and z[i,j]=1, enforce this discharge value

        # Upper bound when all conditions are met
        constraints.append(
            discharge_scenario2b
            <= battery_power_max_mw * interval_length_hours * y
            + M * (1 - gen_or_load)
            + M * (1 - z[:, j])  # Big-M for gen_or_load=1  # Big-M for z[i,j]=1
        )

        constraints.append(
            discharge_scenario2b
            <= discharge_scenario2b_values[j]
            + M * (1 - y)
            + M * (1 - gen_or_load)  # Big-M for y=1
            + M * (1 - z[:, j])  # Big-M for gen_or_load=1  # Big-M for z[i,j]=1
        )

        # Lower bound when all conditions are met
        constraints.append(
            discharge_scenario2b
            >= discharge_scenario2b_values[j]
            - M * (1 - y)
            - M * (1 - gen_or_load)  # Big-M for y=1
            - M * (1 - z[:, j])  # Big-M for gen_or_load=1  # Big-M for z[i,j]=1
        )

    # Final e_discharge
    constraints.append(e_discharge == discharge_scenario1 + discharge_scenario2a + discharge_scenario2b)
    constraints.append(cp.sum(e_discharge) <= battery_energy_max_mwh * cycle_limit)

    regup_mwh = regup_mw * interval_length_hours
    regdown_mwh = regdown_mw * interval_length_hours

    energy_revenue = cp.multiply(net_energy, energy_prices)
    regup_revenue = cp.multiply(regup_mwh, regup_prices)
    regdown_revenue = cp.multiply(regdown_mwh, regdown_prices)

    revenue = cp.sum(energy_revenue + regup_revenue + regdown_revenue)
    objective = cp.Maximize(revenue)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver="CBC", verbose=verbose, maximumSeconds=600)

    logger.info(f"Optimal revenue: ${prob.value:,.2f}")

    schedule = pd.DataFrame(
        {
            "start_datetime": time,
            "end_datetime": data["end_datetime"].values,
            "interval_beginning_date": time.dt.date,
            "charge_mwh_award": charge_mw_award.value,
            "discharge_mwh_award": discharge_mw_award.value,
            "regdown_mw_award": regdown_mw.value,
            "regup_mw_award": regup_mw.value,
            "avg_net_power": net_power_output.value,
            "soc_mwh": soc.value[:-1],
            "energy_pnl": energy_revenue.value,
            "regup_revenue": regup_revenue.value,
            "regdown_revenue": regdown_revenue.value,
            "price_energy": energy_prices,
            "price_raisereg": regup_prices,
            "price_lowerreg": regdown_prices,
            "energy_discharged": e_discharge.value,
            "energy_discharged_cycles": e_discharge.value / battery_energy_max_mwh,
        }
    )

    validated_schedule = BatteryScheduleSchema.validate(schedule)
    if dryrun:
        return validated_schedule

    validated_schedule.to_csv("outputs/optimization_schedule.csv", index=False)
    logger.info("Schedule saved to outputs/optimization_schedule.csv")

    if debug:
        additional_schedule = pd.DataFrame(
            {
                "energy_soc_impact": energy_soc_impact.value,
                "reg_up_soc_impact": reg_up_soc_impact.value,
                "reg_down_soc_impact": reg_down_soc_impact.value,
                "y": y.value,
                "gen_or_load": gen_or_load.value,
                "gen_or_load_prev": gen_or_load_prev.value,
                "E_charge_both": E_charge_both.value,
                "E_discharge_both": E_discharge_both.value,
                "abs_prev_net_power_award": abs_prev_net_power_award.value,
                "abs_net_power_award": abs_net_power_award.value,
                "dtc_impact": dtc_impact.value,
                "ctd_impact": ctd_impact.value,
                "discharge_scenario1": discharge_scenario1.value,
                "load_times_charge": load_times_charge.value,
            }
        )
        debug_schedule = pd.concat([validated_schedule, additional_schedule], axis=1)
        debug_schedule.to_csv("outputs/optimization_schedule_debug.csv", index=False)

    return revenue.value, np.sum(e_discharge.value / battery_energy_max_mwh)
