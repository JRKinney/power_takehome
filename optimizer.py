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
    discharge_mw_award = cp.Variable(num_intervals, nonneg=True)
    charge_mw_award = cp.Variable(num_intervals, nonneg=True)
    net_power_award = cp.Variable(num_intervals, nonneg=True)
    regdown_mw = cp.Variable(num_intervals, nonneg=True)
    regup_mw = cp.Variable(num_intervals, nonneg=True)
    # State of charge (MWh) at each time step (we index 0..t_hours)
    soc = cp.Variable(num_intervals + 1)
    energy_soc_impact = cp.Variable(num_intervals)
    reg_up_soc_impact = cp.Variable(num_intervals)
    reg_down_soc_impact = cp.Variable(num_intervals)
    # For tracking and limiting cycles
    e_discharge = cp.Variable(num_intervals, nonneg=True)

    # Variables for ramping calculations
    # 1 if power crosses zero
    y = cp.Variable(num_intervals, boolean=True)
    # Fraction of interval before crossing zero, but discretized in order to maintain linearity
    alpha = cp.Variable(num_intervals, nonneg=True)

    # Create contraints
    constraints = []

    # Initial SOC
    constraints.append(soc[0] == starting_soc)

    # We can only charge or discharge at any given time. Gen = True. Note this makes this a MILP
    gen_or_load = cp.Variable(num_intervals, boolean=True)
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
    abs_prev_net_power_award = cp.Variable(num_intervals, nonneg=True)
    abs_net_power_award = cp.Variable(num_intervals, nonneg=True)

    # Define absolute values
    constraints.append(abs_prev_net_power_award >= -1 * prev_net_power_award)
    constraints.append(abs_prev_net_power_award >= prev_net_power_award)
    constraints.append(abs_net_power_award >= -1 * net_power_award)
    constraints.append(abs_net_power_award >= net_power_award)

    # Now we need to linearize: alpha = abs_prev_net_power_award / (abs_prev_net_power_award + abs_net_power_award)
    # This only applies when y=1 (power crosses zero)

    # When y=0, alpha doesn't matter, but set it to 0
    constraints.append(alpha <= y)

    # When y=1, we need the ratio calculation
    # Use the constraint: alpha * (abs_prev_net_power_award + abs_net_power_award) = abs_prev_net_power_award
    # Linearize using big-M method because otherwise we are multiplying two variables and thus nonlinear

    # The constraint above only applies when y=1
    constraints.append(alpha * (abs_prev_net_power_award + abs_net_power_award) - abs_prev_net_power_award <= M * (1 - y))
    constraints.append(alpha * (abs_prev_net_power_award + abs_net_power_award) - abs_prev_net_power_award >= -M * (1 - y))
    # Add safety constraint to avoid division by zero
    constraints.append(abs_prev_net_power_award + abs_net_power_award >= epsilon)

    # ENERGY CALCULATION BASED ON CHARGING/DISCHARGING STATES
    # Case 1: No sign change (y=0)
    # Note: We calculate both cases, but only one will contribute due to the gen_or_load values

    # Case 1a: Both charging (gen_or_load_prev=0, gen_or_load=0)
    E_charge_both = interval_length_hours * (charge_mw_award_prev + charge_mw_award) / 2 * η_c

    # Case 1b: Both discharging (gen_or_load_prev=1, gen_or_load=1)
    E_discharge_both = -interval_length_hours * (discharge_mw_award_prev + discharge_mw_award) / 2 / η_d

    # Case 2: Sign change (y=1)

    # Case 2a: Charging to discharging (gen_or_load_prev=0, gen_or_load=1)
    # First portion (alpha): charging at average rate charge_mw_award_prev/2
    # Second portion (1-alpha): discharging at average rate discharge_mw_award/2
    E_c_to_d = interval_length_hours * (
        alpha * (charge_mw_award_prev / 2) * η_c
        + (1 - alpha) * (-1 * discharge_mw_award / 2) / η_d  # Charging portion  # Discharging portion
    )

    # Case 2b: Discharging to charging (gen_or_load_prev=1, gen_or_load=0)
    # First portion (alpha): discharging at average rate discharge_mw_award_prev/2
    # Second portion (1-alpha): charging at average rate charge_mw_award/2
    E_d_to_c = interval_length_hours * (
        alpha * (-1 * discharge_mw_award_prev / 2) / η_d
        + (1 - alpha) * (charge_mw_award / 2) * η_c  # Discharging portion  # Charging portion
    )

    # Combine all cases - only one will apply based on binary variables
    constraints.append(
        energy_soc_impact
        == (1 - y) * ((1 - gen_or_load) * E_charge_both + gen_or_load * E_discharge_both)
        + y * ((1 - gen_or_load) * E_d_to_c + gen_or_load * E_c_to_d)
    )

    # Regulation SoC impact
    constraints.append(reg_up_soc_impact == -1 * interval_length_hours * regup_throughput * regup_mw / η_d)
    constraints.append(reg_down_soc_impact == -1 * interval_length_hours * regdown_throughput * regdown_mw * η_c)

    constraints.append(soc[1:] == soc[:-1] + energy_soc_impact + reg_up_soc_impact + reg_down_soc_impact)

    # State of charge limits
    constraints.append(soc[:num_intervals] >= 0)
    constraints.append(soc[:num_intervals] <= battery_energy_max_mwh)

    # Limit the battery cycles
    # Calculate discharge amount from the cases 1b, 2a, and 2b
    constraints.append(
        e_discharge
        == (1 - y) * gen_or_load * -1 * E_discharge_both
        + y  # 1b
        * (
            (1 - gen_or_load) * (alpha * (discharge_mw_award_prev / 2) / η_d)
            + gen_or_load * ((1 - alpha) * (discharge_mw_award / 2) / η_d)  # 2a
        )  # 2b
    )
    constraints.append(cp.sum(e_discharge) <= battery_energy_max_mwh * cycle_limit)

    regup_mwh = regup_mw * interval_length_hours
    regdown_mwh = regdown_mw * interval_length_hours

    energy_revenue = cp.multiply(net_energy, energy_prices)
    regup_revenue = cp.multiply(regup_mwh, regup_prices)
    regdown_revenue = cp.multiply(regdown_mwh, regdown_prices)

    revenue = cp.sum(energy_revenue + regup_revenue + regdown_revenue)
    objective = cp.Maximize(revenue)

    prob = cp.Problem(objective, constraints)
    prob.solve(
        solver=cp.GLPK_MI,
        verbose=verbose,
        glpk={"mipgap": 0.01, "tmlim": 600},  # stop when within 1% of optimal  # time‑limit of 600 seconds
    )

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
                "alpha": alpha.value,
                "gen_or_load": gen_or_load.value,
                "gen_or_load_prev": gen_or_load_prev.value,
                "E_charge_both": E_charge_both.value,
                "E_discharge_both": E_discharge_both.value,
                "E_d_to_c": E_d_to_c.value,
                "E_c_to_d": E_c_to_d.value,
                "abs_prev_net_power_award": abs_prev_net_power_award.value,
                "abs_net_power_award": abs_net_power_award.value,
            }
        )
        debug_schedule = pd.concat([validated_schedule, additional_schedule], axis=1)
        debug_schedule.to_csv("outputs/optimization_schedule_debug.csv", index=False)

    return revenue.value, np.sum(e_discharge.value / battery_energy_max_mwh)
