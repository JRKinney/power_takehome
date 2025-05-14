import pandas as pd
import cvxpy as cp
import pandera as pa
from loguru import logger

from schemas import BatteryScheduleSchema, InputDataframe

# Relies on the data having been pre-processed and saved as a parquet file by the format_data.py script


def calculate_energy_soc_impact(
    power: cp.Expression,
    prev_power: cp.Expression,
    interval_length_hours: float,
    charge_efficiency: float,
    discharge_efficiency: float,
) -> cp.Expression:
    """
    The impact to the battery SoC is complicated due to the ramping and efficiency losses. When the
    battery is charging, the efficiency losses mean that the battery input is power * charge_efficiency.
    When the battery is discharging, the efficiency losses mean that the battery output is power / discharge_efficiency.
    When the ramping during an interval crosses 0, we need a piecewise function for SOC impact. That is implemented here.
    """
    power_output_change = power - prev_power  # Delta across the interval of ramping
    # If the interval began in a charging state, this is the time spent charging
    # If the interval began in a discharging state, this is the time spent discharging
    pct_of_interval_in_start_state = (power_output_change - power) / power_output_change

    # Case 1: Both powers are positive (discharging throughout interval)
    # Energy drawn from battery = (5/60) * (P_prev/efficiency + P_current/efficiency)/2
    if (power.value >= 0) and (prev_power.value >= 0):
        return -1 * interval_length_hours * (prev_power + power) / (2 * discharge_efficiency)
    # Case 2: Both powers are negative (charging throughout interval)
    # Energy added to battery = (5/60) * (P_prev*efficiency + P_current*efficiency)/2
    elif (power.value <= 0) and (prev_power.value <= 0):
        return -1 * interval_length_hours * charge_efficiency * (prev_power + power) / 2
    # Case 3: Zero crossing cases
    # For discharge to charge transition (p_prev > 0, p_current < 0)
    elif (prev_power.value > 0) and (power.value < 0):
        return (
            -1
            * interval_length_hours
            * (
                prev_power / 2 * pct_of_interval_in_start_state / (discharge_efficiency)
                + power / 2 * (1 - pct_of_interval_in_start_state) * charge_efficiency
            )
        )
    # For charge to discharge transition (p_prev > 0, p_current < 0)
    else:  # (prev_power.value < 0) and (power.value > 0):
        return (
            -1
            * interval_length_hours
            * (
                prev_power / 2 * pct_of_interval_in_start_state * charge_efficiency
                + power / 2 * (1 - pct_of_interval_in_start_state) / (discharge_efficiency)
            )
        )


@pa.check_types
def optimizer(
    battery_config: dict,
    policy_config: dict,
    data: InputDataframe,
) -> None:
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
    charge_efficiency = battery_config["charge_eff"]
    discharge_efficiency = battery_config["discharge_eff"]

    # Unpack policy config
    starting_soc = policy_config["starting_soc_percentage"] * battery_energy_max_mwh
    regup_throughput = policy_config["regraise_throughput"]
    regdown_throughput = policy_config["reglower_throughput"]

    time = data["start_datetime"]
    energy_prices = data["price_energy"].values
    regup_prices = data["price_raisereg"].values
    regdown_prices = data["price_lowerreg"].values

    # Create the Decision variables
    # Battery charge and discharge power (MW) per 5 minute interval
    discharge_mw = cp.Variable(num_intervals, nonneg=True)
    charge_mw = cp.Variable(num_intervals, nonneg=True)
    regdown_mw = cp.Variable(num_intervals, nonneg=True)
    regup_mw = cp.Variable(num_intervals, nonneg=True)
    # State of charge (MWh) at each time step (we index 0..t_hours)
    soc = cp.Variable(num_intervals + 1)

    # Create contraints
    constraints = []

    # Initial SOC
    constraints.append(soc[0] == starting_soc)

    # We can only charge or discharge at any given time. Gen = True. Note this makes this a MILP
    gen_or_load = cp.Variable(num_intervals, boolean=True)
    constraints.append(discharge_mw <= battery_power_max_mw * gen_or_load)
    constraints.append(charge_mw <= battery_power_max_mw * (1 - gen_or_load))

    # Even though we know from the simplified problem what the regup throughput is, we should
    # not oversubscribe to the regulation market. Also limit the charge/discharge power
    # to the battery power limits
    constraints.append(discharge_mw + regup_mw <= battery_power_max_mw)
    constraints.append(charge_mw + regdown_mw <= battery_power_max_mw)

    # SoC changes for charging/discharging in the energy market based on linear ramping rule
    # This works by lagging the charge/discharge power by one time step, calculating the change in power
    # This is complicated because of the discontinuity around 0 where the effiency losses go from
    # causing a gain in SOC to a loss in SOC.

    net_power = discharge_mw - charge_mw
    prev_net_power = cp.hstack([0, net_power[:-1]])
    avg_power = (net_power - prev_net_power) / 2

    # For each interval, model the SOC change with ramping and efficiency
    for t in range(num_intervals):
        # Total SOC impact for this interval
        energy_soc_impact = calculate_energy_soc_impact(
            net_power[t], prev_net_power[t], interval_length_hours, charge_efficiency, discharge_efficiency
        )

        # Add regulation impacts
        reg_up_impact = interval_length_hours * regup_throughput * regup_mw[t] / discharge_efficiency
        reg_down_impact = interval_length_hours * regdown_throughput * regdown_mw[t] * charge_efficiency

        # SOC constraint
        constraints.append(soc[t + 1] == soc[t] + energy_soc_impact + reg_up_impact + reg_down_impact)

    # State of charge limits
    constraints.append(soc[:num_intervals] >= 0)
    constraints.append(soc[:num_intervals] <= battery_energy_max_mwh)

    net_energy = avg_power * interval_length_hours
    regup_mwh = regup_mw * interval_length_hours
    regdown_mwh = regdown_mw * interval_length_hours

    revenue = cp.sum(
        cp.multiply(net_energy, energy_prices) + cp.multiply(regup_mwh, regup_prices) + cp.multiply(regdown_mwh, regdown_prices)
    )
    objective = cp.Maximize(revenue)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GLPK_MI, verbose=True)

    logger.info(f"Optimal revenue: ${prob.value:,.2f}")

    schedule = pd.DataFrame(
        {
            "start_datetime": time,
            "end_datetime": data["end_datetime"].values,
            "interval_beginning_date": time.dt.date,
            "charge_mwh_award": charge_mw.value,
            "discharge_mwh_award": discharge_mw.value,
            "regdown_mw_award": regdown_mw.value,
            "regup_mw_award": regup_mw.value,
            "avg_net_power": avg_power.value,
            "soc_mwh": soc.value[:-1],
            "revenue": revenue.value,
            "price_energy": energy_prices,
            "price_raisereg": regup_prices,
            "price_lowerreg": regdown_prices,
        }
    )

    validated_schedule = BatteryScheduleSchema.validate(schedule)
    validated_schedule.to_csv("outputs/optimization_schedule.csv", index=False)
    logger.info("Schedule saved to outputs/optimization_schedule.csv")
