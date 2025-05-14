import pandas as pd
import cvxpy as cp
import pandera as pa
from loguru import logger

from schemas import BatteryScheduleSchema, InputDataframe


@pa.check_types
def optimizer(
    battery_config: dict,
    policy_config: dict,
    data: InputDataframe,
    debug: bool = False,
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
    epsilon = 1e-6

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
    # Battery awards (MW) per 5 minute interval
    discharge_mw_award = cp.Variable(num_intervals, nonneg=True)
    charge_mw_award = cp.Variable(num_intervals, nonneg=True)
    regdown_mw = cp.Variable(num_intervals, nonneg=True)
    regup_mw = cp.Variable(num_intervals, nonneg=True)
    # State of charge (MWh) at each time step (we index 0..t_hours)
    soc = cp.Variable(num_intervals + 1)

    # Create contraints
    constraints = []

    # Initial SOC
    constraints.append(soc[0] == starting_soc)

    # We can only charge or discharge at any given time. Gen = True.
    gen_or_load = cp.Variable(num_intervals, boolean=True)
    constraints.append(discharge_mw_award <= battery_power_max_mw * gen_or_load)
    constraints.append(charge_mw_award <= battery_power_max_mw * (1 - gen_or_load))

    # Create a constraint that the battery must always be doing something in order to simplify the
    # soc calculation. This is a bit of a hack, but it works for the given problem.
    constraints.append(discharge_mw_award + charge_mw_award >= epsilon)

    # Even though we know from the simplified problem what the regup throughput is, we should
    # not oversubscribe to the regulation market. Also limit the charge/discharge power
    # to the battery power limits
    constraints.append(discharge_mw_award + regup_mw <= battery_power_max_mw)
    constraints.append(charge_mw_award + regdown_mw <= battery_power_max_mw)

    # SoC changes for charging/discharging in the energy market based on linear ramping rule
    # This works by lagging the charge/discharge power by one time step, calculating the change in power
    # This is complicated because of the discontinuity around 0 where the effiency losses go from
    # causing a gain in SOC to a loss in SOC.
    net_power_award = discharge_mw_award - charge_mw_award
    prev_net_power_award = cp.hstack([0, net_power_award[:-1]])
    net_power_output = (net_power_award - prev_net_power_award) / 2
    gen_or_load_prev = cp.hstack([True, gen_or_load[:-1]])

    # Case when battery starts discharging and ends discharging
    # Here we can ignore the 0 --> discharge and discharge --> 0 and 0 --> cases becayse of the
    # discharge_mw_award + charge_mw_award >= epsilon constraint
    gen_gen = gen_or_load * gen_or_load_prev
    gen_load = gen_or_load * (1 - gen_or_load_prev)
    load_gen = (1 - gen_or_load) * gen_or_load_prev
    load_load = (1 - gen_or_load) * (1 - gen_or_load_prev)

    # Types of intervals
    # Simple cases
    # 1. Discharge to Discharge
    gen_gen_measured_energy = gen_gen * net_power_output * interval_length_hours
    gen_gen_actual_amount_soc_change = (-1) * gen_gen_measured_energy / discharge_efficiency
    # 2. Charge to Charge
    load_load_measured_energy = load_load * net_power_output * interval_length_hours
    load_load_actual_amount_soc_change = (-1) * load_load_measured_energy * charge_efficiency

    # Complex cases: In these, the ramping causes the battery to go from charge to discharge or vice versa within the interval
    power_output_change = net_power_award - prev_net_power_award
    # If the interval began in a charging state, this is the % time spent charging
    # If the interval began in a discharging state, this is the % time spent discharging
    pct_of_interval_in_start_state = (power_output_change - net_power_award) / power_output_change

    # 3. Discharge to Charge (complex case)
    gen_load_gen_side_measured_energy = gen_load * (
        pct_of_interval_in_start_state * prev_net_power_award / 2 * interval_length_hours
    )
    gen_load_gen_side_soc_change = (-1) * gen_load_gen_side_measured_energy / discharge_efficiency
    gen_load_load_side_measured_energy = gen_load * (
        (1 - pct_of_interval_in_start_state) * net_power_award / 2 * interval_length_hours
    )
    gen_load_load_side_soc_change = (-1) * gen_load_load_side_measured_energy * charge_efficiency
    gen_load_actual_amount_soc_change = gen_load_gen_side_soc_change + gen_load_load_side_soc_change
    # 4. Charge to Discharge (complex case)
    load_gen_load_side_measured_energy = load_gen * (
        pct_of_interval_in_start_state * prev_net_power_award / 2 * interval_length_hours
    )
    load_gen_load_side_soc_change = (-1) * load_gen_load_side_measured_energy * charge_efficiency
    load_gen_gen_side_measured_energy = load_gen * (
        (1 - pct_of_interval_in_start_state) * net_power_award / 2 * interval_length_hours
    )
    load_gen_gen_side_soc_change = (-1) * load_gen_gen_side_measured_energy / discharge_efficiency
    load_gen_actual_amount_soc_change = load_gen_gen_side_soc_change + load_gen_load_side_soc_change

    total_soc_change_from_energy = (
        gen_gen_actual_amount_soc_change
        + load_load_actual_amount_soc_change
        + gen_load_actual_amount_soc_change
        + load_gen_actual_amount_soc_change
    )
    total_discharge_measured_energy_from_energy = (
        gen_gen_measured_energy + gen_load_gen_side_measured_energy + load_gen_gen_side_measured_energy
    )
    total_charge_measured_energy_from_energy = (
        load_load_measured_energy + load_gen_load_side_measured_energy + gen_load_load_side_measured_energy
    )

    total_discharge_measured_energy_from_regup = interval_length_hours * regup_throughput * regup_mw
    reg_up_soc_impact = -1 * total_discharge_measured_energy_from_regup / discharge_efficiency
    total_charge_measured_energy_from_regdown = interval_length_hours * regdown_throughput * regdown_mw
    reg_down_soc_impact = -1 * total_charge_measured_energy_from_regdown * charge_efficiency

    total_discharge_measured_energy = total_discharge_measured_energy_from_energy + total_discharge_measured_energy_from_regup
    total_charge_measured_energy = total_charge_measured_energy_from_energy + total_charge_measured_energy_from_regdown

    constraints.append(soc[1:] == soc[:-1] + total_soc_change_from_energy + reg_up_soc_impact + reg_down_soc_impact)

    # State of charge limits
    constraints.append(soc[:num_intervals] >= 0)
    constraints.append(soc[:num_intervals] <= battery_energy_max_mwh)

    net_energy = net_power_output * interval_length_hours
    regup_mwh = regup_mw * interval_length_hours
    regdown_mwh = regdown_mw * interval_length_hours

    energy_revenue = cp.multiply(net_energy, energy_prices)
    regup_revenue = cp.multiply(regup_mwh, regup_prices)
    regdown_revenue = cp.multiply(regdown_mwh, regdown_prices)

    revenue = cp.sum(energy_revenue + regup_revenue + regdown_revenue)
    objective = cp.Maximize(revenue)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GLPK_MI, verbose=True)

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
            "revenue": revenue.value,
            "energy_revenue": energy_revenue.value,
            "regup_revenue": regup_revenue.value,
            "regdown_revenue": regdown_revenue.value,
            "price_energy": energy_prices,
            "price_raisereg": regup_prices,
            "price_lowerreg": regdown_prices,
        }
    )

    if debug:
        additional_schedule = pd.DataFrame(
            {
                "start_datetime": time,
                "net_power_award": net_power_award.value,
                "prev_net_power_award": prev_net_power_award.value,
                "gen_or_load": gen_or_load.value,
                "gen_or_load_prev": gen_or_load_prev.value,
                "gen_gen": gen_gen.value,
                "gen_load": gen_load.value,
                "load_gen": load_gen.value,
                "load_load": load_load.value,
                "gen_gen_measured_energy": gen_gen_measured_energy.value,
                "gen_gen_actual_amount_soc_change": gen_gen_actual_amount_soc_change.value,
                "load_load_measured_energy": load_load_measured_energy.value,
                "load_load_actual_amount_soc_change": load_load_actual_amount_soc_change.value,
                "power_output_change": power_output_change.value,
                "pct_of_interval_in_start_state": pct_of_interval_in_start_state.value,
                "gen_load_gen_side_measured_energy": gen_load_gen_side_measured_energy.value,
                "gen_load_gen_side_soc_change": gen_load_gen_side_soc_change.value,
                "gen_load_load_side_measured_energy": gen_load_load_side_measured_energy.value,
                "gen_load_load_side_soc_change": gen_load_load_side_soc_change.value,
                "gen_load_actual_amount_soc_change": gen_load_actual_amount_soc_change.value,
                "load_gen_load_side_measured_energy": load_gen_load_side_measured_energy.value,
                "load_gen_load_side_soc_change": load_gen_load_side_soc_change.value,
                "load_gen_gen_side_measured_energy": load_gen_gen_side_measured_energy.value,
                "load_gen_gen_side_soc_change": load_gen_gen_side_soc_change.value,
                "load_gen_actual_amount_soc_change": load_gen_actual_amount_soc_change.value,
                "total_soc_change_from_energy": total_soc_change_from_energy.value,
                "total_discharge_measured_energy_from_energy": total_discharge_measured_energy_from_energy.value,
                "total_charge_measured_energy_from_energy": total_charge_measured_energy_from_energy.value,
                "total_discharge_measured_energy_from_regup": total_discharge_measured_energy_from_regup.value,
                "reg_up_soc_impact": reg_up_soc_impact.value,
                "total_charge_measured_energy_from_regdown": total_charge_measured_energy_from_regdown.value,
                "reg_down_soc_impact": reg_down_soc_impact.value,
                "total_discharge_measured_energy": total_discharge_measured_energy.value,
                "total_charge_measured_energy": total_charge_measured_energy.value,
            }
        )
        schedule = schedule.merge(additional_schedule, on="start_datetime", how="left")

    validated_schedule = BatteryScheduleSchema.validate(schedule)
    validated_schedule.to_csv("outputs/optimization_schedule.csv", index=False)
    logger.info("Schedule saved to outputs/optimization_schedule.csv")
