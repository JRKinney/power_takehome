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
    net_power_award = discharge_mw_award - charge_mw_award
    prev_net_power_award = cp.hstack([0, net_power_award[:-1]])
    net_power_output = (net_power_award - prev_net_power_award) / 2
    net_energy = net_power_output * interval_length_hours

    energy_soc_impact = net_energy / η_d
    reg_up_soc_impact = -1 * interval_length_hours * regup_throughput * regup_mw / η_d
    reg_down_soc_impact = -1 * interval_length_hours * regdown_throughput * regdown_mw * η_c

    constraints.append(soc[1:] == soc[:-1] + energy_soc_impact + reg_up_soc_impact + reg_down_soc_impact)

    # State of charge limits
    constraints.append(soc[:num_intervals] >= 0)
    constraints.append(soc[:num_intervals] <= battery_energy_max_mwh)

    # Limit the battery to 5 cycles
    e_discharge = cp.maximum(energy_soc_impact + reg_up_soc_impact + reg_down_soc_impact, 0)
    e_charge = cp.minimum(energy_soc_impact + reg_up_soc_impact + reg_down_soc_impact, 0)
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
        verbose=True,
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
            "energy_charged": e_charge.value,
            "energy_discharged_cycles": e_discharge.value / battery_energy_max_mwh,
        }
    )

    validated_schedule = BatteryScheduleSchema.validate(schedule)
    validated_schedule.to_csv("outputs/optimization_schedule.csv", index=False)
    logger.info("Schedule saved to outputs/optimization_schedule.csv")

    return revenue.value, np.sum(e_discharge.value / battery_energy_max_mwh)
