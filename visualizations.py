from schemas import BatteryScheduleDataframe
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandera as pa
import pandas as pd


@pa.check_types
def price_chart(df: BatteryScheduleDataframe) -> Figure:
    """Returns a matplotlib Figure with market price plots."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.step(df['end_datetime'], df['price_energy'], where='post', label='Energy Price')
    ax.step(df['end_datetime'], df['price_raisereg'], where='post', label='RegRaise Price')
    ax.step(df['end_datetime'], df['price_lowerreg'], where='post', label='RegLower Price')

    ax.set_xlabel('End Datetime')
    ax.set_ylabel('Price (AUD/MWh)')
    ax.set_title('Market Prices Over Time (Stepwise)')

    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()

    plt.tight_layout()
    return fig


@pa.check_types
def plot_awards(df: BatteryScheduleDataframe) -> Figure:
    """Returns a matplotlib Figure with award plots (in MW)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    df = df.copy()
    df['energy_mwh_award'] = df['discharge_mwh_award'] - df['charge_mwh_award']

    ax.step(df['end_datetime'], df['energy_mwh_award'], where='post', label='Energy Award (MW)')
    ax.step(df['end_datetime'], df['regdown_mw_award'], where='post', label='RegLower Award (MW)')
    ax.step(df['end_datetime'], df['regup_mw_award'], where='post', label='RegRaise Award (MW)')

    ax.set_xlabel('End Datetime')
    ax.set_ylabel('Award (MW)')
    ax.set_title('Market Awards')

    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()

    plt.tight_layout()
    return fig


@pa.check_types
def plot_avg_net_power(df: BatteryScheduleDataframe) -> Figure:
    """Returns a matplotlib Figure showing average net power over time."""
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.step(df['end_datetime'], df['avg_net_power'], where='post', label='Average Power (MW)', color='tab:blue')

    ax.set_xlabel('End Datetime')
    ax.set_ylabel('Average Power (MW)')
    ax.set_title('Average Net Power Over Time')

    ax.grid(True)
    fig.autofmt_xdate()

    plt.tight_layout()
    return fig


@pa.check_types
def plot_soc(df: BatteryScheduleDataframe) -> Figure:
    """Returns a matplotlib Figure showing state of charge (MWh) over time."""
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.step(df['end_datetime'], df['soc_mwh'], where='post', label='Energy (MWh)', color='tab:green')

    ax.set_xlabel('End Datetime')
    ax.set_ylabel('Energy (MWh)')
    ax.set_title('State of Charge Over Time')

    ax.grid(True)
    fig.autofmt_xdate()

    plt.tight_layout()
    return fig

@pa.check_types
def plot_cumulative_pnl(df: BatteryScheduleDataframe) -> Figure:
    """Returns a matplotlib Figure with cumulative PnL."""
    fig, ax = plt.subplots(figsize=(12, 6))

    df = df.copy()
    df['cost'] = df['energy_pnl'].apply(lambda x: min(x, 0)) * -1
    df['revenue'] = df['energy_pnl'].apply(lambda x: max(x, 0)) + df['regup_revenue'] + df['regdown_revenue']
    df['profit'] = df['energy_pnl'] + df['regup_revenue'] + df['regdown_revenue']

    df['cum_cost'] = df['cost'].cumsum()
    df['cum_revenue'] = df['revenue'].cumsum()
    df['cum_profit'] = df['profit'].cumsum()

    ax.step(df['end_datetime'], df['cum_cost'], where='post', label='Cumulative Cost (AUD)')
    ax.step(df['end_datetime'], df['cum_revenue'], where='post', label='Cumulative Revenue (AUD)')
    ax.step(df['end_datetime'], df['cum_profit'], where='post', label='Cumulative Profit (AUD)')

    ax.set_xlabel('End Datetime')
    ax.set_ylabel('Cumulative $AUD')
    ax.set_title('Cumulative Profit and Loss')

    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()

    plt.tight_layout()
    return fig

@pa.check_types
def plot_discharge_cycles(df: BatteryScheduleDataframe) -> Figure:
    """Returns a matplotlib Figure with discharge cycles."""
    fig, ax = plt.subplots(figsize=(12, 6))

    df = df.copy()
    df['cum_cycles'] = df['energy_discharged_cycles'].cumsum()

    ax.step(df['end_datetime'], df['cum_cycles'], where='post', label='Discharge Cycles')

    ax.set_xlabel('End Datetime')
    ax.set_ylabel('Cumulative $AUD')
    ax.set_title('Cumulative Profit and Loss')

    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()

    plt.tight_layout()
    return fig