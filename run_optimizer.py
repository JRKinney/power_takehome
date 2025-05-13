import json
import pandas as pd

# from optimizer import optimizer
from optimizer import optimizer
from data.format_data import format_data, clean_data
from visualizations import price_chart, plot_awards, plot_avg_net_power, plot_soc, plot_cumulative_pnl, plot_discharge_cycles
from schemas import BatteryScheduleSchema

if __name__ == "__main__":
    # Run the optimizer
    battery_config = json.load(open("battery_config.json"))
    policy_config = json.load(open("policy_config.json"))
    formatted_data_path = format_data('aemo_prices_20231228.xlsx')
    cleaned_data_path = clean_data(formatted_data_path)
    input_data = pd.read_parquet(cleaned_data_path)
    revenue, cycles = optimizer(battery_config, policy_config, input_data)

    output_df = BatteryScheduleSchema.validate(pd.read_csv('outputs/optimization_schedule.csv'))
    summary = {'Total profit (AUD)': revenue, 'Total number of cycles': cycles}
    json.dump(summary, open('outputs/summary.json', 'w'))

    price_chart_fig = price_chart(output_df)
    awards_fig = plot_awards(output_df)
    power_fig = plot_avg_net_power(output_df)
    soc_fig = plot_soc(output_df)
    cumulative_fig = plot_cumulative_pnl(output_df)
    discharge_fig = plot_discharge_cycles(output_df)

    price_chart_fig.savefig("outputs/prices.png", dpi=300, bbox_inches='tight')
    awards_fig.savefig("outputs/awards.png", dpi=300, bbox_inches='tight')
    power_fig.savefig("outputs/power.png", dpi=300, bbox_inches='tight')
    soc_fig.savefig("outputs/soc.png", dpi=300, bbox_inches='tight')
    cumulative_fig.savefig("outputs/cumulative.png", dpi=300, bbox_inches='tight')
    discharge_fig.savefig("outputs/discharge.png", dpi=300, bbox_inches='tight')
