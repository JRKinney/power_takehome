import unittest
import pandas as pd
import json

from optimizer import optimizer


class TestDischarge(unittest.TestCase):
    """Test the assert that the battery discharges at full power at an artificially high price."""

    def test_price_spiking(self):
        input_data = pd.read_parquet("data/aemo_prices_20231228_cleaned.parquet")
        battery_config = json.load(open("battery_config.json"))
        policy_config = json.load(open("policy_config.json"))

        input_data.loc[5, "price_energy"] = 100000
        input_data.loc[10, "price_raisereg"] = 100000
        input_data.loc[15, "price_lowerreg"] = 100000
        schedule = optimizer(battery_config, policy_config, input_data, dryrun=True, verbose=False)

        self.assertAlmostEqual(schedule.loc[5, "discharge_mwh_award"], battery_config["battery_power_max_mw"], places=3)
        self.assertAlmostEqual(schedule.loc[10, "regup_mw_award"], battery_config["battery_power_max_mw"], places=3)
        self.assertAlmostEqual(schedule.loc[15, "regdown_mw_award"], battery_config["battery_power_max_mw"], places=3)

    def test_price_dropping(self):
        input_data = pd.read_parquet("data/aemo_prices_20231228_cleaned.parquet")
        battery_config = json.load(open("battery_config.json"))
        policy_config = json.load(open("policy_config.json"))

        input_data.loc[5, "price_energy"] = -100000
        input_data.loc[10, "price_raisereg"] = 0
        input_data.loc[15, "price_lowerreg"] = 0
        schedule = optimizer(battery_config, policy_config, input_data, dryrun=True, verbose=False)

        self.assertAlmostEqual(schedule.loc[5, "charge_mwh_award"], battery_config["battery_power_max_mw"], places=3)
        self.assertAlmostEqual(schedule.loc[10, "regup_mw_award"], 0, places=3)
        self.assertAlmostEqual(schedule.loc[15, "regdown_mw_award"], 0, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
