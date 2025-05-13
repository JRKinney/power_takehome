import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema
from typing import TypeVar, Annotated

DataFrame = TypeVar("DataFrame", bound=pd.DataFrame)

BatteryScheduleSchema = DataFrameSchema(
    {
        "start_datetime": Column(
            pa.dtypes.DateTime,
            nullable=False,
            coerce=True,
        ),
        "end_datetime": Column(
            pa.dtypes.DateTime,
            nullable=False,
            coerce=True,
        ),
        "price_energy": Column(
            float,
            nullable=False,
        ),
        "price_raisereg": Column(
            float,
            nullable=False,
        ),
        "price_lowerreg": Column(
            float,
            nullable=False,
        ),
        "charge_mwh_award": Column(
            float,
            nullable=False,
        ),
        "discharge_mwh_award": Column(
            float,
            nullable=False,
        ),
        "regdown_mw_award": Column(
            float,
            nullable=False,
        ),
        "regup_mw_award": Column(
            float,
            nullable=False,
        ),
        "avg_net_power": Column(
            float,
            nullable=False,
        ),
        "soc_mwh": Column(
            float,
            nullable=False,
        ),
        "energy_pnl": Column(
            float,
            nullable=False,
        ),
        "regup_revenue": Column(
            float,
            nullable=False,
        ),
        "regdown_revenue": Column(
            float,
            nullable=False,
        ),
        # "energy_discharged": Column(
        #     float,
        #     nullable=False,
        # ),
        # "energy_discharged_cycles": Column(
        #     float,
        #     nullable=False,
        # ),
        # "energy_charged": Column(
        #     float,
        #     nullable=False,
        # ),
    }
)

BatteryScheduleDataframe = Annotated[DataFrame, BatteryScheduleSchema]

InputDataSchema = DataFrameSchema(
    {
        "start_datetime": Column(
            pa.dtypes.DateTime,
            nullable=False,
            coerce=True,
        ),
        "end_datetime": Column(
            pa.dtypes.DateTime,
            nullable=False,
            coerce=True,
        ),
        "price_energy": Column(
            float,
            nullable=False,
        ),
        "price_raisereg": Column(
            float,
            nullable=False,
        ),
        "price_lowerreg": Column(
            float,
            nullable=False,
        ),
    }
)

InputDataframe = Annotated[DataFrame, InputDataSchema]
