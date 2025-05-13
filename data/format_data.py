import pandas as pd
from loguru import logger
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def format_data(filename: str = 'aemo_prices_20231228.xlsx') -> Path:
    input_data_path = PROJECT_ROOT / 'data' / 'aemo_prices_20231228.xlsx'
    output_data_path = PROJECT_ROOT / 'data' / 'aemo_prices_20231228.parquet'

    df = pd.read_excel(input_data_path, sheet_name='data')
    # Type conversions
    df["end_datetime"] = pd.to_datetime(df["end_datetime"])
    df["start_datetime"] = pd.to_datetime(df["start_datetime"])
    for col in ["price_energy", "price_raisereg", "price_lowerreg"]:
        df[col] = pd.to_numeric(df[col])

    # Parquet to maintain schema
    df.to_parquet(output_data_path, engine="pyarrow", index=False)
    logger.info(f"Data formatted and saved to {output_data_path}")
    return output_data_path

def clean_data(data_path: Path) -> Path:
    """
    Problems to fix based on EDA:
    - From 1600 to 1630, we are missing prices
        Solution: Create a 5 minute time grid and null fill
        with the last known price
    """
    cleaned_suffix = Path(data_path.parts[-1].replace('.parquet', '_cleaned.parquet'))
    cleaned_data_path = Path(*data_path.parts[:-1]) / cleaned_suffix

    df = pd.read_parquet(data_path)
    
    # Create a 5 minute time grid
    time_grid = pd.DataFrame({'start_datetime': pd.date_range(start=df['start_datetime'].min(), 
                                                end=df['end_datetime'].max(), 
                                                freq='5min',
                                                inclusive='left')})
    time_grid['end_datetime'] = time_grid['start_datetime'] + pd.Timedelta(minutes=5)

    # Join the original data with the time grid
    cleaned_df = time_grid.merge(df, on=['start_datetime', 'end_datetime'], how='left')

    # Forward fill the missing values
    cleaned_df['price_energy'] = cleaned_df['price_energy'].ffill()
    cleaned_df['price_raisereg'] = cleaned_df['price_raisereg'].ffill()
    cleaned_df['price_lowerreg'] = cleaned_df['price_lowerreg'].ffill()

    # Parquet to maintain schema
    cleaned_df.to_parquet(cleaned_data_path, engine="pyarrow", index=False)
    logger.info(f"Data cleaned and saved to {cleaned_data_path}")
    return cleaned_data_path


if __name__ == "__main__":
    formatted_data_path = format_data()
    cleaned_data_path = clean_data(formatted_data_path)
