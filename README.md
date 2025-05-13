# Powerline Takehome Project

This repository contains a takehome interview project for Powerline. The goal of the project is to create anoptimization model for an energy storage site in AU

## Requirements

- Python 3.8+
- Required libraries (see `requirements.txt`)

## Setup


1. Put .xlsx data file in the /data folder and call it aemo_prices_20231228.xlsx

2. Install dependencies (ideally do this in a python virtual env you've made):
    ```bash
    pip install -r requirements.txt
    ```

3. Run the optimization
    ```bash
    python run_optimizer.py
    ```

4. Investigate results in the /outputs directory

## Modelling Decisions
The main decision I have gotten stuck on in the time I have is how to linearly model the efficiency losses in the case when ramping causes the battery to flip from charging to discharging in the middle of an interval. The ouput power affects SoC differently in the case of charging and discharging. With charging, the SoC change is power * efficiency_loss and with discharging it is power / efficiency_loss. I haven't yet been able to figure out how to do this piecewise function in a linear way and that is what I would work on next with a bit more time.

A couple other things I have included are a cycle per day limit (right now set to 5 so it makes no impact, but could be adjusted). I've included all of the inputs about the battery and problem into config files (battery and policy config json files).

## Testing

## Price-Maker Modeling Task
In the case where we are not just selecting awards, but instead interacting with the market in a bidirectional way, I would want to adjust the objective function. Instead of having the objective function be:
$$
\sum_{i=1}^{t} P_i \cdot \pi_i
$$
where $P_i$ is the price at time $i$ and $\pi_i$ is the power at time $i$,
We would want $P_i$ and $\pi_i$ to be functions of the submitted bid curve. Let $\vec{B_i}$ be the bids at time $i$. Then we want the objective to be
$$
\sum_{i=1}^{t} f_{Pi}(\vec{B_i}) \cdot f_{\pi i}(\vec{B_i})
$$