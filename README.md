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
In the case where we are not just selecting awards, but instead interacting with the market in a bidirectional way, I would want to adjust the objective function. Instead of having the objective function be (simplifying to just talk about energy for a second):
$$
\sum_{i=1}^{t} P_i \cdot \pi_i
$$
where $P_i$ is the price at time $i$ and $\pi_i$ is the power at time $i$,
We would want $P_i$ and $\pi_i$ to be functions of the submitted bid curve. Let $\vec{B_i}$ be the bids at time $i$. Then we want the objective to be
$$
\sum_{i=1}^{t} f_{Pi}(\vec{B_i}) \cdot f_{\pi i}(\vec{B_i})
$$
where $f_{Pi}$ and $f_{\pi i}$ are functions which take in the bid curve and return, respectively, a clearing price and an award power for that interval. We would want to deduce these functions based on the bids in the market at that interval. For the simplest case where we self-schedule (bid a high price) a single MW, we would expect the relationship to match that of the previous objective function because the clearing price will not be affected. As our bids cause $\pi_i$ to increase, we would expect $P_i$ to decrease.

The full objective would include a similar logic for the regulation markets as well
$$
\sum_{i=1}^{t} f^e_{Pi}(\vec{B^{e}_{i}}) \cdot f^{e}_{\pi {i}}(\vec{B^{e}_{i}}) + f^{rr}_{Pi}(\vec{B^{rr}_{i}}) \cdot f^{rr}_{\pi {i}}(\vec{B^{rr}_{i}}) + f^{rl}_{Pi}(\vec{B^{rl}_i}) \cdot f^{rl}_{\pi i}(\vec{B^{rl}_i})
$$
where $rr$ and $rl$ superscripts indicate the applicability to regraise and reglower respectively
In our example day of 12/28, this would likely play out as lowering the peak energy price in the evening and lowering the regulation raise price just prior to that spike because we bid large amounts in both of those intervals. This would be especially impactful in the regulation market as that is significantly smaller than the energy market.
