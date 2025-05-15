# Powerline Takehome Project

This repository contains a takehome interview project for Powerline. The goal of the project is to create anoptimization model for an energy storage site in AU

## Setup


1. Put .xlsx data file in the /data folder and call it `aemo_prices_20231228.xlsx`

2. Install dependencies (ideally do this in a python virtual env you've made):

    **optional**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

    ```bash
    pip install -r requirements.txt
    ```
    ^ These are a bit overloaded, sorry!

3. Run the optimization
    ```bash
    python run_optimizer.py
    ```

4. Investigate results in the /outputs directory

## Modeling Decisions
The main decision I have gotten stuck on in the time I have is how to linearly model the efficiency losses in the case when ramping causes the battery to flip from charging to discharging in the middle of an interval. The ouput power affects SoC differently in the case of charging and discharging. With charging, the SoC change is power * efficiency_loss and with discharging it is power / efficiency_loss. We need a piecewise function. I created this in the `optimizer_zero_crossing.py` file, but haven't had much time to test it and evaluate if the model is working as expected. It will run, but take some time! Generally, in the case of a zero crossing ramp interval what I want to do is find the integral/area of two triangles - one on the charge side and one on the discharge side. The base of the triangle is the time spent in the first state and the height is the previous award power. Then I can do $\frac{1}{2}base \cdot height$. This requires a few tricks to maintain linearity including the big M method to avoid multiplication of binary variables and discretizing the variable the controls how much of the interval is in the first state (charging if going from charging to discharging and vice versa) in order to avoid multiplication of variables in the function
$$
\frac{\lvert \text{prev\_net\_power\_award} \rvert}{\lvert \text{prev\_net\_power\_award} \rvert + \lvert \text{net\_power\_award} \rvert}
$$

 My simplification for the base optimizer.py file is to divide the net power by the discharge inefficiency - effectively just treat every interval as a discharging interval. This is wrong, but works ok for the problem at hand because throughout the day without a set end SoC we do more discharging than charging and the efficiency losses are somewhat small. In order to try the other method that I am working on, you can swap the import in `run_optimizer.py` to be `from optimizer_zero_crossing import optimizer`.

A couple other things I have included are a cycle per day limit (right now set to 5 so it makes no impact, but could be adjusted). I've included all of the inputs about the battery and problem into config files (battery and policy config json files) so that those can be easily adjusted. Last, in the zero crossing optimizer I added the option to save all of the variables for debugging.

## Testing
I have written a couple of tests for the basic logic of awarding the complete capacity of the battery when prices are very high (for energy and reg), charging fully when prices are very negative, and taking no reg awards when the prices are 0. These tests can be run with
```bash
python test_discharge_charge.py
```

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
where $f_{Pi}$ and $f_{\pi i}$ are functions which take in the bid curve and return, respectively, a clearing price and an award power for that interval. We would want to deduce these functions based on the bids in the market at that interval. For the simplest case where we self-schedule (bid a high price) a single MW, we would expect the relationship to match that of the previous objective function because the clearing price will not be affected. As our bids cause the awarded power $\pi_i$ to increase, we would expect $P_i$ to decrease because we would need to offer at a price low enough such that $\pi_i$ gets awarded.

The full objective would include a similar logic for the regulation markets as well
$$
\sum_{i=1}^{t} f^e_{Pi}(\vec{B^{e}_{i}}) \cdot f^{e}_{\pi {i}}(\vec{B^{e}_{i}}) + f^{rr}_{Pi}(\vec{B^{rr}_{i}}) \cdot f^{rr}_{\pi {i}}(\vec{B^{rr}_{i}}) + f^{rl}_{Pi}(\vec{B^{rl}_i}) \cdot f^{rl}_{\pi i}(\vec{B^{rl}_i})
$$
where $rr$ and $rl$ superscripts indicate the applicability to regraise and reglower respectively
In our example day of 12/28, this would play out as lowering the peak energy price for the high price spike in the evening and lowering the regulation raise price just prior to that spike because we offer large amounts of energy and regulation raise in both of those intervals. I imagine this would be especially impactful in the regulation market as that is significantly smaller than the energy market. The effect will depend on the supply stack though. In general, we would expect this change in formulation to lower the total profit of the system. We would also expect less of the behavior we are currently seeing where the battery is generally either doing 100MW or 0MW and instead see more moderate power. This would lower the cycle usage of the battery.
