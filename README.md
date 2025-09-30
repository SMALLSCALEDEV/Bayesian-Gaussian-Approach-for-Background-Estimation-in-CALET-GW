# Bayesian-Gaussian-Approach-for-Background-Estimation-in-CALET-GW
# CALET GW Background Modelling

This repository implements the Bayesian Gaussian Process + change-point framework 
for background estimation in CALET gravitational-wave follow-ups, as described in:

**Bisweswar Sen (2025), "Improved Background Estimation in CALET GW Follow-ups: A Bayesian Gaussian Process Approach"**

## Features
- Gaussian Process regression (`background.py`)
- Change-point detection with hybrid GP models (`changepoints.py`)
- Detection statistics including background uncertainty (`detection.py`)
- Bayesian credible upper limits (`upperlimits.py`)
- Injection–recovery simulations (`injections.py`)
- High-level orchestrator (`pipeline.py`)
- Reproducible, modular Python code

## Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
