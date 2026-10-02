"""Production-grade evaluation harness for the agent.

Beyond headline QA accuracy, these modules measure the things a deployed agent lives or dies
on: calibration with uncertainty intervals (stats.py), agent behaviour and groundedness
(behavior.py), and whether the decision loop actually helps versus a single-shot baseline
(ablation.py). See EVALUATION.md for the methodology and results.
"""
