# rl-experiments

A personal project for learning reinforcement learning on financial time-series data: PPO and recurrent-PPO
agents trained in a simulated trading environment, with the emphasis on **how you evaluate an agent** rather
than on returns.

I built this to understand the training and evaluation loop end to end — environment design, reward shaping,
hyperparameter search, and, above all, how easy it is to fool yourself with a good-looking backtest.

## What this is / what this is not

**It is** a research and learning setup: a reproducible pipeline where an idea can be trained, tuned, and then
tested on data it has never seen.

**It is not** a production system and not financial advice. Nothing here is deployed against a live account,
and the results are not presented as a strategy that works.

## Approach

**Algorithms** — PPO and RecurrentPPO (LSTM policy) via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
and sb3-contrib. MLP policy with two 256-unit hidden layers; observations normalised with `VecNormalize`.

**Environment** — a simulated account with a small starting balance and micro lots, so that transaction costs
matter relative to position size. Spread, commission and slippage are charged on every trade rather than
assumed away.

**Timeframes** — 15m, 1h and 1d, with transfer learning between timeframes and instruments so a policy trained
on one series can be used as a starting point for another.

**Reward** — realised P&L in pips, minus transaction costs, plus a rolling Sharpe term; penalties for
overtrading and for holding too long, an incentive for trading with the prevailing moving-average trend, and
ATR-based dynamic stop-loss / take-profit. Losses are weighted more heavily than equivalent gains, so the agent
is not rewarded for taking large tail risk to smooth its average.

## Evaluation

This is the part the project exists for.

- **Walk-forward validation** — the agent is trained on one window and evaluated on the next, repeatedly, so a
  result only counts if it survives data the model has never seen.
- **Hyperparameter search** with [Optuna](https://optuna.org/), rather than hand-tuning until a number looks good.
- **Backtest reporting** — equity curves and per-run charts written to `outputs/`, so failure cases can be read
  rather than summarised into a single metric.
- **Failure analysis** — the interesting runs are the ones that lose money in a specific, explainable way.

## Structure

```
config/     configuration and run settings
src/
  data/     loading and preprocessing of price series
  core/     the RL environment
  ui/       dashboard and visualisation
  utils/    reporting helpers
scripts/    entry points: train_agent, train_recurrent, test_agent,
            optimize_hyperparams, walk_forward
tests/      unit tests
outputs/    backtest results and charts
```

## Running it

Requires Python 3 and the packages in `Requirements.txt`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r Requirements.txt

python scripts/train_agent.py          # train a PPO agent
python scripts/optimize_hyperparams.py # Optuna search
python scripts/walk_forward.py         # walk-forward validation
python scripts/test_agent.py           # evaluate a saved agent
```

## Notes

Built as a learning project; I used AI coding assistance while working on it. The goal was to understand
reinforcement learning and, in particular, honest evaluation methodology — not to ship a trading product.

MIT-style personal project. Use at your own risk.
