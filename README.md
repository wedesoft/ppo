# Proximal Policy Optimization for Continuous Action Space

A port of the [PPO implementation by XinJingHao](https://github.com/XinJingHao/PPO-Continuous-Pytorch) to Clojure with added environment implementation.

## Installation

* Install Java
* Install [Clojure](https://clojure.org/) (clj tool)
* Install [Python 3.13](https://www.python.org/)
* Install [uv](https://docs.astral.sh/uv/)

Install the Python packages using `uv sync`.
If your system comes with a newer version of Python, run `uv lock` first.
If your system comes with an older version or if you want to use a Torch version with GPU support, you need to edit the *project.toml* file before running `uv lock` and `uv sync`.

Make sure using `uv python list` that the uv environment is using the system installed Python executable otherwise it seems to fail to import the `_ctypes` module.

Run `uv run clj -M -m ppo.xor` to check things are working.

## Run

* Run interactive REPL: `uv run clojure -M:rebel`
* Run Pendulum: `clj -M -m ppo.pendulum`

# External links

* [John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
* [PPO implementation by XinJingHao](https://github.com/XinJingHao/PPO-Continuous-Pytorch)
* [Gymnasium library](https://gymnasium.farama.org/)
* [libpython-clj2](https://github.com/clj-python/libpython-clj)
* [libpython-clj examples by Gigasquid](https://github.com/gigasquid/libpython-clj-examples/)
* [Torch distributions](https://docs.pytorch.org/docs/stable/distributions.html)
