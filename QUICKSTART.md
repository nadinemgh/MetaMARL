# Quickstart

This page takes you from a fresh clone to a first bilevel run of the fishery
benchmark, then shows where its results are and how to change the experiment. It
needs no Weights & Biases account and no GPU. The figures quoted below come from a
run on a laptop; yours may differ in the last digits if your hardware does.

## 1. Install and check

```bash
git clone git@github.com:deep-introspection/MetaMARL.git
cd MetaMARL
uv sync
WANDB_MODE=offline uv run metamarl check examples/bilevel_fishery/config.yaml
```

`uv sync` creates `.venv/` with Python 3.12, the runtime dependencies and the
development tools. The `check` command builds the fishery experiment described in
`examples/bilevel_fishery/config.yaml` without training it. Its first two log lines
show the hash-seed guard restarting the process with `PYTHONHASHSEED=0`, and its
last line is `Config OK: core.optimizers.bilevel.BilevelConfig`.

The unit tests, with the doctests, confirm the installation in about forty
seconds:

```bash
uv run python -m pytest -m "not integration and not notebook" -q
```

## 2. A first bilevel run

```bash
WANDB_MODE=offline uv run python -m examples.bilevel_fishery.debug \
    --outer-iters 2 --train-iters 2 --num-agents 2 --horizon 20 --reporter csv
```

This runs two generations of the outer Evolution Strategies optimizer. In each
generation the regulator proposes four values of the fishing quota, and for each
of them the inner APPO optimizer trains two fishers for two iterations on
20-step episodes, then evaluates them. The run takes about fifteen seconds.

The lines worth reading are the ones the ES prints at every generation. The
`BEFORE UPDATE` line lists the quotas of the generation and the fitness of each:

```text
[ES] BEFORE UPDATE | gen=0 | mean=[0.5] | sigma=0.15000 | population=[[0.5053...], [0.4377...], [0.4946...], [0.5622...]] | fitness=[2.5979..., 2.5989..., 2.5978..., 2.6060...]
[ES] gen=0 | best=2.6060 | mean=2.6002+/-0.0034 | var=0.0000 | sigma=0.1500
...
[Bilevel] Run finished | iters=2 | converged=False | mechanism=[0.5622445] | best_fitness=2.6060
```

The quota is a fraction of the stock, so the candidates live in `[0, 1]`; `mean`
is the centre of the search and `sigma` its spread, both in that space. The fitness
of a candidate is the realised harvest relative to the maximum sustainable yield
plus twice the mean normalised biomass, both measured over the last steps of the
evaluation episodes (see `examples/bilevel_fishery/regulator_env.py`). The last line reports
the best candidate of the whole run. `converged=False` is expected: the
convergence stop is off unless `convergence_eps` is set in the ES configuration.

The run is reproducible. Running the same command again gives the same `fitness`
vectors to the last digit, because the seeds are fixed and the hash-seed guard
fixes the order in which RLlib batches the agents.

## 3. Read the outputs

With `--reporter csv` every plot of the run is written as a CSV file. The files of
the regulator and those of the inner optimizer go to two directories named after
the run's world and the optimizer class:

```text
results/bilevel/fishery_world_<id>-ESOptimizer/     7 files
results/bilevel/fishery_world_<id>-RayOptimizer/    9 files
```

The ES directory holds the fitness over the generations, the fitness of every
candidate against its quota, and the comparison of training and evaluation for the
return and the normalised biomass. The inner directory holds the training curves:
episode return, biomass statistics, policy entropy and the losses. Every file has
the same long format, one row per point:

```text
query,x,series,value,error,color
Candidate fitness vs quota,0.5622444748878479,Evaluated mechanisms [by_mechanism=3],2.6060359477996826,,0
```

`series` names the curve a point belongs to and `color` carries the generation for
the scatter plots, so a file loads directly into pandas and pivots on `series`.
The plots themselves are declared in `examples/bilevel_fishery/queries.py`; each
`Query` names the logged metrics it reads and how to reduce them. Without
`--reporter csv` the same queries are rendered as Plotly figures in Weights &
Biases (under `wandb/` when offline). In a run this short the policy-loss file holds
only `nan` values, because APPO reports its losses only once every twenty
gradient updates.

## 4. Change the experiment

The YAML file and the debug script describe the same experiment in two ways. The
YAML file is the declarative form: every mapping with a `_target_` key is built by
calling that class, `_calls_` lists the builder methods applied to it in order,
`_symbol_` imports an object without calling it, and `_tuple_` builds a tuple. The
`run` section at the end lists the calls `metamarl run` applies to the built
experiment. For instance, the searched mechanism is declared inside the
regulator's agent:

```yaml
- _target_: core.mechanism.algorithms.quota.Quota
  id: quota
  acts_on:
    _tuple_: [fisherman, harvest]
  obs_map:
    resource_level: fish
  default:
    _target_: numpy.asarray
    _args_: [0.56224]
```

`acts_on` says which mechanism of which agents the quota constrains, `obs_map`
which state variable it reads, and `default` the quota the fishers face before
the regulator's first candidate arrives. The ES settings (`sigma`, `mean_lr`, the
number of generations under `episodes`) sit in the `training` call of the
regulator, and the fishery's ecology under the society's `env_config`. After an
edit, `metamarl check` tells you whether the file still builds; a configuration
error names the YAML path that failed and its cause.

The debug script, `examples/bilevel_fishery/debug.py`, builds the same experiment
with the Python builders of `BilevelConfig`, and its options change the sizes
without editing anything: `--outer-iters`, `--train-iters`, `--num-agents`,
`--horizon` and `--reporter`. `--help` lists them with their defaults, which are
the full experiment.

## 5. Where to go next

The other two benchmarks run the same way: `python -m examples.cartpole.debug`
checks the whole pipeline on a task whose fitness is known in advance, and
`python -m examples.fresh_water.debug` searches the eight rules of a water policy
for a reservoir shared by farms. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
explains what happens inside a generation and how to write your own benchmark or
mechanism. The notebooks in `tutorials/` present the abstractions on the fishery
(`metamarl_fishery_tutorial.ipynb`), the mechanism algorithms
(`mechanism_algorithms.ipynb`), the creation of a custom benchmark
(`custom_benchmark_creation.ipynb`), the plotting layer (`visualization.ipynb`) and
a research note on mechanisms in continuous time
(`continuous_time_mechanism_vector_fields.ipynb`).
