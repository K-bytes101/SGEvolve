# SGEvolve

LLM-driven evolutionary framework. Each **problem** lives under `problem/<name>/` with a `config.yaml` and `evaluator.py`. The evolution loop stores individuals in a local SQLite database.

This repository supports three experiment entry points:

| Script | Purpose |
|--------|---------|
| `main.py` | General benchmarks under `problem/<problem_name>/` |
| `ale_run.py` | [ALE-Bench](https://github.com/SakanaAI/ALE-Bench) heuristic contest tasks |
| `industryOR_run.py` | IndustryOR tasks |

---

## Prerequisites

- **Python** 3.10–3.14 (see `requires-python` in `pyproject.toml`; `.python-version` pins 3.12 for local development).
- **[uv](https://docs.astral.sh/uv/)** for environments and dependency locking (`uv.lock`).
- **Docker** (required for ALE-Bench judging): ensure the Docker daemon is running.
- **LLM API**: configure `API_KEY` and `BASE_URL` for your model endpoint (see below).

---

## Quick start with uv

Run all commands from the `SGEvolve/` repository root. Environment-variable examples below use Bash syntax.

```bash
# Create venv and install core dependencies (some evaluators need extra packages)
uv sync
```

Optional dependency groups (defined in `pyproject.toml`):

```bash
# ALE-Bench (installs ale-bench from Git tag v1.4.0)
uv sync --extra ale-bench

# IndustryOR (requires both datasets and gurobipy, plus a Gurobi license for solving)
uv sync --extra industry-or --extra industry-or-modeling

# Install everything for all three experiments
uv sync --extra all-experiments
```

Run scripts inside the project environment:

```bash
uv run python main.py --help
uv run --extra ale-bench python ale_run.py --help
uv run --extra industry-or --extra industry-or-modeling python industryOR_run.py --help
```

---

## Environment variables

### All LLM-driven runs

| Variable | Required by | Description |
|----------|-------------|-------------|
| `API_KEY` | `main.py`, `ale_run.py`, `industryOR_run.py` | API key for the LLM provider |
| `BASE_URL` | `main.py`, `ale_run.py`, `industryOR_run.py` | API root URL; omit the trailing `/v1` or `/v1beta` |

`utils/llm_requester.py` selects the request format by model-name prefix: `gemini-` uses `/v1beta/models/<model>:generateContent`, while `gpt-` uses `/v1/chat/completions`. Both use Bearer authentication. Set the `model` and property `evaluator` model names in the problem's `config.yaml` to names supported by your endpoint.

### Main experiment (`main.py`)

| Variable | Description |
|----------|-------------|
| `EXP_NUM` | Optional. Experiment index for the SQLite DB filename (`local_population_<EXP_NUM>.db`). Default: `1`. |

### ALE-Bench (`ale_run.py`)

Only `ALE_BENCH_DATA` has a default. The other three variables must be set, including for `--result`:

| Variable | Role |
|----------|------|
| `ALE_BENCH_DATA` | Dataset root; defaults to `./problem/ALE-Bench/dataset/ALE-Bench` |
| `ALE_BENCH_SHARED` | Required. Shared ALE-Bench directory |
| `TMPDIR` | Required. Temporary directory |
| `ALE_BENCH_CACHE` | Required. Cache directory |

Refer to the [ALE-Bench documentation](https://github.com/SakanaAI/ALE-Bench) for dataset setup and Docker requirements.

---

## Running the three experiments

### 1. `main.py` — general problems

Each problem directory must contain `config.yaml` and `evaluator.py`.

Select a problem with `--problem` (default: `sofa_moving`). Available general problems are `circle_packing`, `sofa_moving`, `heilbronn_problem`, `sphere_packing_7D`, and `max_to_min_distance_ratio`.

```bash
uv sync
export API_KEY="YOUR_API_KEY"
export BASE_URL="https://your-api-host"
uv run python main.py --problem circle_packing --cold-start --evolve

# Continue from the same database without clearing the population
uv run python main.py --problem circle_packing --evolve
```

- `--cold-start` clears the selected database and generates the initial population. Add `--evolve` to continue directly into evolution, or run `--evolve` separately later.
- `--inject_init_solution` takes effect only with `--cold-start` and requires solution files in the directory specified by `init_solution_dir`. These directories are not bundled with the general problems.
- Keep the same `--problem` and `EXP_NUM` when continuing an experiment.

Some evaluators import extra packages (for example **Shapely** for `sofa_moving`). If import fails, install the missing library into the same uv environment (for example `uv add shapely`) or add it to `pyproject.toml`.

### 2. `ale_run.py` — ALE-Bench

```bash
uv sync --extra ale-bench
export API_KEY="YOUR_API_KEY"
export BASE_URL="https://your-api-host"
export ALE_BENCH_DATA="$PWD/problem/ALE-Bench/dataset/ALE-Bench"
export ALE_BENCH_SHARED="$PWD/.ale-bench/shared"
export TMPDIR="$PWD/.ale-bench/tmp"
export ALE_BENCH_CACHE="$PWD/.ale-bench/cache"
mkdir -p "$ALE_BENCH_SHARED" "$TMPDIR" "$ALE_BENCH_CACHE"

uv run --extra ale-bench python ale_run.py --problem-ids ahc008 --cold-start --evolve
uv run --extra ale-bench python ale_run.py --problem-ids ahc008 --evolve
```

Optional flags:

- `--problem-ids ahc008 ahc011` — select tasks. The default list is `ahc008`, `ahc011`, `ahc015`, `ahc016`, `ahc024`, `ahc025`, `ahc026`, `ahc027`, `ahc039`, and `ahc046`.
- `--cold-start` — clear each selected task's database and generate an initial population.
- `--inject_init_solution` — use with `--cold-start` to also inject `problem/ALE-Bench/ALE_Agent_baseline/self-refine_16x/<ID>_initial_program.cpp`. The file must exist for each selected task.
- `--result` — disable cold start, injection, and evolution; run private evaluation of stored individuals and print scores, ranks, and performance. This still starts ALE-Bench sessions and requires Docker, the dataset, the path variables above, `API_KEY`, and `BASE_URL`.

Each task uses `problem/ALE-Bench/local_population_<ID>.db`.

### 3. `industryOR_run.py` — IndustryOR

Requires the on-disk dataset at `problem/IndustryOR/dataset` (loaded via `datasets.load_from_disk`, using its `test` split), the `industry-or` and `industry-or-modeling` extras, and a Gurobi license suitable for the models being solved.

IndustryOR uses Gurobi modeling mode by default, with configuration in `problem/IndustryOR/modeling_mode/config.yaml`. The optional `--modeling_mode` flag is retained for compatibility with existing commands.

```bash
uv sync --extra industry-or --extra industry-or-modeling
export API_KEY="YOUR_API_KEY"
export BASE_URL="https://your-api-host"
uv run --extra industry-or --extra industry-or-modeling python industryOR_run.py --modeling_mode --cold_start --evolve

# Continue from existing populations
uv run --extra industry-or --extra industry-or-modeling python industryOR_run.py --modeling_mode --evolve
```

The cold-start flag is `--cold_start` (underscore), unlike the other two scripts. It clears each task's database before initialization. The script processes every task in the `test` split; there is no task-selection flag. Databases are stored in `problem/IndustryOR/modeling_mode/local_population_id_<ID>.db`, and results are written to `problem/IndustryOR/industryOR_results.json`.

---

## Project layout (high level)

- `agents/` — LLM agents (generator, mutator, crossover, extractor, reward system).
- `population/` — individuals, SQLite persistence (`sqlmodel`).
- `runner_core/` — island evolution and migration workers.
- `problem/` — problem definitions (`config.yaml`, `evaluator.py`, optional DB files).
- `utils/` — LLM HTTP client and shared helpers.

---

## Lockfile and reproducibility

Dependencies are pinned in `uv.lock`. After editing `pyproject.toml`, refresh the lockfile with:

```bash
uv lock
```


## Limitations

Search performance depends on the backbone LLM, the task evaluator, and the available computational budget. Generated programs and numerical results require task-specific evaluation and validation.

## License

The original SGEvolve code is released under the [MIT License](LICENSE). Third-party code, datasets, and dependencies remain subject to their respective licenses and terms of use.
