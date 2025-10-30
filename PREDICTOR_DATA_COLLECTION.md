# Predictor Data Collection: Feature Logging

This document describes the instrumentation added to collect training data for work unit predictors.

## Overview

Three algorithms have been instrumented to log features before execution and performance metrics after completion:

1. **Feasibility Pump (FP)** - Main heuristic for finding feasible solutions
2. **PDLP** - First-order LP solver used for polytope projection
3. **Constraint Propagation (CP)** - Variable rounding with bounds propagation

Note: **Feasibility Jump (FJ)** already has a working predictor and doesn't need additional instrumentation.

## Log Format

All logs use `CUOPT_LOG_INFO` level with structured prefixes for easy parsing:

### Feasibility Pump (FP)

**Features logged before execution:**
```
FP_FEATURES: n_variables=%d n_constraints=%d n_integer_vars=%d n_binary_vars=%d
FP_FEATURES: nnz=%lu sparsity=%.6f nnz_stddev=%.6f unbalancedness=%.6f
FP_FEATURES: initial_feasibility=%d initial_excess=%.6f initial_objective=%.6f
FP_FEATURES: initial_ratio_of_integers=%.6f initial_n_integers=%d
FP_FEATURES: alpha=%.6f check_distance_cycle=%d cycle_detection_length=%d
FP_FEATURES: has_cutting_plane=%d time_budget=%.6f
```

**Results logged after execution:**
```
FP_RESULT: iterations=%d time_taken=%.6f termination=<REASON>
```

**Termination reasons:**
- `TIME_LIMIT` - Time budget exhausted
- `TIME_LIMIT_AFTER_ROUND` - Time limit during rounding phase
- `FEASIBLE_LP_PROJECTION` - Found feasible via LP projection
- `FEASIBLE_LP_VERIFIED` - Found feasible via high-precision LP
- `FEASIBLE_AFTER_ROUND` - Found feasible after rounding
- `FEASIBLE_DISTANCE_CYCLE` - Found feasible during distance cycle handling
- `INFEASIBLE_DISTANCE_CYCLE` - Distance cycle detected, no feasible found
- `ASSIGNMENT_CYCLE` - Assignment cycle detected

**Location:** `cpp/src/mip/local_search/feasibility_pump/feasibility_pump.cu::run_single_fp_descent`

---

### PDLP (LP Solver)

**Features logged before execution:**
```
PDLP_FEATURES: n_variables=%d n_constraints=%d nnz=%lu
PDLP_FEATURES: sparsity=%.6f nnz_stddev=%.6f unbalancedness=%.6f
PDLP_FEATURES: has_warm_start=%d time_limit=%.6f iteration_limit=%d
PDLP_FEATURES: tolerance=%.10f check_infeasibility=%d return_first_feasible=%d
```

**Results logged after execution:**
```
PDLP_RESULT: iterations=%d time_ms=%lld termination=%d
PDLP_RESULT: primal_objective=%.10f dual_objective=%.10f gap=%.10f
PDLP_RESULT: l2_primal_residual=%.10f l2_dual_residual=%.10f
```

**Termination status codes:**
- `0` - NoTermination
- `1` - NumericalError
- `2` - Optimal
- `3` - PrimalInfeasible
- `4` - DualInfeasible
- `5` - IterationLimit
- `6` - TimeLimit
- `7` - PrimalFeasible
- `8` - ConcurrentLimit

**Location:** `cpp/src/mip/relaxed_lp/relaxed_lp.cu::get_relaxed_lp_solution`

---

### Constraint Propagation (CP)

**Features logged before execution:**
```
CP_FEATURES: n_variables=%d n_constraints=%d n_integer_vars=%d
CP_FEATURES: nnz=%lu sparsity=%.6f
CP_FEATURES: n_unset_vars=%d initial_excess=%.6f time_budget=%.6f
CP_FEATURES: round_all_vars=%d lp_run_time_after_feasible=%.6f
```

**Results logged after execution:**
```
CP_RESULT: time_ms=%lld termination=<STATUS> iterations=%d
```

**Termination status:**
- `BRUTE_FORCE_SUCCESS` - Succeeded via simple rounding
- `SUCCESS` - Found feasible solution
- `FAILED` - Did not find feasible solution

**Location:** `cpp/src/mip/local_search/rounding/constraint_prop.cu::apply_round`

---

## Data Collection Workflow

### 1. Run Solver with Logging Enabled

Ensure the log level is set to `INFO` or higher to capture the feature logs:

```bash
export CUOPT_LOG_LEVEL=INFO
# or
export CUOPT_LOG_LEVEL=DEBUG
```

### 2. Parse Logs

Use the following pattern to extract training data:

```python
import re
import json

def parse_fp_features(log_lines):
    """Parse FP features from log lines"""
    features = {}

    # Match feature lines
    for line in log_lines:
        if "FP_FEATURES:" in line:
            # Extract key=value pairs
            matches = re.findall(r'(\w+)=([\d.e+-]+)', line)
            features.update({k: float(v) if '.' in v else int(v)
                           for k, v in matches})

    return features

def parse_fp_result(log_lines):
    """Parse FP results from log lines"""
    for line in log_lines:
        if "FP_RESULT:" in line:
            match = re.search(r'iterations=(\d+) time_taken=([\d.]+) termination=(\w+)', line)
            if match:
                return {
                    'iterations': int(match.group(1)),
                    'time_taken': float(match.group(2)),
                    'termination': match.group(3)
                }
    return None

# Similar functions for PDLP and CP...
```

### 3. Create Training Dataset

Combine features and results into training examples:

```python
training_data = []

for problem_run in log_files:
    features = parse_fp_features(problem_run)
    result = parse_fp_result(problem_run)

    if features and result:
        training_data.append({
            'features': features,
            'target': result['iterations'],  # or result['time_taken']
            'metadata': {
                'termination': result['termination']
            }
        })
```

### 4. Train Predictor Model

Use the same approach as the existing FJ predictor:

```python
# Example using XGBoost (like FJ predictor)
import xgboost as xgb

# Prepare data
X = [sample['features'] for sample in training_data]
y = [sample['target'] for sample in training_data]

# Train model
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)
model.fit(X, y)

# Save model for C++ code generation
# (Similar to cpp/src/utilities/models/fj_predictor/)
```

---

## Feature Descriptions

### Problem Structure Features

- **n_variables** - Total number of decision variables
- **n_constraints** - Total number of constraints
- **n_integer_vars** - Number of integer/binary variables
- **n_binary_vars** - Number of binary (0/1) variables
- **nnz** - Non-zero coefficients in constraint matrix
- **sparsity** - Matrix sparsity: nnz / (n_constraints × n_variables)
- **nnz_stddev** - Standard deviation of non-zeros per constraint row
- **unbalancedness** - Load balancing metric for constraint matrix

### Solution State Features (FP)

- **initial_feasibility** - Whether starting solution is feasible (0/1)
- **initial_excess** - Sum of constraint violations
- **initial_objective** - Objective value of initial solution
- **initial_ratio_of_integers** - Fraction of integer vars already integral
- **initial_n_integers** - Count of integer vars at integral values

### Algorithm Configuration Features (FP)

- **alpha** - Weight between original objective and distance objective
- **check_distance_cycle** - Whether distance-based cycle detection is enabled
- **cycle_detection_length** - Number of recent solutions tracked
- **has_cutting_plane** - Whether objective cutting plane was added
- **time_budget** - Allocated time in seconds

### Solver Configuration Features (PDLP)

- **has_warm_start** - Whether initial primal/dual solution provided
- **time_limit** - Time budget in seconds
- **iteration_limit** - Maximum iterations allowed
- **tolerance** - Optimality tolerance
- **check_infeasibility** - Whether to detect infeasibility
- **return_first_feasible** - Whether to return on first primal feasible

### Rounding Configuration Features (CP)

- **n_unset_vars** - Integer variables not yet set
- **round_all_vars** - Whether to round all variables or selective
- **lp_run_time_after_feasible** - Time budget for post-feasibility LP

---

## Integration with Existing Predictor

The FJ predictor already exists at `cpp/src/utilities/models/fj_predictor/`. The same workflow can be used:

1. Collect training data as described above
2. Train XGBoost model
3. Export to C++ using TreeLite (as done for FJ)
4. Integrate into solver with work unit → iteration conversion

Example from FJ predictor:
```cpp
// cpp/src/mip/feasibility_jump/feasibility_jump.cu:1283-1291
if (settings.work_unit_limit != std::numeric_limits<double>::infinity()) {
    std::map<std::string, float> features_map = get_feature_vector(0);
    float iter_prediction = std::max(
        (f_t)0.0,
        (f_t)ceil(context.work_unit_predictors.fj_predictor.predict_scalar(features_map))
    );
    CUOPT_LOG_DEBUG("FJ determ: Estimated number of iterations for %f WU: %f",
                    settings.work_unit_limit,
                    iter_prediction);
    settings.iteration_limit = std::min(settings.iteration_limit, (i_t)iter_prediction);
}
```

---

## Next Steps

1. **Collect Data**: Run solver on diverse problem sets with logging enabled
2. **Analyze**: Examine feature importance and correlation with execution time
3. **Train Models**: Build iteration predictors for FP, PDLP, and CP
4. **Validate**: Test predictors maintain solution quality while achieving determinism
5. **Deploy**: Integrate trained models into solver (similar to FJ predictor)
6. **Hierarchical Allocation**: Implement work unit budget allocation across nested algorithms

---

## Notes

- Line Segment Search was excluded as it can be predicted from FJ predictor (it runs FJ internally)
- CP iteration tracking needs enhancement (currently logs 0 iterations)
- Consider adding more dynamic features during execution for better predictions
- The termination reasons can help understand when algorithms succeed/fail
