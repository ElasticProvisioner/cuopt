/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/bnb_worker.hpp>
#include <dual_simplex/logger.hpp>
#include <dual_simplex/mip_node.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/types.hpp>
#include <utilities/omp_helpers.hpp>
#include <utilities/pcgenerator.hpp>

#include <omp.h>
#include <cmath>
#include <cstdint>

namespace cuopt::linear_programming::dual_simplex {

// =============================================================================
// Pseudo-cost utility functions (lock-free implementations)
// These can be called directly with snapshot data or through pseudo_costs_t methods
// =============================================================================

template <typename f_t>
struct pseudo_cost_averages_t {
  f_t down_avg;
  f_t up_avg;
};

// used to get T from omp_atomic_t<T> based on the fact that omp_atomic_t<T>::operator++ returns T
template <typename T>
using underlying_type = decltype(std::declval<T&>()++);

// Necessary because omp_atomic_t<f_t> may be passed instead of f_t
template <typename MaybeWrappedI, typename MaybeWrappedF>
auto compute_pseudo_cost_averages(const MaybeWrappedF* pc_sum_down,
                                  const MaybeWrappedF* pc_sum_up,
                                  const MaybeWrappedI* pc_num_down,
                                  const MaybeWrappedI* pc_num_up,
                                  size_t n)
{
  using underlying_f_t = underlying_type<MaybeWrappedF>;
  using underlying_i_t = underlying_type<MaybeWrappedI>;

  underlying_i_t num_initialized_down = 0;
  underlying_i_t num_initialized_up   = 0;
  underlying_f_t pseudo_cost_down_avg = 0.0;
  underlying_f_t pseudo_cost_up_avg   = 0.0;

  for (size_t j = 0; j < n; ++j) {
    if (pc_num_down[j] > 0) {
      ++num_initialized_down;
      if (std::isfinite(pc_sum_down[j])) {
        pseudo_cost_down_avg += pc_sum_down[j] / pc_num_down[j];
      }
    }
    if (pc_num_up[j] > 0) {
      ++num_initialized_up;
      if (std::isfinite(pc_sum_up[j])) { pseudo_cost_up_avg += pc_sum_up[j] / pc_num_up[j]; }
    }
  }

  pseudo_cost_down_avg =
    (num_initialized_down > 0) ? pseudo_cost_down_avg / num_initialized_down : 1.0;
  pseudo_cost_up_avg = (num_initialized_up > 0) ? pseudo_cost_up_avg / num_initialized_up : 1.0;

  return pseudo_cost_averages_t<underlying_f_t>{pseudo_cost_down_avg, pseudo_cost_up_avg};
}

// Variable selection using pseudo-cost product scoring
// Returns the best variable to branch on
template <typename i_t, typename f_t>
i_t variable_selection_from_pseudo_costs(const f_t* pc_sum_down,
                                         const f_t* pc_sum_up,
                                         const i_t* pc_num_down,
                                         const i_t* pc_num_up,
                                         i_t n_vars,
                                         const std::vector<i_t>& fractional,
                                         const std::vector<f_t>& solution)
{
  const i_t num_fractional = fractional.size();
  if (num_fractional == 0) return -1;

  auto [pc_down_avg, pc_up_avg] =
    compute_pseudo_cost_averages(pc_sum_down, pc_sum_up, pc_num_down, pc_num_up, n_vars);

  i_t branch_var    = fractional[0];
  f_t max_score     = std::numeric_limits<f_t>::lowest();
  constexpr f_t eps = f_t(1e-6);

  for (i_t j : fractional) {
    f_t pc_down      = pc_num_down[j] != 0 ? pc_sum_down[j] / pc_num_down[j] : pc_down_avg;
    f_t pc_up        = pc_num_up[j] != 0 ? pc_sum_up[j] / pc_num_up[j] : pc_up_avg;
    const f_t f_down = solution[j] - std::floor(solution[j]);
    const f_t f_up   = std::ceil(solution[j]) - solution[j];
    f_t score        = std::max(f_down * pc_down, eps) * std::max(f_up * pc_up, eps);
    if (score > max_score) {
      max_score  = score;
      branch_var = j;
    }
  }

  return branch_var;
}

// Objective estimate using pseudo-costs (lock-free implementation)
// Returns lower_bound + estimated cost to reach integer feasibility
template <typename i_t, typename f_t>
f_t obj_estimate_from_arrays(const f_t* pc_sum_down,
                             const f_t* pc_sum_up,
                             const i_t* pc_num_down,
                             const i_t* pc_num_up,
                             i_t n_vars,
                             const std::vector<i_t>& fractional,
                             const std::vector<f_t>& solution,
                             f_t lower_bound)
{
  auto [pc_down_avg, pc_up_avg] =
    compute_pseudo_cost_averages(pc_sum_down, pc_sum_up, pc_num_down, pc_num_up, n_vars);

  f_t estimate      = lower_bound;
  constexpr f_t eps = f_t(1e-6);

  for (i_t j : fractional) {
    f_t pc_down      = pc_num_down[j] != 0 ? pc_sum_down[j] / pc_num_down[j] : pc_down_avg;
    f_t pc_up        = pc_num_up[j] != 0 ? pc_sum_up[j] / pc_num_up[j] : pc_up_avg;
    const f_t f_down = solution[j] - std::floor(solution[j]);
    const f_t f_up   = std::ceil(solution[j]) - solution[j];
    estimate += std::min(std::max(pc_down * f_down, eps), std::max(pc_up * f_up, eps));
  }

  return estimate;
}

template <typename i_t, typename f_t>
struct reliability_branching_settings_t {
  // Lower bound for the maximum number of LP iterations for a single trial branching
  i_t lower_max_lp_iter = 10;

  // Upper bound for the maximum number of LP iterations for a single trial branching
  i_t upper_max_lp_iter = 500;

  // Priority of the tasks created when running the trial branching in parallel.
  // Set to 1 to have the same priority as the other tasks.
  i_t task_priority = 5;

  // The maximum number of candidates initialized by strong branching in a single
  // node
  i_t max_num_candidates = 100;

  // Define the maximum number of iteration spent in strong branching.
  // Let `bnb_lp_iter` = total number of iterations in B&B, then
  // `max iter in strong branching = bnb_lp_factor * bnb_lp_iter + bnb_lp_offset`.
  // This is used for determining the `reliable_threshold`.
  f_t bnb_lp_factor = 0.5;
  i_t bnb_lp_offset = 100000;

  // Maximum and minimum points in curve to determine the value
  // of the `reliable_threshold` based on the current number of LP
  // iterations in strong branching and B&B. Since it is a
  // a curve, the actual value of `reliable_threshold` may be
  // higher than `max_reliable_threshold`.
  // Only used when `reliable_threshold` is negative
  i_t max_reliable_threshold = 5;
  i_t min_reliable_threshold = 1;
};

template <typename i_t, typename f_t>
class pseudo_costs_t {
 public:
  explicit pseudo_costs_t(i_t num_variables)
    : pseudo_cost_sum_down(num_variables),
      pseudo_cost_sum_up(num_variables),
      pseudo_cost_num_down(num_variables),
      pseudo_cost_num_up(num_variables),
      pseudo_cost_mutex(num_variables)
  {
  }

  void update_pseudo_costs(mip_node_t<i_t, f_t>* node_ptr, f_t leaf_objective);

  void resize(i_t num_variables)
  {
    pseudo_cost_sum_down.assign(num_variables, 0);
    pseudo_cost_sum_up.assign(num_variables, 0);
    pseudo_cost_num_down.assign(num_variables, 0);
    pseudo_cost_num_up.assign(num_variables, 0);
    pseudo_cost_mutex.resize(num_variables);
  }

  void initialized(i_t& num_initialized_down,
                   i_t& num_initialized_up,
                   f_t& pseudo_cost_down_avg,
                   f_t& pseudo_cost_up_avg) const;

  f_t obj_estimate(const std::vector<i_t>& fractional,
                   const std::vector<f_t>& solution,
                   f_t lower_bound,
                   logger_t& log);

  i_t variable_selection(const std::vector<i_t>& fractional,
                         const std::vector<f_t>& solution,
                         logger_t& log);

  i_t reliable_variable_selection(mip_node_t<i_t, f_t>* node_ptr,
                                  const std::vector<i_t>& fractional,
                                  const std::vector<f_t>& solution,
                                  const simplex_solver_settings_t<i_t, f_t>& settings,
                                  const std::vector<variable_type_t>& var_types,
                                  bnb_worker_data_t<i_t, f_t>* worker_data,
                                  const bnb_stats_t<i_t, f_t>& bnb_stats,
                                  f_t upper_bound,
                                  int max_num_tasks,
                                  logger_t& log);

  void update_pseudo_costs_from_strong_branching(const std::vector<i_t>& fractional,
                                                 const std::vector<f_t>& root_soln);

  uint32_t compute_state_hash() const
  {
    return detail::compute_hash(pseudo_cost_sum_down) ^ detail::compute_hash(pseudo_cost_sum_up) ^
           detail::compute_hash(pseudo_cost_num_down) ^ detail::compute_hash(pseudo_cost_num_up);
  }

  uint32_t compute_strong_branch_hash() const
  {
    return detail::compute_hash(strong_branch_down) ^ detail::compute_hash(strong_branch_up);
  }

  f_t calculate_pseudocost_score(i_t j,
                                 const std::vector<f_t>& solution,
                                 f_t pseudo_cost_up_avg,
                                 f_t pseudo_cost_down_avg) const;

  reliability_branching_settings_t<i_t, f_t> reliability_branching_settings;

  std::vector<omp_atomic_t<f_t>> pseudo_cost_sum_up;
  std::vector<omp_atomic_t<f_t>> pseudo_cost_sum_down;
  std::vector<omp_atomic_t<i_t>> pseudo_cost_num_up;
  std::vector<omp_atomic_t<i_t>> pseudo_cost_num_down;
  std::vector<f_t> strong_branch_down;
  std::vector<f_t> strong_branch_up;
  std::vector<omp_mutex_t> pseudo_cost_mutex;
  omp_atomic_t<i_t> num_strong_branches_completed = 0;
  omp_atomic_t<int64_t> strong_branching_lp_iter  = 0;
};

template <typename i_t, typename f_t>
void strong_branching(const user_problem_t<i_t, f_t>& original_problem,
                      const lp_problem_t<i_t, f_t>& original_lp,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      f_t start_time,
                      const std::vector<variable_type_t>& var_types,
                      const std::vector<f_t> root_soln,
                      const std::vector<i_t>& fractional,
                      f_t root_obj,
                      const std::vector<variable_status_t>& root_vstatus,
                      const std::vector<f_t>& edge_norms,
                      pseudo_costs_t<i_t, f_t>& pc);

}  // namespace cuopt::linear_programming::dual_simplex
