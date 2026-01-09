/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/logger.hpp>
#include <dual_simplex/mip_node.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/types.hpp>
#include <utilities/omp_helpers.hpp>

#include <omp.h>
#include <cmath>
#include <cstdint>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
class pseudo_costs_t {
 public:
  explicit pseudo_costs_t(i_t num_variables)
    : pseudo_cost_sum_down(num_variables),
      pseudo_cost_sum_up(num_variables),
      pseudo_cost_num_down(num_variables),
      pseudo_cost_num_up(num_variables)
  {
  }

  void update_pseudo_costs(mip_node_t<i_t, f_t>* node_ptr, f_t leaf_objective);

  void resize(i_t num_variables)
  {
    pseudo_cost_sum_down.resize(num_variables);
    pseudo_cost_sum_up.resize(num_variables);
    pseudo_cost_num_down.resize(num_variables);
    pseudo_cost_num_up.resize(num_variables);
  }

  void initialized(i_t& num_initialized_down,
                   i_t& num_initialized_up,
                   f_t& pseudo_cost_down_avg,
                   f_t& pseudo_cost_up_avg) const;

  i_t variable_selection(const std::vector<i_t>& fractional,
                         const std::vector<f_t>& solution,
                         logger_t& log);

  void update_pseudo_costs_from_strong_branching(const std::vector<i_t>& fractional,
                                                 const std::vector<f_t>& root_soln);

  // Compute a deterministic hash of the pseudo-cost state for divergence detection
  uint32_t compute_state_hash() const
  {
    uint32_t hash    = 0x811c9dc5;  // FNV-1a offset basis
    auto hash_double = [&hash](double val) {
      // Convert to fixed-point representation for exact comparison
      int64_t fixed = static_cast<int64_t>(val * 1000000.0);
      hash ^= static_cast<uint32_t>(fixed & 0xFFFFFFFF);
      hash *= 0x01000193;  // FNV-1a prime
      hash ^= static_cast<uint32_t>((fixed >> 32) & 0xFFFFFFFF);
      hash *= 0x01000193;
    };
    auto hash_int = [&hash](i_t val) {
      hash ^= static_cast<uint32_t>(val);
      hash *= 0x01000193;
    };

    // Hash pseudo-cost sums and counts
    for (size_t j = 0; j < pseudo_cost_sum_down.size(); ++j) {
      hash_double(pseudo_cost_sum_down[j]);
      hash_double(pseudo_cost_sum_up[j]);
      hash_int(pseudo_cost_num_down[j]);
      hash_int(pseudo_cost_num_up[j]);
    }
    return hash;
  }

  // Compute hash of strong branching results
  uint32_t compute_strong_branch_hash() const
  {
    uint32_t hash    = 0x811c9dc5;
    auto hash_double = [&hash](double val) {
      if (std::isnan(val)) {
        hash ^= 0xDEADBEEF;  // Special marker for NaN
      } else if (std::isinf(val)) {
        hash ^= val > 0 ? 0xCAFEBABE : 0xBADCAFE;  // Inf markers
      } else {
        int64_t fixed = static_cast<int64_t>(val * 1000000.0);
        hash ^= static_cast<uint32_t>(fixed & 0xFFFFFFFF);
        hash *= 0x01000193;
        hash ^= static_cast<uint32_t>((fixed >> 32) & 0xFFFFFFFF);
      }
      hash *= 0x01000193;
    };

    for (size_t k = 0; k < strong_branch_down.size(); ++k) {
      hash_double(strong_branch_down[k]);
      hash_double(strong_branch_up[k]);
    }
    return hash;
  }

  std::vector<f_t> pseudo_cost_sum_up;
  std::vector<f_t> pseudo_cost_sum_down;
  std::vector<i_t> pseudo_cost_num_up;
  std::vector<i_t> pseudo_cost_num_down;
  std::vector<f_t> strong_branch_down;
  std::vector<f_t> strong_branch_up;

  omp_mutex_t mutex;
  omp_atomic_t<i_t> num_strong_branches_completed = 0;
};

template <typename i_t, typename f_t>
void strong_branching(const lp_problem_t<i_t, f_t>& original_lp,
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
