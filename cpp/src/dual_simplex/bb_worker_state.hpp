/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/bb_event.hpp>
#include <dual_simplex/bounds_strengthening.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/mip_node.hpp>
#include <dual_simplex/types.hpp>
#include <utilities/work_limit_timer.hpp>

#include <deque>
#include <memory>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

// Per-worker state for BSP (Bulk Synchronous Parallel) branch-and-bound
template <typename i_t, typename f_t>
struct bb_worker_state_t {
  int worker_id{0};

  // Local node queue - buffer of nodes assigned to this worker for the current horizon
  std::deque<mip_node_t<i_t, f_t>*> local_queue;

  // Current node being processed (may be paused at horizon boundary)
  mip_node_t<i_t, f_t>* current_node{nullptr};

  // Worker's virtual time clock (cumulative work units)
  double clock{0.0};

  // Events generated during this horizon
  bb_event_batch_t<i_t, f_t> events;

  // Event sequence counter for deterministic tie-breaking
  int event_sequence{0};

  // LP problem copy for this worker (bounds modified per node)
  std::unique_ptr<lp_problem_t<i_t, f_t>> leaf_problem;

  // Basis factorization state
  std::unique_ptr<basis_update_mpf_t<i_t, f_t>> basis_factors;

  // Bounds strengthening (node presolver)
  std::unique_ptr<bounds_strengthening_t<i_t, f_t>> node_presolver;

  // Working vectors for basis
  std::vector<i_t> basic_list;
  std::vector<i_t> nonbasic_list;

  // Work unit context for this worker
  work_limit_context_t work_context;

  // Whether basis needs recomputation for next node
  bool recompute_bounds_and_basis{true};

  // Statistics
  i_t nodes_processed_this_horizon{0};
  double work_units_this_horizon{0.0};

  // Constructor
  explicit bb_worker_state_t(int id) : worker_id(id), work_context("BB_Worker_" + std::to_string(id))
  {
  }

  // Initialize worker with problem data
  void initialize(const lp_problem_t<i_t, f_t>& original_lp,
                  const csr_matrix_t<i_t, f_t>& Arow,
                  const std::vector<variable_type_t>& var_types,
                  i_t refactor_frequency,
                  bool deterministic)
  {
    // Create copy of LP problem for this worker
    leaf_problem = std::make_unique<lp_problem_t<i_t, f_t>>(original_lp);

    // Initialize basis factors
    const i_t m = leaf_problem->num_rows;
    basis_factors = std::make_unique<basis_update_mpf_t<i_t, f_t>>(m, refactor_frequency);

    // Initialize bounds strengthening
    std::vector<char> row_sense;
    node_presolver =
      std::make_unique<bounds_strengthening_t<i_t, f_t>>(*leaf_problem, Arow, row_sense, var_types);

    // Initialize working vectors
    basic_list.resize(m);
    nonbasic_list.clear();

    // Configure work context
    work_context.deterministic = deterministic;
  }

  // Reset for new horizon
  void reset_for_horizon(double horizon_start, double horizon_end)
  {
    // Reset clock to horizon_start for consistent VT timestamps across workers
    clock = horizon_start;
    events.clear();
    events.horizon_start = horizon_start;
    events.horizon_end   = horizon_end;
    event_sequence       = 0;
    nodes_processed_this_horizon = 0;
    work_units_this_horizon = 0.0;
    // Also sync work_context to match clock for consistent tracking
    work_context.global_work_units_elapsed = horizon_start;
  }

  // Add a node to the local queue
  void enqueue_node(mip_node_t<i_t, f_t>* node) { local_queue.push_back(node); }

  // Get next node to process
  mip_node_t<i_t, f_t>* dequeue_node()
  {
    if (current_node != nullptr) {
      // Resume paused node
      mip_node_t<i_t, f_t>* node = current_node;
      current_node = nullptr;
      return node;
    }
    if (local_queue.empty()) { return nullptr; }
    mip_node_t<i_t, f_t>* node = local_queue.front();
    local_queue.pop_front();
    return node;
  }

  // Check if worker has work available
  bool has_work() const { return current_node != nullptr || !local_queue.empty(); }

  // Get number of nodes in local queue (including paused node)
  size_t queue_size() const
  {
    return local_queue.size() + (current_node != nullptr ? 1 : 0);
  }

  // Record an event
  void record_event(bb_event_t<i_t, f_t> event)
  {
    event.event_sequence = event_sequence++;
    events.add(std::move(event));
  }

  // Pause current node processing at horizon boundary
  void pause_current_node(mip_node_t<i_t, f_t>* node, double accumulated_vt)
  {
    node->accumulated_vt = accumulated_vt;
    node->bsp_state      = bsp_node_state_t::PAUSED;
    current_node         = node;

    record_event(bb_event_t<i_t, f_t>::make_paused(clock, worker_id, node->node_id, 0, accumulated_vt));
  }

  // Record node branching event
  void record_branched(mip_node_t<i_t, f_t>* node,
                       i_t down_child_id,
                       i_t up_child_id,
                       i_t branch_var,
                       f_t branch_val)
  {
    record_event(bb_event_t<i_t, f_t>::make_branched(clock,
                                                    worker_id,
                                                    node->node_id,
                                                    0,
                                                    down_child_id,
                                                    up_child_id,
                                                    node->lower_bound,
                                                    branch_var,
                                                    branch_val));
  }

  // Record integer solution found
  void record_integer_solution(mip_node_t<i_t, f_t>* node, f_t objective)
  {
    record_event(bb_event_t<i_t, f_t>::make_integer_solution(clock, worker_id, node->node_id, 0, objective));
  }

  // Record node fathomed
  void record_fathomed(mip_node_t<i_t, f_t>* node, f_t lower_bound)
  {
    record_event(bb_event_t<i_t, f_t>::make_fathomed(clock, worker_id, node->node_id, 0, lower_bound));
  }

  // Record node infeasible
  void record_infeasible(mip_node_t<i_t, f_t>* node)
  {
    record_event(bb_event_t<i_t, f_t>::make_infeasible(clock, worker_id, node->node_id, 0));
  }

  // Record numerical error
  void record_numerical(mip_node_t<i_t, f_t>* node)
  {
    record_event(bb_event_t<i_t, f_t>::make_numerical(clock, worker_id, node->node_id, 0));
  }

  // Update clock with work units
  void advance_clock(double work_units)
  {
    clock += work_units;
    work_units_this_horizon += work_units;
    work_context.record_work(work_units);
  }
};

// Container for all worker states in BSP B&B
template <typename i_t, typename f_t>
class bb_worker_pool_t {
 public:
  bb_worker_pool_t() = default;

  // Initialize pool with specified number of workers
  void initialize(int num_workers,
                  const lp_problem_t<i_t, f_t>& original_lp,
                  const csr_matrix_t<i_t, f_t>& Arow,
                  const std::vector<variable_type_t>& var_types,
                  i_t refactor_frequency,
                  bool deterministic)
  {
    workers_.clear();
    workers_.reserve(num_workers);
    for (int i = 0; i < num_workers; ++i) {
      workers_.emplace_back(i);
      workers_.back().initialize(original_lp, Arow, var_types, refactor_frequency, deterministic);
    }
  }

  // Get worker by ID
  bb_worker_state_t<i_t, f_t>& operator[](int worker_id) { return workers_[worker_id]; }

  const bb_worker_state_t<i_t, f_t>& operator[](int worker_id) const { return workers_[worker_id]; }

  // Get number of workers
  int size() const { return static_cast<int>(workers_.size()); }

  // Reset all workers for new horizon
  void reset_for_horizon(double horizon_start, double horizon_end)
  {
    for (auto& worker : workers_) {
      worker.reset_for_horizon(horizon_start, horizon_end);
    }
  }

  // Collect all events from all workers into a single sorted batch
  bb_event_batch_t<i_t, f_t> collect_and_sort_events()
  {
    bb_event_batch_t<i_t, f_t> all_events;
    for (auto& worker : workers_) {
      for (auto& event : worker.events.events) {
        all_events.add(std::move(event));
      }
      worker.events.clear();
    }
    all_events.sort_for_replay();
    return all_events;
  }

  // Check if any worker has work
  bool any_has_work() const
  {
    for (const auto& worker : workers_) {
      if (worker.has_work()) return true;
    }
    return false;
  }

  // Get total queue size across all workers
  size_t total_queue_size() const
  {
    size_t total = 0;
    for (const auto& worker : workers_) {
      total += worker.queue_size();
    }
    return total;
  }

  // Iterator support
  auto begin() { return workers_.begin(); }
  auto end() { return workers_.end(); }
  auto begin() const { return workers_.begin(); }
  auto end() const { return workers_.end(); }

 private:
  std::vector<bb_worker_state_t<i_t, f_t>> workers_;
};

}  // namespace cuopt::linear_programming::dual_simplex

