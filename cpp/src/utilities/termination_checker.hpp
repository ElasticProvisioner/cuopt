/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/timer.hpp>

namespace cuopt::linear_programming {

/**
 * @brief Controls solver termination based on time limit, user interrupt, and parent termination.
 *
 * This class owns its own timer and automatically registers with the global interrupt handler.
 * It can be linked to a parent termination object to inherit
 * termination conditions.
 *
 * Usage:
 *   // Root termination (main solver)
 *   termination_checker_t termination(60.0, termination_checker_t::root_tag_t{});
 *
 *   // Slave termination (sub-MIP, linked to parent)
 *   termination_checker_t sub_termination(10.0, parent_termination);
 *
 */
class termination_checker_t {
 public:
  // Separate tag to force any declaration of a root termination checker to be explicit
  struct root_tag_t {};
  /**
   * @brief Construct a termination object.
   * @param time_limit Time limit in seconds.
   * @param parent Parent termination object to check for termination.
   */
  explicit termination_checker_t(double time_limit, const termination_checker_t& parent)
    : timer_(time_limit), parent_(&parent)
  {
  }
  explicit termination_checker_t(double time_limit, root_tag_t)
    : timer_(time_limit), parent_(nullptr)
  {
  }

  void set_termination_callback(bool (*termination_callback)(void*),
                                void* termination_callback_data)
  {
    termination_callback_      = termination_callback;
    termination_callback_data_ = termination_callback_data;
  }

  bool check() const
  {
    if (termination_callback_ != nullptr && termination_callback_(termination_callback_data_)) {
      return true;
    }
    if (timer_.check_time_limit()) { return true; }
    if (parent_ != nullptr && parent_->check()) { return true; }
    return false;
  }

  double get_time_limit() const { return timer_.get_time_limit(); }
  double remaining_time() const { return timer_.remaining_time(); }
  double elapsed_time() const { return timer_.elapsed_time(); }

 private:
  timer_t timer_;
  const termination_checker_t* parent_{nullptr};
  // avoid including <functional> which is heavy. this is a top-level header
  bool (*termination_callback_)(void*) = nullptr;
  void* termination_callback_data_     = nullptr;
};

}  // namespace cuopt::linear_programming
