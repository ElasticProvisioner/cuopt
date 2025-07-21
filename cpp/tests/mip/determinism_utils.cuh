/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <thread>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>

namespace cuopt::linear_programming::test {

static __global__ void spin_kernel(volatile int* flag, unsigned long long timeout_clocks = 10000000)
{
  long long int start_clock, sample_clock;
  start_clock = clock64();

  while (!*flag) {
    sample_clock = clock64();

    if (sample_clock - start_clock > timeout_clocks) { break; }
  }
}

static void launch_spin_kernel_stream_thread(rmm::cuda_stream_view stream_view)
{
  rmm::device_scalar<int> flag(0, stream_view);
  while (true) {
    int blocks  = rand() % 64 + 1;
    int threads = rand() % 1024 + 1;
    spin_kernel<<<blocks, threads, 0, stream_view>>>(flag.data());
    cudaStreamSynchronize(stream_view);
    std::this_thread::sleep_for(std::chrono::milliseconds(rand() % 1000 + 1));
  }
}

static inline void launch_spin_kernel_stream(rmm::cuda_stream_view stream_view)
{
  std::thread spin_thread(launch_spin_kernel_stream_thread, stream_view);
  spin_thread.detach();
}
}  // namespace cuopt::linear_programming::test