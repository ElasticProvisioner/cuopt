/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <mps_parser/parser.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <cmath>

#include <rmm/mr/pool_memory_resource.hpp>

#include "benchmark_helper.hpp"

template <typename T>
auto host_copy(T const* device_ptr, size_t size, rmm::cuda_stream_view stream_view)
{
  if (!device_ptr) return std::vector<T>{};
  std::vector<T> host_vec(size);
  raft::copy(host_vec.data(), device_ptr, size, stream_view);
  stream_view.synchronize();
  return host_vec;
}

template <typename T>
auto host_copy(rmm::device_uvector<T> const& device_vec)
{
  return host_copy(device_vec.data(), device_vec.size(), device_vec.stream());
}

bool is_frational(double in)
{
  return std::fabs(in - std::round(in)) > 1e-5;
}

template <typename T>
void print(std::string_view const name, std::vector<T> const& container)
{
  std::cout << name << "=[";
  for (auto const& item : container) {
    std::cout << item << ",";
  }
  std::cout << "]\n";
}

std::tuple<cuopt::mps_parser::mps_data_model_t<int, double>, std::vector<cuopt::mps_parser::mps_data_model_t<int, double>>, std::vector<std::pair<int, double>>> create_batch_problem(const cuopt::mps_parser::mps_data_model_t<int, double>& op_problem, const cuopt::linear_programming::optimization_problem_solution_t<int, double>& solution, bool no_init_lower)
{ 
  std::vector<double> primal_sol = host_copy(solution.get_primal_solution());
  
  std::vector<std::pair<int, double>> pairs;
  for (size_t i = 0; i < op_problem.get_variable_types().size(); ++i)
  {
    auto c = op_problem.get_variable_types()[i];
    // Is integer in the MIP problem and current solution is factional
    if (c == 'I' && is_frational(primal_sol[i]))
      pairs.emplace_back(i, primal_sol[i]);
  }
  
  const int batch_size = pairs.size();

  if (batch_size == 0)
  {
    std::cout << "No fractional var, exiting" << std::endl;
    exit(0);
  }
  std::cout << "Found " << batch_size << " factional integer variables" << std::endl;

  // Create the problem batch to solve them individually

  const int total_size = no_init_lower ? batch_size : batch_size * 2;
  if (no_init_lower)
    std::cout << "Running strong branching with only lower bounds" << std::endl;
  else
    std::cout << "Running strong branching with lower and upper bounds" << std::endl;

  std::cout << "Runnning strong branching on " << total_size << " problems" << std::endl;

  std::vector<cuopt::mps_parser::mps_data_model_t<int, double>> problems(total_size, op_problem);

  // Create the upper bounds
  for (int i = 0; i < batch_size; i++)
    problems[i].get_variable_upper_bounds()[pairs[i].first] = std::floor(pairs[i].second);
  // Create the lower bounds
  if (!no_init_lower)
    for (int i = 0; i < batch_size; i++)
      problems[i + batch_size].get_variable_lower_bounds()[pairs[i].first] = std::ceil(pairs[i].second);
  // Create batch problem on the original problem
  cuopt::mps_parser::mps_data_model_t<int, double> batch_problem(op_problem);
  /*const auto& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const auto& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  std::vector<double> new_variable_lower_bounds(variable_lower_bounds.size() * total_size);
  std::vector<double> new_variable_upper_bounds(variable_upper_bounds.size() * total_size);

  for (int i = 0; i < total_size; i++)
    for (size_t j = 0; j < variable_lower_bounds.size(); ++j)
      new_variable_lower_bounds[i * variable_lower_bounds.size() + j] = problems[i].get_variable_lower_bounds()[j];
  for (int i = 0; i < total_size; i++)
    for (size_t j = 0; j < variable_upper_bounds.size(); ++j)
      new_variable_upper_bounds[i * variable_upper_bounds.size() + j] = problems[i].get_variable_upper_bounds()[j];

  batch_problem.set_variable_lower_bounds(new_variable_lower_bounds.data(), new_variable_lower_bounds.size());
  batch_problem.set_variable_upper_bounds(new_variable_upper_bounds.data(), new_variable_upper_bounds.size());*/

  return {batch_problem, problems, pairs};
}

static bool is_incorrect_objective(double reference, double objective)
{
  if (reference == 0) { return std::abs(objective) > 0.001; }
  if (objective == 0) { return std::abs(reference) > 0.001; }
  return std::abs((reference - objective) / reference) > 0.001;
}

template <
    class result_t   = std::chrono::milliseconds,
    class clock_t    = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

void bench(
  const raft::handle_t& handle,
  cuopt::mps_parser::mps_data_model_t<int, double>& original_problem, // Only useful for warm start
  cuopt::mps_parser::mps_data_model_t<int, double>& batch_problem,
  std::vector<cuopt::mps_parser::mps_data_model_t<int, double>>& problems,
  cuopt::linear_programming::optimization_problem_solution_t<int, double>& dual_simplex_solution,
  std::vector<std::pair<int, double>>& pairs,
  bool compare_with_baseline,
  bool init_primal_dual,
  bool init_step_size,
  bool init_primal_weight,
  bool use_optimal_batch_size,
  bool warm_start_from_dual_simplex,
  bool project_initial_primal)
{
  std::cout << "[Benchmark] Running with: " << "primal dual init: " << init_primal_dual << ", step size init: " << init_step_size << ", primal weight init: " << init_primal_weight << ", use optimal batch size: " << use_optimal_batch_size << ", warm start from dual simplex: " << warm_start_from_dual_simplex << ", project initial primal: " << project_initial_primal << std::endl;

  std::vector<cuopt::linear_programming::optimization_problem_solution_t<int, double>> sols;

  rmm::device_uvector<double> initial_primal(0, handle.get_stream());
  rmm::device_uvector<double> initial_dual(0, handle.get_stream());
  double initial_step_size = std::numeric_limits<double>::signaling_NaN();
  double initial_primal_weight = std::numeric_limits<double>::signaling_NaN();

  // store solve time for solving original problem if warm_start is used
  double warm_start_time = 0.0;
  bool needs_warm_start_solution =
      init_primal_dual || init_step_size || init_primal_weight;

  if (warm_start_from_dual_simplex)
  {
    if (init_primal_dual) {
      initial_primal = rmm::device_uvector<double>(dual_simplex_solution.get_primal_solution(), dual_simplex_solution.get_primal_solution().stream());
      initial_dual = rmm::device_uvector<double>(dual_simplex_solution.get_dual_solution(), dual_simplex_solution.get_dual_solution().stream());
    }
  }
  else if (needs_warm_start_solution)
  {
    // Solving the original to get its primal / dual vectors
    // Should not be necessary but weird behavior with conccurent halt is making things crash
    cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings_local;
    settings_local.method = cuopt::linear_programming::method_t::PDLP;
    settings_local.hyper_params.project_initial_primal = project_initial_primal;

    cuopt::linear_programming::optimization_problem_solution_t<int, double> original_solution = cuopt::linear_programming::solve_lp(&handle, original_problem, settings_local);

    std::cout << "Original problem solved by PDLP in " << original_solution.get_additional_termination_information().solve_time << " using " << original_solution.get_additional_termination_information().number_of_steps_taken << std::endl;
    if (init_primal_dual && !warm_start_from_dual_simplex) {
      initial_primal = rmm::device_uvector<double>(original_solution.get_primal_solution(), original_solution.get_primal_solution().stream());
      initial_dual = rmm::device_uvector<double>(original_solution.get_dual_solution(), original_solution.get_dual_solution().stream());
    }
    if (init_step_size) {
      initial_step_size = original_solution.get_pdlp_warm_start_data().initial_step_size_;
    }
    if (init_primal_weight) {
      initial_primal_weight = original_solution.get_pdlp_warm_start_data().initial_primal_weight_;
    }
    // Store the solve time for the original PDLP (for time correction)
    warm_start_time = original_solution.get_additional_termination_information().solve_time;
  }

  auto start = std::chrono::steady_clock::now();

  if (compare_with_baseline)
  {
    for (size_t i = 0; i < problems.size(); ++i)
    {
      // Should not be necessary but weird behavior with conccurent halt is making things crash
      cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings_local;
      settings_local.method = cuopt::linear_programming::method_t::PDLP;
      settings_local.iteration_limit = 100000;
      settings_local.hyper_params.project_initial_primal = project_initial_primal;

      if (init_primal_dual)
      {
        settings_local.set_initial_primal_solution(initial_primal.data(), initial_primal.size(), initial_primal.stream());
        settings_local.set_initial_dual_solution(initial_dual.data(), initial_dual.size(), initial_dual.stream());
      }
      if (init_step_size)
      {
        settings_local.set_initial_step_size(initial_step_size);
      }
      if (init_primal_weight)
      {
        settings_local.set_initial_primal_weight(initial_primal_weight);
      }

      sols.emplace_back(cuopt::linear_programming::solve_lp(&handle, problems[i], settings_local, true, true/*, "batch_instances/custom_" + std::to_string(i) + ".mps"*/));
      std::cout << "Version " << i << " solved " << sols[i].get_termination_status_string() << " using " << sols[i].get_additional_termination_information().number_of_steps_taken << std::endl;
    }
    double total_time_sec = since(start).count() / 1000.0;
    if (needs_warm_start_solution)
      total_time_sec += warm_start_time;
    std::cout << "All solved in " << total_time_sec << std::endl;
  }

  // Should not be necessary but weird behavior with conccurent halt is making things crash/mer
  cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings_local;
  settings_local.method = cuopt::linear_programming::method_t::PDLP;
  settings_local.iteration_limit = 100000;
  settings_local.hyper_params.project_initial_primal = project_initial_primal;
  constexpr bool only_upper = false;

  if (init_primal_dual)
  {
    settings_local.set_initial_primal_solution(initial_primal.data(), initial_primal.size(), initial_primal.stream());
    settings_local.set_initial_dual_solution(initial_dual.data(), initial_dual.size(), initial_dual.stream());
  }
  if (init_step_size)
  {
    settings_local.set_initial_step_size(initial_step_size);
  }
  if (init_primal_weight)
  {
    settings_local.set_initial_primal_weight(initial_primal_weight);
  }

    auto start_batch = std::chrono::steady_clock::now();
    cuopt::linear_programming::optimization_problem_solution_t<int, double> batch_solution(cuopt::linear_programming::pdlp_termination_status_t::NumericalError, handle.get_stream());
    if (use_optimal_batch_size)
    {
      std::vector<int> fractional;
      std::vector<double> root_soln_x;
      for (size_t i = 0; i < pairs.size(); ++i)
      {
        fractional.push_back(pairs[i].first);
        root_soln_x.push_back(pairs[i].second);
      }
      batch_solution = cuopt::linear_programming::batch_pdlp_solve(&handle, batch_problem, fractional, root_soln_x, settings_local);
    }
    else
    {
      for (size_t i = 0; i < pairs.size(); ++i)
        settings_local.new_bounds.push_back({pairs[i].first, problems[i].get_variable_lower_bounds()[pairs[i].first], std::floor(pairs[i].second)});
      if (!only_upper)
        for (size_t i = 0; i < pairs.size(); ++i)
          settings_local.new_bounds.push_back({pairs[i].first, std::ceil(pairs[i].second), problems[i].get_variable_upper_bounds()[pairs[i].first]});
      batch_solution = cuopt::linear_programming::solve_lp(&handle, batch_problem, settings_local, true, false);
    }
    cudaDeviceSynchronize();
    auto end_batch = std::chrono::steady_clock::now();
    std::cout << "Batch problem solved in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_batch - start_batch).count() / 1000.0 << " seconds using " << batch_solution.get_additional_termination_information().number_of_steps_taken << " steps" << std::endl;
    if (needs_warm_start_solution) {
      std::cout << "Total (including warm start original PDLP solve) batch solve time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_batch - start).count() / 1000.0 << " seconds" << std::endl;
    }

  if (compare_with_baseline)
  {
    for (size_t i = 0; i < sols.size(); ++i)
    {
      if (sols[i].get_termination_status() != batch_solution.get_termination_status(i))
              std::cout << "Terminations not equal at: " << i << " " << sols[i].get_termination_status_string() << " " << batch_solution.get_termination_status_string(i) << std::endl;

      if (is_incorrect_objective(sols[i].get_additional_termination_information().primal_objective, batch_solution.get_additional_termination_information(i).primal_objective))
        std::cout << "Objectives not equal at: " << i << " " << sols[i].get_additional_termination_information().primal_objective << " " << batch_solution.get_additional_termination_information(i).primal_objective << std::endl;
    }
  }
}

int main(int argc, char* argv[])
{
  // Initialize raft handle here to make sure it's destroyed very last
  const raft::handle_t handle_;

  cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings;


  // Setup up RMM memory pool
  auto memory_resource = make_pool();
  rmm::mr::set_current_device_resource(memory_resource.get());


  //std::vector<std::string> problem_list = {"scpj4scip", "neos8", "cod105"};//{"afiro_original"};//{"app1-1"};//;
  
  // Take problem name from command line
  std::string problem_name = argv[1];

  bool compare_with_baseline = false;
  //for (const auto& problem_name : problem_list)
  {
    // Parse MPS file
    cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>("batch_instances/" + problem_name + ".mps");

    // Solve using Dual Simplex to have the least amount of fractional variable
    settings.method = cuopt::linear_programming::method_t::DualSimplex;

    // Solve LP problem
    cuopt::linear_programming::optimization_problem_solution_t<int, double> solution(cuopt::linear_programming::pdlp_termination_status_t::NumericalError, handle_.get_stream());
    
    // If it doesn't already exist, dump the primal and dual solution
    if (!std::filesystem::exists("primal_solution_" + op_problem.get_problem_name() + ".txt")) {
      solution = cuopt::linear_programming::solve_lp(&handle_, op_problem, settings);
      std::cout << "Original problem solved in " << solution.get_additional_termination_information().solve_time << " and " << solution.get_additional_termination_information().number_of_steps_taken << " steps" << std::endl;
        const auto primal_solution = host_copy(solution.get_primal_solution());
        const auto dual_solution = host_copy(solution.get_dual_solution());
        std::ofstream file1("primal_solution_" + op_problem.get_problem_name() + ".txt");
        for (const auto& value : primal_solution)
          file1 << value << std::endl;
        file1.close();
        std::ofstream file2("dual_solution_" + op_problem.get_problem_name() + ".txt");
        for (const auto& value : dual_solution)
          file2 << value << std::endl;
        file2.close();
      }

    // Load the primal and dual solution from the files
    std::ifstream file1("primal_solution_" + op_problem.get_problem_name() + ".txt");
    std::ifstream file2("dual_solution_" + op_problem.get_problem_name() + ".txt");
    std::vector<double> primal_solution;
    std::vector<double> dual_solution;
    for (std::string line; std::getline(file1, line);)
      primal_solution.push_back(std::stod(line));
    for (std::string line; std::getline(file2, line);)
      dual_solution.push_back(std::stod(line));
    file1.close();
    file2.close();

    // Fill the primal and dual solution in the solution object
    solution.get_primal_solution().resize(primal_solution.size(), handle_.get_stream());
    solution.get_dual_solution().resize(dual_solution.size(), handle_.get_stream());
    raft::copy(solution.get_primal_solution().data(), primal_solution.data(), primal_solution.size(), handle_.get_stream());
    raft::copy(solution.get_dual_solution().data(), dual_solution.data(), dual_solution.size(), handle_.get_stream());

    // Create a list of problems for each variante and update op_problem to batchify it
    auto [batch_problem, problems, pairs] = create_batch_problem(op_problem, solution, false);

    std::vector<bool> primal_dual_init = {true, false};
    std::vector<bool> step_size_init = {true, false};
    std::vector<bool> primal_weight_init = {true, false};
    std::vector<bool> use_optimal_batch_size = {true, false};
    std::vector<bool> warm_start_from_dual_simplex = {true, false};
    std::vector<bool> project_initial_primal = {true, false};

    for (const auto& primal_dual_init : primal_dual_init) {
      for (const auto& step_size_init : step_size_init) {
        for (const auto& primal_weight_init : primal_weight_init) {
          for (const auto& use_optimal_batch_size : use_optimal_batch_size) {
            for (const auto& warm_start_from_dual_simplex : warm_start_from_dual_simplex) {
              for (const auto& project_initial_primal : project_initial_primal) {
                bench(handle_, op_problem, batch_problem, problems, solution, pairs, compare_with_baseline, primal_dual_init, step_size_init, primal_weight_init, use_optimal_batch_size, warm_start_from_dual_simplex, project_initial_primal);
              }
            }
          }
        }
      }
    }
  }

  return 0;
}