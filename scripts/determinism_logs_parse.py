#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Parse log files containing algorithm feature logs and export to pickle format for training.

Supports parsing of:
- FP (Feasibility Pump): FP_FEATURES and FP_RESULT logs
- PDLP (LP Solver): PDLP_FEATURES and PDLP_RESULT logs
- CP (Constraint Propagation): CP_FEATURES and CP_RESULT logs
- FJ (Feasibility Jump): Legacy FJ: format

IMPORTANT - Grep Specificity:
The parser uses EXACT pattern matching with grep to filter logs efficiently.
For example, when parsing FP logs:
- Grep pattern: 'FP_FEATURES:' and 'FP_RESULT:' (with colon)
- Matches ONLY predictor log lines, NOT general FP debug/info lines
- A log with 10,000 lines containing "FP" might have only 100 predictor lines
- Grep filters down to only the relevant lines before Python processing

Performance optimizations for very large log files:
- Single grep call per algorithm (instead of separate calls for features/results)
- Uses grep's -n flag to get line numbers for efficient pairing
- Minimal Python string processing (split instead of regex)
- Single-pass parsing with dictionary accumulation
- Avoids redundant string operations
- DRY refactoring: Generic parser eliminates duplicate code (~105 lines removed)
- Real-time progress indicators (every 10K lines, every 10 files)

Usage:
    python determinism_logs_parse.py <input_directory> --algorithm FP [-o output.feather]
    python determinism_logs_parse.py <input_directory> --algorithm PDLP [-o output.feather]
    python determinism_logs_parse.py <input_directory> --algorithm CP [-o output.feather]
    python determinism_logs_parse.py <input_directory> --algorithm FJ [-o output.feather]
"""

import argparse
import subprocess
import os
import glob
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple


SUPPORTED_ALGORITHMS = ['FP', 'PDLP', 'CP', 'FJ']


def parse_value(value_str: str) -> Any:
    """Convert string value to appropriate type (int, float, or str)."""
    try:
        # Handle special float values first (nan, inf, -inf)
        value_lower = value_str.lower()
        if value_lower in ('nan', 'inf', '-inf', '+inf'):
            return float(value_str)

        # Try to parse as float if it contains a decimal point or scientific notation
        if '.' in value_str or 'e' in value_str.lower():
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        # Keep as string if conversion fails
        return value_str


def parse_key_value_line(line: str, prefix: str) -> Dict[str, Any]:
    """
    Parse a line containing key=value pairs after removing prefix.

    Example
    -------
        "FP_FEATURES: n_variables=100 n_constraints=50"
        -> {'n_variables': 100, 'n_constraints': 50}
    """
    entry = {}

    # Remove prefix
    if prefix in line:
        line = line.split(prefix, 1)[1].strip()

    # Parse key=value pairs
    # Handle both space-separated and comma-separated
    for kv_pair in line.split():
        if '=' in kv_pair:
            key, value = kv_pair.split('=', 1)
            # Remove trailing commas
            value = value.rstrip(',')
            entry[key] = parse_value(value)

    return entry


def parse_generic_algorithm_logs(log_files: List[str],
                                 algorithm: str,
                                 algorithm_name: str) -> List[Dict[str, Any]]:
    """
    Generic parser for algorithm feature and result logs.

    Matches <ALGORITHM>_FEATURES lines with subsequent <ALGORITHM>_RESULT lines.
    Uses grep efficiently to minimize Python-side processing.

    Args:
        log_files: List of log file paths to parse
        algorithm: Algorithm prefix (e.g., 'FP', 'PDLP', 'CP')
        algorithm_name: Full name for display (e.g., 'Feasibility Pump')

    Returns
    -------
        List of dictionaries with combined features and results
    """
    print(f"\nParsing {algorithm} ({algorithm_name}) logs...")
    print(f"  Running grep on {len(log_files)} files...")

    # Construct grep patterns with EXACT match requirements
    # The colon at the end ensures we ONLY match the feature/result log lines
    # and ignore all other lines containing the algorithm name
    features_pattern = f'{algorithm}_FEATURES:'
    result_pattern = f'{algorithm}_RESULT:'

    # Use grep with:
    # -H: Always print filename (even with single file)
    # -n: Print line numbers for correct pairing
    # -e: Multiple patterns to match
    # This ensures we ONLY get the specific predictor log lines, not debug/info lines
    cmd = ['grep', '-Hn', '-e', features_pattern, '-e', result_pattern] + log_files

    result = subprocess.run(cmd, capture_output=True, text=True)

    if not result.stdout:
        print(f"  No {algorithm} logs found")
        return []

    # Count lines for progress indication
    total_lines = result.stdout.count('\n')
    print(f"  Processing {total_lines} matching lines...")

    # Process grep output efficiently
    # Format: filename:linenum:<ALGORITHM>_FEATURES: key1=value1 ...
    entries_by_file = {}
    lines_processed = 0
    files_seen = set()

    for line in result.stdout.split('\n'):
        if not line:
            continue

        lines_processed += 1

        # Progress update every 10000 lines
        if lines_processed % 10000 == 0:
            pct = (lines_processed / total_lines * 100) if total_lines > 0 else 0
            print(f"  Progress: {pct:.1f}% ({lines_processed}/{total_lines} lines, {len(files_seen)} files)", end='\r')

        # Split on first two colons to get filename, linenum, and content
        # The rest of the line after the second colon is the log content
        parts = line.split(':', 2)
        if len(parts) < 3:
            continue

        filename = os.path.basename(parts[0])
        linenum = int(parts[1])
        content = parts[2]  # This includes everything after linenum

        if filename not in entries_by_file:
            entries_by_file[filename] = {'features': [], 'results': []}
            files_seen.add(filename)

        # Double-check pattern match (grep already filtered, but be extra safe)
        # This ensures we ONLY process lines with the exact patterns we want
        if features_pattern in content:
            # Parse features - only if pattern is present
            features = parse_key_value_line(content, features_pattern)
            if features:  # Only add if parsing succeeded
                entries_by_file[filename]['features'].append((linenum, features))
        elif result_pattern in content:
            # Parse results - only if pattern is present
            results = parse_key_value_line(content, result_pattern)
            if results:  # Only add if parsing succeeded
                entries_by_file[filename]['results'].append((linenum, results))

    # Clear progress line
    if lines_processed > 0:
        print(f"  Processed {lines_processed} lines from {len(files_seen)} files    ")

    # Match features with results
    # IMPORTANT: Multiple FEATURES lines followed by multiple RESULT lines form ONE complete entry
    # We need to merge consecutive lines of the same type, then combine them
    print(f"  Merging features and results...")
    entries = []
    files_processed = 0
    total_files = len(entries_by_file)

    for filename, data in entries_by_file.items():
        files_processed += 1

        # Progress update every 10 files
        if files_processed % 10 == 0 or files_processed == total_files:
            print(f"  Merging: {files_processed}/{total_files} files, {len(entries)} entries found", end='\r')

        # Combine features and results by line number
        # Group consecutive FEATURES lines and consecutive RESULT lines
        all_items = []
        for linenum, features in data['features']:
            all_items.append((linenum, 'features', features))
        for linenum, results in data['results']:
            all_items.append((linenum, 'results', results))

        # Sort by line number
        all_items.sort(key=lambda x: x[0])

        # Merge consecutive items of the same type
        current_features = {}
        current_results = {}
        last_type = None

        for linenum, item_type, content in all_items:
            # If we transition from RESULT back to FEATURES, save the previous entry
            if item_type == 'features' and last_type == 'results':
                if current_features and current_results:
                    # Create combined entry
                    entry = {'file': filename}
                    entry.update(current_features)
                    entry.update(current_results)

                    # Rename 'iterations' to 'iter' for consistency
                    if 'iterations' in entry:
                        entry['iter'] = entry.pop('iterations')

                    entries.append(entry)

                    # Reset for next entry
                    current_features = {}
                    current_results = {}

            if item_type == 'features':
                # Accumulate features
                current_features.update(content)
            else:  # results
                # Accumulate results
                current_results.update(content)

            last_type = item_type

        # Don't forget the last entry in the file
        if current_features and current_results:
            entry = {'file': filename}
            entry.update(current_features)
            entry.update(current_results)

            if 'iterations' in entry:
                entry['iter'] = entry.pop('iterations')

            entries.append(entry)

    # Clear progress line and show final count
    if total_files > 0:
        print(f"  Found {len(entries)} complete entries from {total_files} files       ")

    return entries


# Algorithm-specific wrappers for the generic parser
# These provide a clean API and eliminate code duplication

def parse_fp_logs(log_files: List[str]) -> List[Dict[str, Any]]:
    """Parse Feasibility Pump feature and result logs."""
    return parse_generic_algorithm_logs(log_files, 'FP', 'Feasibility Pump')


def parse_pdlp_logs(log_files: List[str]) -> List[Dict[str, Any]]:
    """Parse PDLP (LP Solver) feature and result logs."""
    return parse_generic_algorithm_logs(log_files, 'PDLP', 'LP Solver')


def parse_cp_logs(log_files: List[str]) -> List[Dict[str, Any]]:
    """Parse Constraint Propagation feature and result logs."""
    return parse_generic_algorithm_logs(log_files, 'CP', 'Constraint Propagation')


def parse_fj_logs(log_files: List[str]) -> List[Dict[str, Any]]:
    """
    Parse legacy Feasibility Jump logs (original format).

    Parses lines containing "FJ:" with key=value pairs.
    Uses grep efficiently to minimize Python-side processing.
    """
    print("\nParsing FJ (Feasibility Jump) legacy logs...")
    print(f"  Running grep on {len(log_files)} files...")

    # Use grep to efficiently extract ONLY lines with the exact "FJ:" pattern
    # Note: FJ uses legacy format with just "FJ:" prefix (not FJ_FEATURES/FJ_RESULT)
    # The colon ensures we don't match other FJ-related debug lines
    cmd = ['grep', '-H', 'FJ:'] + log_files
    result = subprocess.run(cmd, capture_output=True, text=True)

    if not result.stdout:
        print(f"  No FJ logs found")
        return []

    # Count lines for progress indication
    total_lines = result.stdout.count('\n')
    print(f"  Processing {total_lines} matching lines...")

    # Parse grep output efficiently
    entries = []
    lines_processed = 0

    for line in result.stdout.split('\n'):
        if not line:
            continue

        lines_processed += 1

        # Progress update every 10000 lines
        if lines_processed % 10000 == 0:
            pct = (lines_processed / total_lines * 100) if total_lines > 0 else 0
            print(f"  Progress: {pct:.1f}% ({lines_processed}/{total_lines} lines, {len(entries)} entries)", end='\r')

        # Grep output format: filename:FJ: key1=value1 key2=value2 ...
        parts = line.split(':', 2)
        if len(parts) < 3:
            continue

        filename = os.path.basename(parts[0])
        content = parts[2]

        # Remove "FJ:" prefix if present
        if content.startswith('FJ:'):
            content = content[3:].strip()

        # Parse key-value pairs
        entry = {'file': filename}
        for kv_pair in content.split():
            if '=' in kv_pair:
                key, value = kv_pair.split('=', 1)
                entry[key] = parse_value(value)

        # Only add entry if it has more than just the filename
        if len(entry) > 1:
            entries.append(entry)

    # Clear progress line
    if lines_processed > 0:
        print(f"  Found {len(entries)} entries from {total_lines} lines       ")

    return entries


def print_statistics(entries: List[Dict[str, Any]], algorithm: str) -> None:
    """Print statistics about parsed entries."""
    if not entries:
        print(f"\n  No entries found for {algorithm}")
        return

    unique_files = set(entry['file'] for entry in entries)
    avg_entries_per_file = len(entries) / len(unique_files) if unique_files else 0

    # Check if 'iter' field exists
    has_iter = all('iter' in entry for entry in entries)

    if has_iter:
        iter_values = [entry['iter'] for entry in entries]
        min_iter = min(iter_values)
        max_iter = max(iter_values)
        avg_iter = sum(iter_values) / len(iter_values)

        print(f"\n  Total entries: {len(entries)}")
        print(f"  Unique files: {len(unique_files)}")
        print(f"  Avg entries per file: {avg_entries_per_file:.2f}")
        print(f"  Iterations (target): min={min_iter}, max={max_iter}, avg={avg_iter:.2f}")
    else:
        print(f"\n  Total entries: {len(entries)}")
        print(f"  Unique files: {len(unique_files)}")
        print(f"  Avg entries per file: {avg_entries_per_file:.2f}")

    # Show sample entry
    if entries:
        print(f"\n  Sample entry (first):")
        sample = entries[0]
        for key, value in sorted(sample.items()):
            print(f"    {key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description='Parse algorithm feature logs and export to Feather format for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported Algorithms:
  FP    - Feasibility Pump (parses FP_FEATURES and FP_RESULT logs)
  PDLP  - LP Solver (parses PDLP_FEATURES and PDLP_RESULT logs)
  CP    - Constraint Propagation (parses CP_FEATURES and CP_RESULT logs)
  FJ    - Feasibility Jump (parses legacy FJ: format)

Examples:
  python determinism_logs_parse.py logs/ --algorithm FP -o fp_data.feather
  python determinism_logs_parse.py logs/ --algorithm PDLP -o pdlp_data.feather
  python determinism_logs_parse.py logs/ --algorithm CP -o cp_data.feather
  python determinism_logs_parse.py logs/ --algorithm FJ -o fj_data.feather

  # Limit to first 10 files for testing
  python determinism_logs_parse.py logs/ --algorithm FP --max-files 10
        """
    )

    parser.add_argument(
        'input_dir',
        help='Directory containing .log files to parse'
    )
    parser.add_argument(
        '--algorithm', '-a',
        required=True,
        choices=SUPPORTED_ALGORITHMS,
        help='Algorithm to parse logs for'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output Feather file path (default: <algorithm>_data.feather)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose output including warnings'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Limit number of log files to process (useful for testing)'
    )

    args = parser.parse_args()

    # Set default output filename based on algorithm
    if args.output is None:
        args.output = f"{args.algorithm.lower()}_data.feather"

    # Find all .log files in the input directory
    print(f"\nScanning {args.input_dir} for .log files...")
    log_files = glob.glob(os.path.join(args.input_dir, '*.log'))

    if not log_files:
        print(f"Error: No .log files found in {args.input_dir}")
        return 1

    print(f"Found {len(log_files)} log files")

    # Apply max-files limit if specified
    if args.max_files is not None and args.max_files > 0:
        if args.max_files < len(log_files):
            log_files = log_files[:args.max_files]
            print(f"Limiting to first {args.max_files} files (--max-files)")
        else:
            print(f"Note: --max-files={args.max_files} is >= total files, using all files")

    # Parse logs based on algorithm
    if args.algorithm == 'FP':
        entries = parse_fp_logs(log_files)
    elif args.algorithm == 'PDLP':
        entries = parse_pdlp_logs(log_files)
    elif args.algorithm == 'CP':
        entries = parse_cp_logs(log_files)
    elif args.algorithm == 'FJ':
        entries = parse_fj_logs(log_files)
    else:
        print(f"Error: Unsupported algorithm: {args.algorithm}")
        return 1

    if not entries:
        print(f"\nError: No entries found for {args.algorithm}")
        print(f"Make sure your logs contain {args.algorithm}_FEATURES and {args.algorithm}_RESULT lines")
        return 1

    # Print statistics
    print_statistics(entries, args.algorithm)

    # Convert to DataFrame
    df = pd.DataFrame(entries)

    # Validate: Check for NaN and infinite values
    print(f"\nValidating data integrity...")

    # Check for NaN
    nan_counts = df.isna().sum()
    columns_with_nan = nan_counts[nan_counts > 0]

    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count

    has_issues = len(columns_with_nan) > 0 or len(inf_counts) > 0

    if has_issues:
        print(f"\n{'='*70}")
        print(f"ERROR: Invalid data detected (NaN/inf values)!")
        print(f"{'='*70}")
        print(f"\nColumns with invalid values:")
        for col, count in columns_with_nan.items():
            pct = (count / len(df)) * 100
            print(f"  {col}: {count} NaN ({pct:.1f}%)")
        for col, count in inf_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {col}: {count} infinite ({pct:.1f}%)")

        # Show which log files have the issues
        rows_with_issues_mask = df.isna().any(axis=1)
        for col in inf_counts.keys():
            if col in df.columns:
                rows_with_issues_mask |= np.isinf(df[col])
        rows_with_nan = df[rows_with_issues_mask]
        problematic_files = rows_with_nan['file'].unique()
        print(f"\nProblematic log files ({len(problematic_files)} total):")
        for i, filename in enumerate(sorted(problematic_files)[:20], 1):
            count_in_file = len(rows_with_nan[rows_with_nan['file'] == filename])
            print(f"  {i}. {filename}: {count_in_file} invalid entries")
        if len(problematic_files) > 20:
            print(f"  ... and {len(problematic_files) - 20} more files")

        print(f"\nSample rows with invalid data:")
        cols_to_show = ['file']
        cols_to_show.extend([col for col in columns_with_nan.index])
        cols_to_show.extend([col for col in inf_counts.keys() if col not in columns_with_nan.index])
        cols_to_show = [col for col in cols_to_show if col in df.columns]
        print(rows_with_nan[cols_to_show].head(10))

        print(f"\nPossible causes:")
        print(f"  1. Algorithm failed/crashed during execution (produced 'nan' in logs)")
        print(f"  2. Incomplete log entries (missing FEATURES or RESULT lines)")
        print(f"  3. Log format doesn't match expected pattern")
        print(f"\nAction required:")
        print(f"  - Review the problematic log files listed above")
        print(f"  - Fix the underlying issues causing NaN values")
        print(f"  - Re-run the algorithm on those problem instances")
        print(f"{'='*70}\n")
        return 1

    print(f"  ✅ No missing values detected")

    # Filter out negative iterations (invalid data)
    if 'iter' in df.columns:
        negative_mask = df['iter'] < 0
        negative_count = negative_mask.sum()

        if negative_count > 0:
            print(f"\n⚠️  Found {negative_count} entries with negative iterations ({negative_count/len(df)*100:.2f}%)")

            # Show which files have negative iterations
            negative_entries = df[negative_mask]
            problematic_files = negative_entries['file'].unique()
            if len(problematic_files) <= 10:
                print(f"   Affected files: {', '.join(sorted(problematic_files))}")
            else:
                print(f"   Affected files: {len(problematic_files)} files")

            # Drop negative iterations
            df = df[~negative_mask].reset_index(drop=True)
            print(f"   → Dropped negative entries, remaining: {len(df)} entries")

            # Check if we have any data left
            if len(df) == 0:
                print(f"\n❌ Error: No valid entries remaining after filtering!")
                print(f"   All entries had negative iterations.")
                return 1

    # Print obtained features (all columns)
    print(f"\nObtained Features ({len(df.columns)} total):")
    print(f"{'='*70}")

    # Separate metadata from actual features
    metadata_cols = ['file']
    target_cols = ['iter', 'iterations']  # Target variable (if present)

    feature_cols = [col for col in df.columns if col not in metadata_cols and col not in target_cols]

    # Print in categories
    if metadata_cols:
        meta_present = [col for col in metadata_cols if col in df.columns]
        if meta_present:
            print(f"  Metadata: {', '.join(meta_present)}")

    target_present = [col for col in target_cols if col in df.columns]
    if target_present:
        print(f"  Target:   {', '.join(target_present)}")

    if feature_cols:
        print(f"  Features ({len(feature_cols)}):")
        for i, col in enumerate(sorted(feature_cols), 1):
            # Print 3 features per line for readability
            if i % 3 == 1:
                print(f"    ", end="")
            print(f"{col:30s}", end="")
            if i % 3 == 0:
                print()  # Newline after 3 features
        if len(feature_cols) % 3 != 0:
            print()  # Final newline if needed

    print(f"{'='*70}")

    # Save to Feather file (Apache Arrow format for fast I/O)
    print(f"\nSaving {len(df)} entries to {args.output}...")
    df.to_feather(args.output, compression='lz4')

    # Get file size
    file_size_bytes = os.path.getsize(args.output)
    file_size_mb = file_size_bytes / (1024 * 1024)

    print(f"\n{'='*70}")
    print(f"✓ Success! Saved {len(df)} entries to {args.output}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"  1. View available features:")
    print(f"     python scripts/train_regressor.py {args.output} --regressor xgboost --list-features")
    print(f"  2. Train a model:")
    print(f"     python scripts/train_regressor.py {args.output} --regressor xgboost --seed 42")
    print(f"  3. Train with early stopping and C++ export:")
    print(f"     python scripts/train_regressor.py {args.output} --regressor xgboost --seed 42 --early-stopping 20 --treelite-compile 8")

    return 0


if __name__ == '__main__':
    exit(main())
