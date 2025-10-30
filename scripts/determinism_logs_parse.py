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
Parse log files containing key-value pairs and export to pickle format.

Usage:
    python determinism_logs_parse.py <input_directory> [-o output.pkl]
"""

import argparse
import pickle
import subprocess
import os
import glob


def parse_value(value_str):
    """Convert string value to appropriate type (int, float, or str)."""
    try:
        # Try to parse as float if it contains a decimal point or scientific notation
        if '.' in value_str or 'e' in value_str.lower():
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        # Keep as string if conversion fails
        return value_str


def main():
    parser = argparse.ArgumentParser(
        description='Parse log files with key-value pairs and export to pickle'
    )
    parser.add_argument(
        'input_dir',
        help='Directory containing .log files to parse'
    )
    parser.add_argument(
        '-o', '--output',
        default='output.pkl',
        help='Output pickle file path (default: output.pkl)'
    )
    args = parser.parse_args()

    # Find all .log files in the input directory
    log_files = glob.glob(os.path.join(args.input_dir, '*.log'))

    if not log_files:
        print(f"No .log files found in {args.input_dir}")
        return

    print(f"Found {len(log_files)} log files")

    # Use grep to efficiently extract all lines containing "FJ:"
    # -H flag ensures filename is included in output (even with single file)
    # -h suppresses filename (we don't want that)
    cmd = ['grep', '-H', 'FJ:'] + log_files
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 and result.returncode != 1:
        # grep returns 1 if no matches found, which is fine
        # Other return codes indicate actual errors
        print(f"Error running grep: {result.stderr}")
        return

    # Parse grep output
    entries = []
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue

        # Grep output format: filename:FJ: key1=value1 key2=value2 ...
        # Split on first colon to separate filename from content
        colon_idx = line.find(':')
        if colon_idx == -1:
            continue

        filename = os.path.basename(line[:colon_idx])
        rest = line[colon_idx + 1:]

        # Remove "FJ:" prefix if present
        if rest.startswith('FJ:'):
            rest = rest[3:].strip()

        # Parse key-value pairs
        entry = {'file': filename}
        for kv_pair in rest.split():
            if '=' in kv_pair:
                key, value = kv_pair.split('=', 1)
                entry[key] = parse_value(value)

        # Only add entry if it has more than just the filename
        if len(entry) > 1:
            entries.append(entry)

    # Calculate statistics
    unique_files = set(entry['file'] for entry in entries)
    avg_entries_per_file = len(entries) / len(unique_files) if unique_files else 0

    # Save to pickle file
    with open(args.output, 'wb') as f:
        pickle.dump(entries, f)

    print(f"Parsed {len(entries)} entries from {len(unique_files)} log files")
    print(f"Average entries per file: {avg_entries_per_file:.2f}")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
