#!/usr/bin/env python3

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Analyze clang-tidy error logs and extract narrowing conversion issues.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


class NarrowingError:
    """Represents a narrowing conversion error."""

    def __init__(
        self,
        file_path: str,
        line: str,
    ):
        self.file_path = file_path
        self.line = line

    def __repr__(self):
        return f"NarrowingError(file={self.file_path}, line={self.line})"

    def to_dict(self):
        return {
            'file_path': self.file_path,
            'line': self.line,
        }


def parse_clang_tidy_log(log_file: str) -> list[NarrowingError]:
    """
    Parse clang-tidy log file and extract narrowing conversion errors.

    Args:
        log_file: Path to the clang-tidy log file

    Returns:
        List of NarrowingError objects
    """
    errors = []

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Simplified pattern - just match lines with "error: narrowing conversion"
    # Example: /path/to/file.cc:97:40: error: narrowing conversion from 'long' to signed type 'int'...
    lines = content.split('\n')

    for i, line in enumerate(lines):
        if 'error: narrowing conversion' in line:
            match = line.split(" ")[0]
            file_path = match.split(":")[0]
            errors.append(NarrowingError(file_path, line))

    return errors


def classify_errors_by_file(
    errors: list[NarrowingError],
) -> dict[str, list[NarrowingError]]:
    """
    Classify errors by file path.

    Args:
        errors: List of NarrowingError objects

    Returns:
        Dictionary mapping file paths to lists of errors
    """
    classified = defaultdict(list)

    for error in errors:
        classified[error.file_path].append(error)

    # Sort errors by line number within each file
    for file_path in classified:
        classified[file_path].sort(key=lambda e: e.line)

    return dict(classified)


def create_chunks(
    classified_errors: dict[str, list[NarrowingError]],
    max_errors_per_chunk: int,
    min_errors_per_chunk: int,
) -> list[tuple[str, dict[str, list[NarrowingError]]]]:
    """
    Create chunks of errors for output files.

    Args:
        classified_errors: Dictionary mapping file paths to error lists
        max_errors_per_chunk: Maximum number of errors per chunk
        min_errors_per_chunk: Minimum errors to justify a separate file

    Returns:
        List of (chunk_name, chunk_data) tuples
    """
    chunks = []
    small_files = {}  # Files with few errors
    chunk_counter = 1

    # Sort files by number of errors (descending)
    sorted_files = sorted(
        classified_errors.items(), key=lambda x: len(x[1]), reverse=True
    )

    for file_path, errors in sorted_files:
        error_count = len(errors)

        if error_count >= max_errors_per_chunk:
            # Split large file into multiple chunks
            num_chunks = (
                error_count + max_errors_per_chunk - 1
            ) // max_errors_per_chunk
            errors_per_sub_chunk = (error_count + num_chunks - 1) // num_chunks

            for i in range(num_chunks):
                start_idx = i * errors_per_sub_chunk
                end_idx = min((i + 1) * errors_per_sub_chunk, error_count)

                chunk_errors = errors[start_idx:end_idx]
                chunk_name = f"chunk_{chunk_counter: 03d}_large_{Path(file_path).name}_part{i + 1}"
                chunks.append((chunk_name, {file_path: chunk_errors}))
                chunk_counter += 1

        elif error_count >= min_errors_per_chunk:
            # Medium-sized file gets its own chunk
            chunk_name = f"chunk_{chunk_counter:03d}_{Path(file_path).name}"
            chunks.append((chunk_name, {file_path: errors}))
            chunk_counter += 1

        else:
            # Small file - accumulate for combined chunk
            small_files[file_path] = errors

    # Combine small files into chunks
    if small_files:
        current_small_chunk = {}
        current_error_count = 0

        for file_path, errors in small_files.items():
            current_small_chunk[file_path] = errors
            current_error_count += len(errors)

            if current_error_count >= min_errors_per_chunk:
                chunk_name = f"chunk_{chunk_counter:03d}_combined"
                chunks.append((chunk_name, current_small_chunk))
                chunk_counter += 1
                current_small_chunk = {}
                current_error_count = 0

        # Add remaining small files
        if current_small_chunk:
            chunk_name = f"chunk_{chunk_counter: 03d}_combined"
            chunks.append((chunk_name, current_small_chunk))

    return chunks


def generate_markdown_report(
    chunk_data: dict[str, list[NarrowingError]], chunk_name: str
) -> str:
    """
    Generate a markdown report for a chunk.

    Args:
        chunk_data: Dictionary mapping file paths to error lists
        chunk_name: Name of the chunk

    Returns:
        Markdown-formatted report string
    """
    total_errors = sum(len(errors) for errors in chunk_data.values())

    report = f"# Clang-Tidy Narrowing Conversion Errors - {chunk_name}\n\n"
    report += f"**Total Errors in this chunk:** {total_errors}\n\n"
    report += f"**Files in this chunk:** {len(chunk_data)}\n\n"
    report += "---\n\n"

    for file_path, errors in sorted(chunk_data.items()):
        report += f"## File: `{file_path}`\n\n"
        report += f"**Number of errors:** {len(errors)}\n\n"
        report += "| Line | Description  |\n"
        report += "|------|-------------|\n"

        for error in errors:
            report += f"| | {error.line} |\n"

        report += "\n"

    return report


def generate_json_report(
    chunk_data: dict[str, list[NarrowingError]], chunk_name: str
) -> str:
    """
    Generate a JSON report for a chunk.

    Args:
        chunk_data: Dictionary mapping file paths to error lists
        chunk_name: Name of the chunk

    Returns:
        JSON-formatted report string
    """
    report_data = {
        'chunk_name': chunk_name,
        'total_errors': sum(len(errors) for errors in chunk_data.values()),
        'file_count': len(chunk_data),
        'files': {},
    }

    for file_path, errors in chunk_data.items():
        report_data['files'][file_path] = {
            'error_count': len(errors),
            'errors': [error.to_dict() for error in errors],
        }

    return json.dumps(report_data, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze clang-tidy logs for narrowing conversion errors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s -i clang_tidy_errors.log -o output_dir

  # Custom chunk sizes
  %(prog)s -i errors.log -o reports --max-chunk 100 --min-chunk 20

  # JSON output format
  %(prog)s -i errors.log -o reports --format json

  # Generate summary only
  %(prog)s -i errors.log --summary-only
        """,
    )

    parser.add_argument(
        '-i', '--input', required=True, help='Input clang-tidy log file'
    )

    parser.add_argument(
        '-o',
        '--output-dir',
        default='narrowing_errors_reports',
        help='Output directory for reports (default: narrowing_errors_reports)',
    )

    parser.add_argument(
        '--max-chunk',
        type=int,
        default=200,
        help='Maximum errors per chunk (default: 200)',
    )

    parser.add_argument(
        '--min-chunk',
        type=int,
        default=50,
        help='Minimum errors to justify separate file (default: 50)',
    )

    parser.add_argument(
        '--format',
        choices=['markdown', 'json', 'both'],
        default='markdown',
        help='Output format (default: markdown)',
    )

    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Generate only a summary report without chunks',
    )

    parser.add_argument(
        '-v', '--verbose', action='store_true', help='Verbose output'
    )

    args = parser.parse_args()

    # Parse the log file
    if args.verbose:
        print(f"Parsing log file: {args.input}")

    errors = parse_clang_tidy_log(args.input)

    if not errors:
        print("No narrowing conversion errors found in the log file.")
        return

    if args.verbose:
        print(f"Found {len(errors)} narrowing conversion errors")

    # Classify by file
    classified_errors = classify_errors_by_file(errors)

    if args.verbose:
        print(f"Errors found in {len(classified_errors)} files")

    # Generate summary
    summary = "# Clang-Tidy Narrowing Conversion Errors - Summary\n\n"
    summary += f"**Total Errors:** {len(errors)}\n\n"
    summary += f"**Total Files:** {len(classified_errors)}\n\n"
    summary += "## Errors by File\n\n"
    summary += "| File | Error Count |\n"
    summary += "|------|-------------|\n"

    for file_path, file_errors in sorted(
        classified_errors.items(), key=lambda x: len(x[1]), reverse=True
    ):
        summary += f"| `{file_path}` | {len(file_errors)} |\n"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write summary
    summary_file = output_dir / "00_summary.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"Summary written to: {summary_file}")

    if args.summary_only:
        return

    # Create chunks
    if args.verbose:
        print(
            f"Creating chunks (max:  {args.max_chunk}, min: {args.min_chunk})"
        )

    chunks = create_chunks(classified_errors, args.max_chunk, args.min_chunk)

    if args.verbose:
        print(f"Created {len(chunks)} chunks")

    # Generate reports for each chunk
    for chunk_name, chunk_data in chunks:
        if args.format in ['markdown', 'both']:
            md_report = generate_markdown_report(chunk_data, chunk_name)
            md_file = output_dir / f"{chunk_name}.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(md_report)
            if args.verbose:
                print(f"Generated:  {md_file}")

        if args.format in ['json', 'both']:
            json_report = generate_json_report(chunk_data, chunk_name)
            json_file = output_dir / f"{chunk_name}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                f.write(json_report)
            if args.verbose:
                print(f"Generated: {json_file}")

    print(f"\nAnalysis complete! Reports saved to: {output_dir}")
    print(f"Total chunks generated: {len(chunks)}")


if __name__ == '__main__':
    main()
