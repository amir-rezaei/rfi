# project_packer.py

import os
from pathlib import Path


def pack_project_to_txt(output_file='project_snapshot.txt', project_dir='.'):
    """
    Scans the project directory for .py files and packs them into a single .txt file.

    Each file's content is preceded by a header line indicating its relative path.

    Args:
        output_file (str): The name of the text file to create.
        project_dir (str): The root directory of the project to scan.
    """
    # Use pathlib for robust path handling
    root = Path(project_dir)
    py_files = sorted(list(root.glob('**/*.py')))

    # Exclude this script itself from being packed
    this_script_path = Path(__file__).resolve()

    count = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # Write a general header for the snapshot file
        f_out.write(f"# Project snapshot created on {__import__('datetime').datetime.now()}\n")
        f_out.write("# This file contains the contents of all .py files in the project.\n\n")

        for file_path in py_files:
            # Skip this file to avoid including the packer script in the output
            if file_path.resolve() == this_script_path:
                continue

            relative_path = file_path.relative_to(root)
            # Use forward slashes for cross-platform consistency
            header = f"----- ./{relative_path.as_posix()} -----"

            print(f"Packing: {relative_path}")

            f_out.write(f"{header}\n")
            try:
                content = file_path.read_text(encoding='utf-8')
                f_out.write(content)
                f_out.write("\n\n")  # Add newlines for separation
                count += 1
            except Exception as e:
                f_out.write(f"!!! ERROR READING FILE: {e} !!!\n\n")

    print(f"\nSuccessfully packed {count} files into '{output_file}'.")


def unpack_project_from_txt(input_file='project_snapshot.txt'):
    """
    Recreates the project file structure from a packed .txt file.

    It reads the text file, uses the header lines to determine file paths,
    and writes the content to those paths, overwriting existing files.

    Args:
        input_file (str): The packed text file to read from.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()

    current_path = None
    content_buffer = []
    files_created = 0

    print("Starting to unpack project files...")

    for line in lines:
        # Check if the line is a file path header
        if line.strip().startswith("----- ./") and "-----" in line:
            # If we were already writing a file, save it first
            if current_path and content_buffer:
                # Ensure directory exists
                dir_name = os.path.dirname(current_path)
                if dir_name and not os.path.exists(dir_name):
                    os.makedirs(dir_name, exist_ok=True)
                with open(current_path, 'w', encoding='utf-8') as f_out:
                    f_out.writelines(content_buffer)
                print(f"Wrote: {current_path}")
                files_created += 1

            # Start a new file
            # Extract path from '----- ./path/to/file.py -----'
            current_path = line.strip().split(" ")[1][2:]
            content_buffer = []
        # Ignore comments and blank lines at the top level
        elif current_path is None and (line.startswith("#") or not line.strip()):
            continue
        else:
            content_buffer.append(line)

    # Write the last file in the buffer
    if current_path and content_buffer:
        os.makedirs(os.path.dirname(current_path), exist_ok=True)
        with open(current_path, 'w', encoding='utf-8') as f_out:
            # Join the buffer and remove the extra newlines added during packing
            f_out.write("".join(content_buffer).strip())
        print(f"Wrote: {current_path}")
        files_created += 1

    print(f"\nSuccessfully created or overwrote {files_created} files.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="A tool to pack a Python project into a single .txt file or unpack it."
    )
    parser.add_argument(
        'action', choices=['pack', 'unpack'],
        help="The action to perform: 'pack' to create a .txt file, 'unpack' to recreate files."
    )
    parser.add_argument(
        '-f', '--file', default='project_snapshot.txt',
        help="The name of the input/output text file."
    )
    parser.add_argument(
        '-d', '--dir', default='.',
        help="The project directory to pack from (used only with 'pack')."
    )

    args = parser.parse_args()

    if args.action == 'pack':
        pack_project_to_txt(output_file=args.file, project_dir=args.dir)
    elif args.action == 'unpack':
        unpack_project_from_txt(input_file=args.file)