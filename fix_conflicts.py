#!/usr/bin/env python3
import re
import subprocess
import sys

def resolve_conflicts(content):
    pattern = r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> dev\n'
    def replace_match(match):
        return match.group(2)
    return re.sub(pattern, replace_match, content, flags=re.DOTALL)

# Find all files with conflicts
result = subprocess.run(
    ['grep', '-r', '<<<<<<', '--include=*.py', '--include=*.rst', '--include=*.md', '.'],
    capture_output=True, text=True
)

files = set()
for line in result.stdout.split('\n'):
    if line and '.venv' not in line:
        filepath = line.split(':')[0]
        files.add(filepath)

print(f"Found {len(files)} files with conflicts")

for filepath in files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        resolved = resolve_conflicts(content)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(resolved)
        print(f"Fixed {filepath}")
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")

print("Done!")