from pathlib import Path
from itertools import islice
path = Path(r'c:\Users\Emil\Documents\Coden\Projekte\Colibri\c\coli')
lines = path.read_text(encoding='utf-8').splitlines()
for i, line in enumerate(lines, 1):
    if line.startswith('def cmd_chat(') or line.startswith('def stream_turn('):
        print(i, line)
        for j in range(i-3, i+8):
            if 1 <= j <= len(lines):
                print(f'{j:4}: {lines[j-1]}')
        print('---')
