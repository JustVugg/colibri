from pathlib import Path
import re
path = Path(r'c:\Users\Emil\Documents\Coden\Projekte\Colibri\c\coli')
text = path.read_text(encoding='utf-8')
starts = [m.start() for m in re.finditer(r'^def cmd_chat\(a\):', text, flags=re.MULTILINE)]
if len(starts) < 2:
    raise SystemExit('expected at least 2 cmd_chat definitions')
# Extract the first cmd_chat block and replace the last one with it.
def next_def_index(start):
    m = re.search(r'^def cmd_[a-zA-Z_]*\(a\):', text[start+1:], flags=re.MULTILINE)
    return start + 1 + m.start() if m else len(text)
first_start = starts[0]
first_end = next_def_index(first_start)
last_start = starts[-1]
last_end = next_def_index(last_start)
first_block = text[first_start:first_end]
text = text[:last_start] + first_block + text[last_end:]
# Patch stream_turn blocks to return {} when stdout is missing.
text = text.replace("    reader=_ensure_stream(p, 'stdout')\n", "    if getattr(p, 'stdout', None) is None:\n        return {}\n    reader=p.stdout\n")
text = text.replace("    reader=_ensure_stream(p, \"stdout\")\n", "    if getattr(p, 'stdout', None) is None:\n        return {}\n    reader=p.stdout\n")
# Write back.
path.write_text(text, encoding='utf-8')
print('fixed')
