from pathlib import Path
p = Path(r'c:\Users\Emil\Documents\Coden\Projekte\Colibri\c\coli')
text = p.read_text(encoding='utf-8')
text = text.replace('p.stderr.read().decode("utf-8","replace")', '_ensure_stream(p, "stderr").read().decode("utf-8","replace")')
text = text.replace('p.stdout.readline()', '_ensure_stream(p, "stdout").readline()')
p.write_text(text, encoding='utf-8')
print('patched')
