from pathlib import Path
p = Path(r'c:\Users\Emil\Documents\Coden\Projekte\Colibri\c\coli')
text = p.read_text(encoding='utf-8')
old = '    sp=Spinner("waking the giant (744B)…"); sp.start()\n    st=stream_turn(p, READY, lambda b: None)'
new = '    sp=Spinner("waking the giant (744B)…"); sp.start()\n    if getattr(p, "stdout", None) is None:\n        sp.stop(); return\n    st=stream_turn(p, READY, lambda b: None)'
if old not in text:
    raise SystemExit('pattern not found')
text = text.replace(old, new)
p.write_text(text, encoding='utf-8')
print('patched')
