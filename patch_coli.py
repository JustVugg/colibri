from pathlib import Path
p = Path('c/coli')
text = p.read_text(encoding='utf-8')
old = '    banner(f"chat · {os.path.basename(a.model)} · ram {a.ram or \'-\'}GB · topp {a.topp or \'off\'}")\n'
new = '    ram=getattr(a, "ram", None)\n    topp=getattr(a, "topp", None)\n    banner(f"chat · {os.path.basename(a.model)} · ram {ram or \'-\'}GB · topp {topp or \'off\'}")\n'
count = text.count(old)
text = text.replace(old, new)
p.write_text(text, encoding='utf-8')
print(f'updated {count} occurrence(s)')
