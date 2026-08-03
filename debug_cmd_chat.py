from pathlib import Path
import importlib.util, os
HERE = r'c:\Users\Emil\Documents\Coden\Projekte\Colibri\c'
loader = importlib.util.spec_from_file_location('coli_debug', os.path.join(HERE,'coli'))
spec = loader
module = importlib.util.module_from_spec(spec)
loader.exec_module(module)
print('cmd_chat line', module.cmd_chat.__code__.co_firstlineno)
print('stream_turn line', module.stream_turn.__code__.co_firstlineno)
print('has _ensure_stream', hasattr(module, '_ensure_stream'))
