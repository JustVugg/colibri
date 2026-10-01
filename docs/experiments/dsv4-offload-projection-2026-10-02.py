"""Render a conditional decode projection from existing records; never run inference."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
STEM = 'dsv4-offload-projection-2026-10-02'
CPU_MS = 28.146735857142847
RESIDUAL_MS = 13.215792657142867
CPU_GB_4 = 1.895102144
GPU_GB_4 = 9.32219956
# Active checkpoint bytes from the retained hardware-roofline inventory.
# At zero GPUs the vocabulary head moves into the CPU byte total.
INPUTS = [
    (4, 35, 1.895102144, 9.322199560),
    (3, 26, 4.037719512, 7.179582192),
    (2, 17, 6.159081968, 5.058219736),
    (1, 8, 8.301699336, 2.915602368),
    (0, 0, 11.217301704, 0.0),
]
rows = []
for cards, gpu_layers, cpu_gb, gpu_gb in INPUTS:
    cpu_layers = 43 - gpu_layers
    cpu_ms = CPU_MS * cpu_gb / CPU_GB_4
    lower = 1000 / (cpu_ms + RESIDUAL_MS)
    upper = 1000 / (cpu_ms + RESIDUAL_MS * gpu_gb / GPU_GB_4)
    rows.append(dict(
        gpus=cards, gpu_layers=gpu_layers, cpu_layers=cpu_layers,
        cpu_active_weight_GB=cpu_gb, gpu_active_weight_GB=gpu_gb,
        fixed_residual_scenario_decode_tps=lower,
        scaled_residual_scenario_decode_tps=upper,
        measured=cards == 4,
        minimum_full_preload_slots=cpu_layers * 256 + gpu_layers * 16,
        locked_checkpoint_plus_CPU_experts_lower_bound_GiB=(
            155.425 + cpu_layers * 256 * 13369344 / 2**30),
    ))
result = dict(
    scope='Conditional C1 target-only short-context decode projection; CPU execution offload',
    source='Existing hardware-roofline weight inventory and attention-detail-profile24; no new inference',
    equation='T_ms = C4 * Wcpu/Wcpu4 + F + (R4-F) * Wgpu/Wgpu4; 0 <= F <= R4',
    cpu_ms_4=CPU_MS, residual_ms_4=RESIDUAL_MS,
    uninstrumented_four_GPU_decode_tps=[24.1431982552, 24.2722],
    assumptions=[
        'CPU elapsed cost scales with active weight bytes at unchanged optimized efficiency',
        'Residual allocation ranges between fully fixed and proportional to remaining GPU weight bytes',
        'Neither assumption is independently calibrated at other placements',
        'Scenario spread is not a statistical confidence interval or guaranteed bound',
        'NUMA ownership and cache behavior must be preserved or redesigned at each placement',
        'Pure CPU needs a separate vocabulary-head calibration and cache path',
    ],
    current_cache_slots=43 * 66,
    constraints=[
        'Current full-tail preload slot budget is insufficient for three or fewer GPUs',
        'One and zero GPUs exceed approximately 251 GiB RAM with the locked checkpoint plus full private CPU expert copies alone',
        'Memory lower bounds exclude dense weights, extra packing, KV, scratch, allocator overhead and the OS',
        'Advanced prototype optimizations are not all integrated into the PR implementation',
        'Historical prefill-inclusive offload results are not revised or plotted as decode results',
    ], rows=rows)
(ROOT / (STEM + '.json')).write_text(json.dumps(result, indent=2) + '\n')

plt.rcParams.update({'font.size': 11, 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(9, 5.4), layout='constrained')
ordered = list(reversed(rows))
x = [r['gpus'] for r in ordered]
low = [r['fixed_residual_scenario_decode_tps'] for r in ordered]
high = [r['scaled_residual_scenario_decode_tps'] for r in ordered]
ax.fill_between(x, low, high, color='#4776b4', alpha=.14)
ax.plot(x, low, '--', color='#7890ac', label='Projection: residual stays fixed')
ax.plot(x, high, '--', color='#305b96', label='Projection: residual scales with GPU weight bytes')
ax.scatter([4], [1000/(CPU_MS+RESIDUAL_MS)], color='#c55d23', s=65,
           zorder=4, label='4 GPUs: measured prototype ~24.1–24.3')
for r in ordered:
    i = r['gpus']
    label = ('24.1–24.3 measured' if i == 4 else
             f"{r['fixed_residual_scenario_decode_tps']:.1f}–{r['scaled_residual_scenario_decode_tps']:.1f} estimated")
    ax.annotate(label, (i, r['scaled_residual_scenario_decode_tps']),
                xytext=(-5 if i == 4 else 0, 12), textcoords='offset points',
                ha='right' if i == 4 else 'center', fontsize=10)
ax.set_xticks(x, ['0\n43 CPU layers', '1\n35 CPU layers', '2\n26 CPU layers',
                  '3\n17 CPU layers', '4\n8 CPU layers'])
ax.set(xlabel='GPU count / target-layer placement', ylabel='Decode tokens/s',
       title='Optimized CPU offload: conditional projection, not a new benchmark',
       ylim=(0, 29), xlim=(-.4, 4.25))
ax.grid(axis='y', alpha=.2)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper left', fontsize=9, frameon=False)
fig.supxlabel('0–3 GPU points require cache/NUMA adaptation. Shading is scenario spread, not a confidence interval.\n'
              'Pure CPU is least certain; unchanged RAM locking + full private copies cannot fit at 0–1 GPUs.',
              fontsize=9)
fig.savefig(ROOT / (STEM + '.svg'))
plt.close(fig)
print(json.dumps(rows, indent=2))
