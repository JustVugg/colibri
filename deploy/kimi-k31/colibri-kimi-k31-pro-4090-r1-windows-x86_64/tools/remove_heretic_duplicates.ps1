# Deletes from the MiMo-V2.6-Pro-MOPD-heretic folder the 131 weight files that are byte-for-
# byte the base model's (each compared in full with its base copy on 2026-10-22), 539.1 GB in all.
# colibri then reads the shards from the base model's folders -- C:\AI\MiMoPro-Shards and
# D:\AI\Models\MiMo-V2.6-Pro-MOPD, two drives at once, as the base model reads them --
# and keeps only model_pp0_ep0_shard0 (the 70 edited o_proj) from the heretic folder
# (START-MIMO-HERETIC.cmd, tools\start_mimo_cuda_4090.ps1 -Model heretic). The weights in
# dflash\ and audio_tokenizer\ go too: colibri never reads them, and the base keeps the same.
#
# Run it once as it is: it checks every file and lists what it would delete. Then with
# -Apply to delete. It needs no administrator rights. Before deleting it checks again,
# for every file, that its base copy is there, that both sizes are the compared ones and
# that their first and last 8 MB are still equal; a file that fails is kept.
# Afterwards the heretic model needs the base model's folders: keep them.
param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$Heretic = 'D:\AI\Models\MiMo-V2.6-Pro-MOPD-heretic'
$Own = 'model_pp0_ep0_shard0.safetensors'
$Files = @(
    @('audio_tokenizer\model.safetensors', 1872618384, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\audio_tokenizer\model.safetensors'),
    @('dflash\dflash_draft_model.safetensors', 5536639384, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\dflash\dflash_draft_model.safetensors'),
    @('model_mtp.safetensors', 2463641280, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_mtp.safetensors'),
    @('model_pp0_ep0_shard1.safetensors', 2001997472, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep0_shard1.safetensors'),
    @('model_pp0_ep100_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep100_shard0.safetensors'),
    @('model_pp0_ep101_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep101_shard0.safetensors'),
    @('model_pp0_ep102_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep102_shard0.safetensors'),
    @('model_pp0_ep103_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep103_shard0.safetensors'),
    @('model_pp0_ep104_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep104_shard0.safetensors'),
    @('model_pp0_ep105_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep105_shard0.safetensors'),
    @('model_pp0_ep106_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep106_shard0.safetensors'),
    @('model_pp0_ep107_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep107_shard0.safetensors'),
    @('model_pp0_ep108_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep108_shard0.safetensors'),
    @('model_pp0_ep109_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep109_shard0.safetensors'),
    @('model_pp0_ep10_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep10_shard0.safetensors'),
    @('model_pp0_ep110_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep110_shard0.safetensors'),
    @('model_pp0_ep111_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep111_shard0.safetensors'),
    @('model_pp0_ep112_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep112_shard0.safetensors'),
    @('model_pp0_ep113_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep113_shard0.safetensors'),
    @('model_pp0_ep114_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep114_shard0.safetensors'),
    @('model_pp0_ep115_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep115_shard0.safetensors'),
    @('model_pp0_ep116_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep116_shard0.safetensors'),
    @('model_pp0_ep117_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep117_shard0.safetensors'),
    @('model_pp0_ep118_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep118_shard0.safetensors'),
    @('model_pp0_ep119_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep119_shard0.safetensors'),
    @('model_pp0_ep11_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep11_shard0.safetensors'),
    @('model_pp0_ep120_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep120_shard0.safetensors'),
    @('model_pp0_ep121_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep121_shard0.safetensors'),
    @('model_pp0_ep122_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep122_shard0.safetensors'),
    @('model_pp0_ep123_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep123_shard0.safetensors'),
    @('model_pp0_ep124_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep124_shard0.safetensors'),
    @('model_pp0_ep125_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep125_shard0.safetensors'),
    @('model_pp0_ep126_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep126_shard0.safetensors'),
    @('model_pp0_ep127_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep127_shard0.safetensors'),
    @('model_pp0_ep12_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep12_shard0.safetensors'),
    @('model_pp0_ep13_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep13_shard0.safetensors'),
    @('model_pp0_ep14_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep14_shard0.safetensors'),
    @('model_pp0_ep15_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep15_shard0.safetensors'),
    @('model_pp0_ep16_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep16_shard0.safetensors'),
    @('model_pp0_ep17_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep17_shard0.safetensors'),
    @('model_pp0_ep18_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep18_shard0.safetensors'),
    @('model_pp0_ep19_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep19_shard0.safetensors'),
    @('model_pp0_ep1_shard0.safetensors', 4151335328, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep1_shard0.safetensors'),
    @('model_pp0_ep20_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep20_shard0.safetensors'),
    @('model_pp0_ep21_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep21_shard0.safetensors'),
    @('model_pp0_ep22_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep22_shard0.safetensors'),
    @('model_pp0_ep23_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep23_shard0.safetensors'),
    @('model_pp0_ep24_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep24_shard0.safetensors'),
    @('model_pp0_ep25_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep25_shard0.safetensors'),
    @('model_pp0_ep26_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep26_shard0.safetensors'),
    @('model_pp0_ep27_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep27_shard0.safetensors'),
    @('model_pp0_ep28_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep28_shard0.safetensors'),
    @('model_pp0_ep29_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep29_shard0.safetensors'),
    @('model_pp0_ep2_shard0.safetensors', 4151335328, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep2_shard0.safetensors'),
    @('model_pp0_ep30_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep30_shard0.safetensors'),
    @('model_pp0_ep31_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep31_shard0.safetensors'),
    @('model_pp0_ep32_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep32_shard0.safetensors'),
    @('model_pp0_ep33_shard0.safetensors', 4151337400, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep33_shard0.safetensors'),
    @('model_pp0_ep34_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep34_shard0.safetensors'),
    @('model_pp0_ep35_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep35_shard0.safetensors'),
    @('model_pp0_ep36_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep36_shard0.safetensors'),
    @('model_pp0_ep37_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep37_shard0.safetensors'),
    @('model_pp0_ep38_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep38_shard0.safetensors'),
    @('model_pp0_ep39_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep39_shard0.safetensors'),
    @('model_pp0_ep3_shard0.safetensors', 4151336160, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep3_shard0.safetensors'),
    @('model_pp0_ep40_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep40_shard0.safetensors'),
    @('model_pp0_ep41_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep41_shard0.safetensors'),
    @('model_pp0_ep42_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep42_shard0.safetensors'),
    @('model_pp0_ep43_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep43_shard0.safetensors'),
    @('model_pp0_ep44_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep44_shard0.safetensors'),
    @('model_pp0_ep45_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep45_shard0.safetensors'),
    @('model_pp0_ep46_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep46_shard0.safetensors'),
    @('model_pp0_ep47_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep47_shard0.safetensors'),
    @('model_pp0_ep48_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep48_shard0.safetensors'),
    @('model_pp0_ep49_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep49_shard0.safetensors'),
    @('model_pp0_ep4_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep4_shard0.safetensors'),
    @('model_pp0_ep50_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep50_shard0.safetensors'),
    @('model_pp0_ep51_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep51_shard0.safetensors'),
    @('model_pp0_ep52_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep52_shard0.safetensors'),
    @('model_pp0_ep53_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep53_shard0.safetensors'),
    @('model_pp0_ep54_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep54_shard0.safetensors'),
    @('model_pp0_ep55_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep55_shard0.safetensors'),
    @('model_pp0_ep56_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep56_shard0.safetensors'),
    @('model_pp0_ep57_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep57_shard0.safetensors'),
    @('model_pp0_ep58_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep58_shard0.safetensors'),
    @('model_pp0_ep59_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep59_shard0.safetensors'),
    @('model_pp0_ep5_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep5_shard0.safetensors'),
    @('model_pp0_ep60_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep60_shard0.safetensors'),
    @('model_pp0_ep61_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep61_shard0.safetensors'),
    @('model_pp0_ep62_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep62_shard0.safetensors'),
    @('model_pp0_ep63_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep63_shard0.safetensors'),
    @('model_pp0_ep64_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep64_shard0.safetensors'),
    @('model_pp0_ep65_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep65_shard0.safetensors'),
    @('model_pp0_ep66_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep66_shard0.safetensors'),
    @('model_pp0_ep67_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep67_shard0.safetensors'),
    @('model_pp0_ep68_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep68_shard0.safetensors'),
    @('model_pp0_ep69_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep69_shard0.safetensors'),
    @('model_pp0_ep6_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep6_shard0.safetensors'),
    @('model_pp0_ep70_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep70_shard0.safetensors'),
    @('model_pp0_ep71_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep71_shard0.safetensors'),
    @('model_pp0_ep72_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep72_shard0.safetensors'),
    @('model_pp0_ep73_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep73_shard0.safetensors'),
    @('model_pp0_ep74_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep74_shard0.safetensors'),
    @('model_pp0_ep75_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep75_shard0.safetensors'),
    @('model_pp0_ep76_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep76_shard0.safetensors'),
    @('model_pp0_ep77_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep77_shard0.safetensors'),
    @('model_pp0_ep78_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep78_shard0.safetensors'),
    @('model_pp0_ep79_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep79_shard0.safetensors'),
    @('model_pp0_ep7_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep7_shard0.safetensors'),
    @('model_pp0_ep80_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep80_shard0.safetensors'),
    @('model_pp0_ep81_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep81_shard0.safetensors'),
    @('model_pp0_ep82_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep82_shard0.safetensors'),
    @('model_pp0_ep83_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep83_shard0.safetensors'),
    @('model_pp0_ep84_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep84_shard0.safetensors'),
    @('model_pp0_ep85_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep85_shard0.safetensors'),
    @('model_pp0_ep86_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep86_shard0.safetensors'),
    @('model_pp0_ep87_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep87_shard0.safetensors'),
    @('model_pp0_ep88_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep88_shard0.safetensors'),
    @('model_pp0_ep89_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep89_shard0.safetensors'),
    @('model_pp0_ep8_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep8_shard0.safetensors'),
    @('model_pp0_ep90_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep90_shard0.safetensors'),
    @('model_pp0_ep91_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep91_shard0.safetensors'),
    @('model_pp0_ep92_shard0.safetensors', 4151337816, 'C:\AI\MiMoPro-Shards\model_pp0_ep92_shard0.safetensors'),
    @('model_pp0_ep93_shard0.safetensors', 4151337816, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep93_shard0.safetensors'),
    @('model_pp0_ep94_shard0.safetensors', 4151337816, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep94_shard0.safetensors'),
    @('model_pp0_ep95_shard0.safetensors', 4151337816, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep95_shard0.safetensors'),
    @('model_pp0_ep96_shard0.safetensors', 4151337816, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep96_shard0.safetensors'),
    @('model_pp0_ep97_shard0.safetensors', 4151337816, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep97_shard0.safetensors'),
    @('model_pp0_ep98_shard0.safetensors', 4151337816, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep98_shard0.safetensors'),
    @('model_pp0_ep99_shard0.safetensors', 4151337816, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep99_shard0.safetensors'),
    @('model_pp0_ep9_shard0.safetensors', 4151336576, 'D:\AI\Models\MiMo-V2.6-Pro-MOPD\model_pp0_ep9_shard0.safetensors')
)

if (Get-Process -Name mimo -ErrorAction SilentlyContinue) { throw 'colibri (mimo.exe) is running: stop it first.' }
if (!(Test-Path -LiteralPath (Join-Path $Heretic $Own) -PathType Leaf)) { throw "$Heretic has no $Own; nothing deleted." }

function Test-SameEnds([string]$a, [string]$b, [long]$size) {
    $n = [int][Math]::Min([long]8MB, $size)
    foreach ($at in @(0L, ($size - $n))) {
        $x = New-Object byte[] $n; $y = New-Object byte[] $n
        $fa = [IO.File]::OpenRead($a); $fb = [IO.File]::OpenRead($b)
        try {
            [void]$fa.Seek($at, 'Begin'); [void]$fb.Seek($at, 'Begin')
            $ra = 0; while ($ra -lt $n) { $k = $fa.Read($x, $ra, $n - $ra); if ($k -le 0) { break }; $ra += $k }
            $rb = 0; while ($rb -lt $n) { $k = $fb.Read($y, $rb, $n - $rb); if ($k -le 0) { break }; $rb += $k }
        } finally { $fa.Dispose(); $fb.Dispose() }
        if ($ra -ne $n -or $rb -ne $n -or -not [Linq.Enumerable]::SequenceEqual($x, $y)) { return $false }
    }
    return $true
}

$done = 0; $bytes = 0L; $kept = 0
foreach ($f in $Files) {
    $name, $size, $base = $f
    $path = Join-Path $Heretic $name
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { Write-Host "already gone: $name"; continue }
    $item = Get-Item -LiteralPath $path
    $why = $null
    if ($item.LinkType) { $why = 'it is a link, not a copy' }
    elseif (!(Test-Path -LiteralPath $base -PathType Leaf)) { $why = "its base copy $base is missing" }
    elseif ($item.Length -ne $size -or (Get-Item -LiteralPath $base).Length -ne $size) { $why = 'its size changed' }
    elseif (!(Test-SameEnds $path $base $size)) { $why = 'its bytes changed' }
    if ($why) { Write-Host "KEPT $name`: $why"; $kept++; continue }
    if ($Apply) { Remove-Item -LiteralPath $path -Force; Write-Host "deleted $name" }
    else { Write-Host "would delete $name (= $base)" }
    $done++; $bytes += $size
}
$verb = $(if ($Apply) { 'Deleted' } else { 'Would delete' })
Write-Host ("{0} {1} file(s), {2:N1} GB; kept {3}." -f $verb, $done, ($bytes / 1e9), $kept)
if (!$Apply) { Write-Host 'Nothing was deleted. Run again with -Apply to delete.' }
