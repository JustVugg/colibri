param([string]$Out, [string]$StopFile, [int]$Seconds = 2)
# Every $Seconds: read MB/s of C: and D:, their busy %, CPU %, GPU %, GPU memory, kimi_k3's
# working set and CPU seconds, into a CSV (invariant culture), until $StopFile exists.
# WMI's formatted performance classes: their names are not localized (Get-Counter's are).
$inv = [System.Globalization.CultureInfo]::InvariantCulture
"time;c_mbs;d_mbs;c_busy;d_busy;cpu;gpu;gpu_mem_mb;ws_gb;proc_cpu_s" | Out-File -FilePath $Out -Encoding ascii
while (-not (Test-Path $StopFile)) {
    Start-Sleep -Seconds $Seconds
    try {
        $disks = Get-CimInstance -ClassName Win32_PerfFormattedData_PerfDisk_LogicalDisk -ErrorAction Stop
        $cd = $disks | Where-Object { $_.Name -eq 'C:' }
        $dd = $disks | Where-Object { $_.Name -eq 'D:' }
        $cpu = (Get-CimInstance -ClassName Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'").PercentProcessorTime
        $g = (& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits) -split ','
        $p = Get-Process kimi_k3 -ErrorAction SilentlyContinue | Select-Object -First 1
        $ws = if ($p) { $p.WorkingSet64 / 1GB } else { 0 }
        $pc = if ($p) { $p.CPU } else { 0 }
        [string]::Format($inv, "{0};{1:F0};{2:F0};{3:F0};{4:F0};{5:F0};{6};{7};{8:F1};{9:F0}", (Get-Date -Format 'HH:mm:ss'),
            $cd.DiskReadBytesPersec / 1MB, $dd.DiskReadBytesPersec / 1MB, 100 - $cd.PercentIdleTime,
            100 - $dd.PercentIdleTime, $cpu, $g[0].Trim(), $g[1].Trim(), $ws, $pc) | Out-File -FilePath $Out -Append -Encoding ascii
    } catch { }
}
