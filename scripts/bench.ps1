param(
    [string]$ModelDir = $env:SYS1_BENCH_MODEL,
    [string]$ModelName = "laya-multilingual",
    [string]$Exe = "",
    [string]$Device = "auto",
    [string]$Dtype = "f32",
    [string]$Confidence = "laya",
    [int]$Port = 18082,
    [int]$Warmup = 3,
    [int]$Sequential = 20,
    [int]$BatchSequential = 10,
    [int]$OptionRequests = 8,
    [int]$BatchMax = 32,
    [int]$BatchWaitMs = 2,
    [switch]$NoBatch,
    [switch]$Tf32,
    [switch]$UniqueStates,
    [switch]$ConcurrencySweep,
    [switch]$OptionSweep
)

$ErrorActionPreference = "Stop"
if (-not $ModelDir) {
    throw "Set SYS1_BENCH_MODEL or pass -ModelDir <laya checkpoint directory>"
}
$ModelDir = (Resolve-Path $ModelDir).Path
$root = Split-Path -Parent $PSScriptRoot
$exe = if ($Exe) {
    (Resolve-Path $Exe).Path
} else {
    Join-Path $root "target\release\sys1.exe"
}
if (-not (Test-Path $exe)) {
    throw "Build first: cargo build --release --offline -p sys1-cli"
}

$logOut = Join-Path $env:TEMP "sys1-bench-$Port.out.log"
$logErr = Join-Path $env:TEMP "sys1-bench-$Port.err.log"
$base = "http://127.0.0.1:$Port"
$stateText = "I was charged twice. Please refund the extra charge today."
$server = $null

function New-JsonContent([string]$json) {
    return [System.Net.Http.StringContent]::new(
        $json,
        [System.Text.Encoding]::UTF8,
        "application/json"
    )
}

function Stats([double[]]$values) {
    $sorted = $values | Sort-Object
    $n = $sorted.Count
    $mean = ($sorted | Measure-Object -Average).Average
    return [pscustomobject]@{
        Count = $n
        Mean = [math]::Round($mean, 2)
        P50 = [math]::Round($sorted[[math]::Max(0, [int][math]::Floor($n * 0.50) - 1)], 2)
        P95 = [math]::Round($sorted[[math]::Max(0, [int][math]::Ceiling($n * 0.95) - 1)], 2)
        Max = [math]::Round($sorted[$n - 1], 2)
    }
}

function New-Client {
    $client = [System.Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromMinutes(5)
    return $client
}

function Wait-Ready {
    $client = New-Client
    try {
        for ($i = 0; $i -lt 240; $i++) {
            try {
                $response = $client.GetAsync("$base/health").GetAwaiter().GetResult()
                if ($response.IsSuccessStatusCode) { return }
            } catch { }
            Start-Sleep -Milliseconds 500
        }
        throw "server did not become ready"
    } finally {
        $client.Dispose()
    }
}

function Measure-SequentialPosts([string]$label, [string]$body, [int]$count) {
    $client = New-Client
    try {
        for ($i = 0; $i -lt $Warmup; $i++) {
            $response = $client.PostAsync("$base/v1/systemone", (New-JsonContent $body)).GetAwaiter().GetResult()
            $response.EnsureSuccessStatusCode() | Out-Null
        }
        $times = [System.Collections.Generic.List[double]]::new()
        $wall = [System.Diagnostics.Stopwatch]::StartNew()
        for ($i = 0; $i -lt $count; $i++) {
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $response = $client.PostAsync("$base/v1/systemone", (New-JsonContent $body)).GetAwaiter().GetResult()
            $response.EnsureSuccessStatusCode() | Out-Null
            $null = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            $sw.Stop()
            $times.Add($sw.Elapsed.TotalMilliseconds)
        }
        $wall.Stop()
        $stats = Stats $times.ToArray()
        [pscustomobject]@{
            Scenario = $label
            Requests = $count
            Concurrency = 1
            MeanMs = $stats.Mean
            P50Ms = $stats.P50
            P95Ms = $stats.P95
            MaxMs = $stats.Max
            Rps = [math]::Round($count / $wall.Elapsed.TotalSeconds, 2)
        }
    } finally {
        $client.Dispose()
    }
}

function Measure-SequentialGets([string]$label, [string]$path, [int]$count) {
    $client = New-Client
    try {
        for ($i = 0; $i -lt $Warmup; $i++) {
            $response = $client.GetAsync("$base$path").GetAwaiter().GetResult()
            $response.EnsureSuccessStatusCode() | Out-Null
        }
        $times = [System.Collections.Generic.List[double]]::new()
        $wall = [System.Diagnostics.Stopwatch]::StartNew()
        for ($i = 0; $i -lt $count; $i++) {
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $response = $client.GetAsync("$base$path").GetAwaiter().GetResult()
            $response.EnsureSuccessStatusCode() | Out-Null
            $null = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            $sw.Stop()
            $times.Add($sw.Elapsed.TotalMilliseconds)
        }
        $wall.Stop()
        $stats = Stats $times.ToArray()
        [pscustomobject]@{
            Scenario = $label
            Requests = $count
            Concurrency = 1
            MeanMs = $stats.Mean
            P50Ms = $stats.P50
            P95Ms = $stats.P95
            MaxMs = $stats.Max
            Rps = [math]::Round($count / $wall.Elapsed.TotalSeconds, 2)
        }
    } finally {
        $client.Dispose()
    }
}

function Measure-Concurrent(
    [string]$label,
    [string]$body,
    [int]$workers,
    [int]$perWorker,
    [bool]$uniqueStates = $false
) {
    $latencies = [System.Collections.Concurrent.ConcurrentBag[double]]::new()
    $wall = [System.Diagnostics.Stopwatch]::StartNew()
    1..$workers | ForEach-Object -Parallel {
        $bag = $using:latencies
        $bodyText = $using:body
        $unique = $using:uniqueStates
        $worker = $_
        $url = "$using:base/v1/systemone"
        $client = [System.Net.Http.HttpClient]::new()
        $client.Timeout = [TimeSpan]::FromMinutes(5)
        try {
            for ($i = 0; $i -lt $using:perWorker; $i++) {
                $requestBody = if ($unique) {
                    $bodyText.Replace(
                        "I was charged twice. Please refund the extra charge today.",
                        "I was charged twice. Please refund the extra charge today. Case $worker-$i."
                    )
                } else {
                    $bodyText
                }
                $content = [System.Net.Http.StringContent]::new(
                    $requestBody,
                    [System.Text.Encoding]::UTF8,
                    "application/json"
                )
                $sw = [System.Diagnostics.Stopwatch]::StartNew()
                $response = $client.PostAsync($url, $content).GetAwaiter().GetResult()
                $response.EnsureSuccessStatusCode() | Out-Null
                $null = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
                $sw.Stop()
                $bag.Add($sw.Elapsed.TotalMilliseconds)
            }
        } finally {
            $client.Dispose()
        }
    } -ThrottleLimit $workers
    $wall.Stop()
    $stats = Stats $latencies.ToArray()
    $total = $workers * $perWorker
    [pscustomobject]@{
        Scenario = $label
        Requests = $total
        Concurrency = $workers
        MeanMs = $stats.Mean
        P50Ms = $stats.P50
        P95Ms = $stats.P95
        MaxMs = $stats.Max
        Rps = [math]::Round($total / $wall.Elapsed.TotalSeconds, 2)
    }
}

function Measure-OptionScenario([string]$kind, [int]$options, [int]$requests) {
    $client = New-Client
    try {
        if ($kind -eq "choice") {
            $criteria = [ordered]@{}
            for ($i = 0; $i -lt $options; $i++) {
                $criteria["c$i"] = "option $i"
            }
            $body = @{
                model = $ModelName
                state = $stateText
                questions = @{
                    pick = @{
                        type = "choice"
                        instructions = "Pick the best option."
                        criteria = $criteria
                    }
                }
            } | ConvertTo-Json -Depth 10 -Compress
            $answerKey = "pick"
        } else {
            $levels = @()
            for ($i = 0; $i -lt $options; $i++) {
                $levels += "level $i"
            }
            $body = @{
                model = $ModelName
                state = $stateText
                questions = @{
                    rating = @{
                        type = "score"
                        instructions = "Rate the content."
                        criteria = $levels
                    }
                }
            } | ConvertTo-Json -Depth 10 -Compress
            $answerKey = "rating"
        }

        # Warm-up request.
        $response = $client.PostAsync("$base/v1/systemone", (New-JsonContent $body)).GetAwaiter().GetResult()
        $response.EnsureSuccessStatusCode() | Out-Null
        $null = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()

        $times = [System.Collections.Generic.List[double]]::new()
        $sums = [System.Collections.Generic.List[double]]::new()
        $valid = $true
        $wall = [System.Diagnostics.Stopwatch]::StartNew()
        for ($i = 0; $i -lt $requests; $i++) {
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $response = $client.PostAsync("$base/v1/systemone", (New-JsonContent $body)).GetAwaiter().GetResult()
            $response.EnsureSuccessStatusCode() | Out-Null
            $json = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json
            $sw.Stop()
            $times.Add($sw.Elapsed.TotalMilliseconds)

            $answer = $json.answers.$answerKey
            $probabilities = @($answer.probabilities.PSObject.Properties)
            if ($probabilities.Count -ne $options) {
                $valid = $false
            }
            $sum = ($probabilities | ForEach-Object { $_.Value } | Measure-Object -Sum).Sum
            $sums.Add($sum)
            if ($kind -eq "score" -and @($answer.legend.PSObject.Properties).Count -ne $options) {
                $valid = $false
            }
        }
        $wall.Stop()

        $stats = Stats $times.ToArray()
        [pscustomobject]@{
            Question = $kind
            Options = $options
            Requests = $requests
            MeanMs = $stats.Mean
            P50Ms = $stats.P50
            P95Ms = $stats.P95
            MaxMs = $stats.Max
            Rps = [math]::Round($requests / $wall.Elapsed.TotalSeconds, 2)
            ProbSum = [math]::Round((($sums | Measure-Object -Average).Average), 4)
            Valid = $valid
        }
    } finally {
        $client.Dispose()
    }
}

try {
    $serveArgs = @(
        "serve",
        "--bind", "127.0.0.1:$Port",
        "--model", "$ModelName=local:$ModelDir",
        "--device", $Device,
        "--dtype", $Dtype,
        "--laya-confidence", $Confidence,
        "--batch-max", "$BatchMax",
        "--batch-wait-ms", "$BatchWaitMs"
    )
    if ($NoBatch) {
        $serveArgs += "--no-batch"
    }
    if ($Tf32) {
        $serveArgs += "--tf32"
    }
    $server = Start-Process -FilePath $exe -ArgumentList $serveArgs -RedirectStandardOutput $logOut -RedirectStandardError $logErr -PassThru -NoNewWindow

    Wait-Ready

    $noul = @{
        model = $ModelName
        state = $stateText
        questions = @{
            billing = @{
                type = "noul"
                instructions = "Is this message about billing?"
            }
        }
    } | ConvertTo-Json -Depth 10 -Compress

    $batch3 = @{
        model = $ModelName
        state = "I was charged twice. Please refund the extra charge today."
        questions = @{
            billing = @{
                type = "noul"
                instructions = "Is this message about billing?"
            }
            urgency = @{
                type = "score"
                instructions = "How urgent is this message?"
                criteria = @("Can wait", "This week", "Today")
            }
            tone = @{
                type = "choice"
                instructions = "What is the tone of this message?"
                criteria = @{
                    angry = "An upset or hostile message"
                    calm = "A neutral or polite message"
                }
            }
        }
    } | ConvertTo-Json -Depth 10 -Compress

    # The first request loads the checkpoint.
    $warm = New-Client
    try {
        $response = $warm.PostAsync("$base/v1/systemone", (New-JsonContent $noul)).GetAwaiter().GetResult()
        $response.EnsureSuccessStatusCode() | Out-Null
        $null = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    } finally {
        $warm.Dispose()
    }

    $results = @()
    $results += Measure-SequentialGets "GET /health" "/health" 50
    $results += Measure-SequentialGets "GET /v1/models" "/v1/models" 50
    $results += Measure-SequentialPosts "POST noul" $noul $Sequential
    $results += Measure-SequentialPosts "POST noul+score+choice" $batch3 $BatchSequential
    $results += Measure-Concurrent "POST noul x4" $noul 4 5 -uniqueStates $UniqueStates
    $results += Measure-Concurrent "POST noul x8" $noul 8 5 -uniqueStates $UniqueStates

    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1 Name, NumberOfCores, NumberOfLogicalProcessors
    $ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)

    Write-Output "## sys1 benchmark"
    Write-Output ""
    Write-Output "- CPU: $($cpu.Name) ($($cpu.NumberOfCores) cores / $($cpu.NumberOfLogicalProcessors) threads)"
    Write-Output "- RAM: $ram GiB"
    Write-Output "- Model: $ModelName ($Dtype, device=$Device)"
    Write-Output "- Model dir: $ModelDir"
    Write-Output ""
    $results | Format-Table -AutoSize

    if ($ConcurrencySweep) {
        Write-Output ""
        Write-Output "## Concurrency sweep (noul)"
        $sweep = @()
        foreach ($workers in 1, 2, 4, 8, 16, 32) {
            $sweep += Measure-Concurrent "noul x$workers" $noul $workers 4 -uniqueStates $UniqueStates
        }
        $sweep | Format-Table -AutoSize
    }

    if ($OptionSweep) {
        Write-Output ""
        Write-Output "## Option sweep"
        $sweep = @()
        foreach ($n in 1, 2, 3, 5, 10, 20, 40) {
            $sweep += Measure-OptionScenario "choice" $n $OptionRequests
        }
        foreach ($n in 2, 3, 5, 10) {
            $sweep += Measure-OptionScenario "score" $n $OptionRequests
        }
        $sweep | Format-Table -AutoSize
    }
} finally {
    if ($server -and -not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
    }
}
