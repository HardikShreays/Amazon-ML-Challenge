# v3 submission: France suffix fix + partition + address-missing threshold, pass-1 scored with 600 trees.
# Runs in an isolated work folder (work_v3) so it cannot disturb a run in work/.
$root = 'D:\Amazon ML challenge'
Set-Location "$root\code\business_entity_resolution"
$env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'
$env:ER_WORK_DIR = "$root\work_v3"
$env:ER_OUTPUT_DIR = "$root\submissions\v3_francefix_600"
$py = "$root\.venv\Scripts\python.exe"
foreach ($stage in 's1_dictionaries', 's1_signatures', 's2_blocking', 's3_features') {
    $t = Get-Date
    Write-Output "[v3] $stage ..."
    & $py -c "from src import $stage; $stage.build('test')"
    if ($LASTEXITCODE -ne 0) { throw "$stage failed" }
    Write-Output ("[v3] $stage finished in {0:N0}s" -f ((Get-Date) - $t).TotalSeconds)
}
& $py -m src.fast_submit --trees 600
if ($LASTEXITCODE -ne 0) { throw 'fast_submit failed' }
Write-Output '[v3] done'
