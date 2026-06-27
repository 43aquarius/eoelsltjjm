$src = 'D:\file\project\LocalContract\local-contract\out'
$dest = 'C:\Users\23529\Desktop\local-contract-cloudflare'

if (Test-Path $dest) {
    Remove-Item $dest -Recurse -Force
}

Copy-Item $src $dest -Recurse -Force

$files = Get-ChildItem $dest -Recurse -File
$totalSize = ($files | Measure-Object -Property Length -Sum).Sum

Write-Host "Copied $($files.Count) files"
$sizeMB = [math]::Round($totalSize / 1MB, 2)
Write-Host "Total size: $sizeMB MB"
Write-Host "Destination: $dest"
Write-Host "Done"
