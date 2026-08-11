$g = "C:\Users\nakul\google-cloud-sdk\bin\gcloud.cmd"
$z = "--zone=asia-south1-b"
$files = Get-ChildItem "G:\Datacurve\Latest_Chals\work-ch5-split-verdict\code\*.py" | ForEach-Object { $_.FullName }
& $g compute scp $files ml-main:/home/nakul/sv/code/ $z 2>&1 | Select-Object -Last 1
Write-Output "PUSHED"
