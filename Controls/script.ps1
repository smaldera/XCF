#script.ps1
param([String]$folderpath, [String]$livetime, [String]$filename)
Write-Output $folderpath
Write-Output $livetime
Write-Output $filename
wsl /home/xcf/amptekhardwareinterface/launch.sh $folderpath $livetime $filename
