param($Work)

if(!$Work) {
    powershell.exe -NoExit ./win_setup.ps1 1
}

pip install virtualenv

if( -not (Test-Path .\virtual\)) {
    virtualenv.exe .\virtual\
}

.\virtual\Scripts\activate.ps1

pip install -r .\requirements.txt