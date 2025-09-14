#!/usr/bin/env pwsh

<#
.SYNOPSIS
Python environment setup for Mogadishu S-Entropy Framework demos

.DESCRIPTION
Sets up isolated Python environment with all required dependencies for
S-entropy demos, plotting utilities, and numerical validation.

.PARAMETER Reinstall
Remove existing environment and create fresh installation

.PARAMETER NoDemos
Skip demo-specific dependencies (minimal scientific Python)

.PARAMETER Development
Install additional development dependencies

.EXAMPLE
./scripts/setup-python.ps1 -Development

.EXAMPLE  
./scripts/setup-python.ps1 -Reinstall -NoDemos
#>

param(
    [switch]$Reinstall,
    [switch]$NoDemos,
    [switch]$Development
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot "venv"
$DemosPath = Join-Path $ProjectRoot "demos"
$RequirementsPath = Join-Path $DemosPath "requirements.txt"

$Green = "Green"
$Yellow = "Yellow"
$Red = "Red"
$Cyan = "Cyan"

function Write-SetupStatus {
    param([string]$Message, [string]$Color = "White")
    Write-Host "🐍 $Message" -ForegroundColor $Color
}

function Write-SetupSuccess {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor $Green
}

function Write-SetupWarning {
    param([string]$Message)
    Write-Host "⚠️ $Message" -ForegroundColor $Yellow
}

function Write-SetupError {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor $Red
}

function Test-PythonAvailability {
    Write-SetupStatus "Checking Python availability..." $Cyan
    
    # Check Python 3
    $pythonCommands = @("python3", "python", "py")
    $pythonCmd = $null
    
    foreach ($cmd in $pythonCommands) {
        try {
            $version = & $cmd --version 2>$null
            if ($version -match "Python 3\.([9-9]|[1-9][0-9])") {
                $pythonCmd = $cmd
                Write-SetupSuccess "Found $cmd : $version"
                break
            }
            elseif ($version -match "Python 3\.([0-8])") {
                Write-SetupWarning "Found $cmd : $version (minimum Python 3.9 recommended)"
                $pythonCmd = $cmd
            }
        }
        catch {
            continue
        }
    }
    
    if (-not $pythonCmd) {
        Write-SetupError "Python 3.9+ not found. Please install Python from https://python.org/"
        Write-SetupStatus "Required for S-entropy demos and numerical validation."
        exit 1
    }
    
    return $pythonCmd
}

function Remove-ExistingEnvironment {
    if (Test-Path $VenvPath) {
        Write-SetupStatus "Removing existing virtual environment..." $Yellow
        Remove-Item -Recurse -Force $VenvPath
        Write-SetupSuccess "Existing environment removed"
    }
}

function Create-VirtualEnvironment {
    param([string]$PythonCmd)
    
    Write-SetupStatus "Creating Python virtual environment..." $Cyan
    
    # Create virtual environment
    & $PythonCmd -m venv $VenvPath
    
    if ($LASTEXITCODE -ne 0) {
        Write-SetupError "Failed to create virtual environment"
        exit 1
    }
    
    Write-SetupSuccess "Virtual environment created at $VenvPath"
}

function Get-VenvPython {
    $venvPython = if ($IsWindows -or $env:OS -eq "Windows_NT") {
        Join-Path $VenvPath "Scripts\python.exe"
    } else {
        Join-Path $VenvPath "bin/python"
    }
    
    if (-not (Test-Path $venvPython)) {
        Write-SetupError "Virtual environment Python not found at $venvPython"
        exit 1
    }
    
    return $venvPython
}

function Install-CoreDependencies {
    param([string]$VenvPython)
    
    Write-SetupStatus "Installing core Python dependencies..." $Cyan
    
    # Upgrade pip first
    Write-SetupStatus "Upgrading pip..."
    & $VenvPython -m pip install --upgrade pip
    
    # Core scientific computing stack
    $coreDeps = @(
        "numpy>=1.24.0",
        "scipy>=1.10.0", 
        "pandas>=2.0.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
        "jupyter>=1.0.0",
        "ipython>=8.0.0"
    )
    
    Write-SetupStatus "Installing scientific computing stack..."
    foreach ($dep in $coreDeps) {
        Write-SetupStatus "Installing $dep..."
        & $VenvPython -m pip install $dep
        
        if ($LASTEXITCODE -ne 0) {
            Write-SetupWarning "Failed to install $dep"
        }
    }
    
    Write-SetupSuccess "Core dependencies installed"
}

function Install-SEntropyDependencies {
    param([string]$VenvPython)
    
    Write-SetupStatus "Installing S-entropy specific dependencies..." $Cyan
    
    # S-entropy framework specific packages
    $sEntropyDeps = @(
        "networkx>=3.0",           # For cellular network modeling
        "sympy>=1.12",             # For symbolic mathematics
        "numba>=0.58.0",           # For JIT compilation of hot paths
        "scikit-learn>=1.3.0",     # For machine learning components
        "statsmodels>=0.14.0",     # For statistical modeling
        "openpyxl>=3.1.0",         # For Excel output
        "h5py>=3.9.0",             # For HDF5 data storage
        "tqdm>=4.66.0",            # For progress bars
        "colorama>=0.4.6",         # For colored terminal output
        "click>=8.1.0"             # For CLI interfaces
    )
    
    foreach ($dep in $sEntropyDeps) {
        Write-SetupStatus "Installing $dep..."
        & $VenvPython -m pip install $dep
        
        if ($LASTEXITCODE -ne 0) {
            Write-SetupWarning "Failed to install $dep"
        }
    }
    
    Write-SetupSuccess "S-entropy dependencies installed"
}

function Install-DevelopmentDependencies {
    param([string]$VenvPython)
    
    if (-not $Development) {
        return
    }
    
    Write-SetupStatus "Installing development dependencies..." $Cyan
    
    $devDeps = @(
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0", 
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
        "sphinx>=7.1.0",
        "sphinx-rtd-theme>=1.3.0",
        "pre-commit>=3.4.0",
        "ipykernel>=6.25.0",
        "jupyter-lab>=4.0.0"
    )
    
    foreach ($dep in $devDeps) {
        Write-SetupStatus "Installing $dep..."
        & $VenvPython -m pip install $dep
    }
    
    Write-SetupSuccess "Development dependencies installed"
}

function Create-RequirementsFile {
    Write-SetupStatus "Creating requirements.txt file..." $Cyan
    
    # Ensure demos directory exists
    if (-not (Test-Path $DemosPath)) {
        New-Item -ItemType Directory -Path $DemosPath | Out-Null
    }
    
    $requirements = @"
# Core scientific computing
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.17.0

# S-entropy framework specific
networkx>=3.0
sympy>=1.12
numba>=0.58.0
scikit-learn>=1.3.0
statsmodels>=0.14.0

# Data handling
openpyxl>=3.1.0
h5py>=3.9.0
tqdm>=4.66.0

# Utilities
colorama>=0.4.6
click>=8.1.0

# Jupyter environment
jupyter>=1.0.0
ipython>=8.0.0
"@

    if ($Development) {
        $requirements += @"

# Development dependencies
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
sphinx>=7.1.0
sphinx-rtd-theme>=1.3.0
pre-commit>=3.4.0
ipykernel>=6.25.0
jupyter-lab>=4.0.0
"@
    }
    
    $requirements | Out-File -FilePath $RequirementsPath -Encoding UTF8
    Write-SetupSuccess "Requirements file created: $RequirementsPath"
}

function Create-DemoStructure {
    Write-SetupStatus "Creating demo directory structure..." $Cyan
    
    $demoDirectories = @(
        $DemosPath,
        (Join-Path $DemosPath "plotting"),
        (Join-Path $DemosPath "validation"), 
        (Join-Path $DemosPath "examples"),
        (Join-Path $DemosPath "data"),
        (Join-Path $DemosPath "results"),
        (Join-Path $DemosPath "notebooks")
    )
    
    foreach ($dir in $demoDirectories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir | Out-Null
            Write-SetupStatus "Created: $dir" $Green
        }
    }
    
    # Create placeholder demo files
    $demoFiles = @{
        "s_entropy_navigation.py" = @"
#!/usr/bin/env python3
"""
S-Entropy Navigation Demo

Demonstrates tri-dimensional S-space navigation and solution discovery
through observer-process integration.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def main():
    print("🌌 S-Entropy Navigation Demo")
    print("=" * 30)
    
    # TODO: Implement S-entropy navigation demonstration
    print("✅ Demo placeholder created")
    
    # Save placeholder results
    results = {"demo": "s_entropy_navigation", "status": "placeholder"}
    results_path = Path("results/s_entropy_navigation.json")
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
"@
        
        "cellular_processing.py" = @"
#!/usr/bin/env python3
"""
Cellular Processing Demo

Demonstrates ATP-constrained dynamics and 99%/1% membrane quantum computer
/ DNA consultation architecture.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def main():
    print("🧬 Cellular Processing Demo") 
    print("=" * 30)
    
    # TODO: Implement cellular processing demonstration
    print("✅ Demo placeholder created")
    
    results = {"demo": "cellular_processing", "status": "placeholder"}
    results_path = Path("results/cellular_processing.json") 
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
"@
        
        "miraculous_dynamics.py" = @"
#!/usr/bin/env python3
"""
Miraculous Dynamics Demo

Demonstrates tri-dimensional differential equations enabling local 
impossibilities while maintaining global S-viability.
"""

import numpy as np
import matplotlib.pyplot as plt  
import json
from pathlib import Path

def main():
    print("⚡ Miraculous Dynamics Demo")
    print("=" * 30)
    
    # TODO: Implement miraculous dynamics demonstration
    print("✅ Demo placeholder created")
    
    results = {"demo": "miraculous_dynamics", "status": "placeholder"}
    results_path = Path("results/miraculous_dynamics.json")
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
"@
    }
    
    foreach ($file in $demoFiles.GetEnumerator()) {
        $filePath = Join-Path $DemosPath $file.Key
        if (-not (Test-Path $filePath)) {
            $file.Value | Out-File -FilePath $filePath -Encoding UTF8
            Write-SetupStatus "Created: $($file.Key)" $Green
        }
    }
    
    Write-SetupSuccess "Demo structure created"
}

function Test-Installation {
    param([string]$VenvPython)
    
    Write-SetupStatus "Testing Python installation..." $Cyan
    
    $testScript = @"
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import json

print("Python version:", sys.version)
print("NumPy version:", np.__version__)
print("Matplotlib version:", plt.matplotlib.__version__)
print("SciPy version:", scipy.__version__)
print("Pandas version:", pd.__version__)

# Test basic functionality
test_data = np.random.randn(100)
test_df = pd.DataFrame({"data": test_data})
print("✅ All packages imported successfully")
print("✅ Basic functionality test passed")
"@
    
    $testPath = Join-Path $ProjectRoot "test_python_setup.py"
    $testScript | Out-File -FilePath $testPath -Encoding UTF8
    
    try {
        & $VenvPython $testPath
        
        if ($LASTEXITCODE -eq 0) {
            Write-SetupSuccess "Python installation test passed"
            Remove-Item $testPath -Force
            return $true
        } else {
            Write-SetupError "Python installation test failed"
            return $false
        }
    }
    catch {
        Write-SetupError "Failed to run Python test: $_"
        return $false
    }
}

function Show-SetupSummary {
    param([string]$VenvPython)
    
    Write-Host "`n🐍 Python Setup Summary" -ForegroundColor $Cyan
    Write-Host "=======================" -ForegroundColor $Cyan
    
    # Get Python version
    $pythonVersion = & $VenvPython --version
    Write-Host "Python: " -NoNewline
    Write-Host $pythonVersion -ForegroundColor $Green
    
    # Get pip version
    $pipVersion = & $VenvPython -m pip --version
    Write-Host "Pip: " -NoNewline
    Write-Host $pipVersion -ForegroundColor $Green
    
    # Show virtual environment path
    Write-Host "Virtual Environment: " -NoNewline
    Write-Host $VenvPath -ForegroundColor $Green
    
    # Show demo directory
    Write-Host "Demos Directory: " -NoNewline
    Write-Host $DemosPath -ForegroundColor $Green
    
    Write-Host "`nActivation commands:" -ForegroundColor $Yellow
    if ($IsWindows -or $env:OS -eq "Windows_NT") {
        Write-Host "PowerShell: " -NoNewline -ForegroundColor $Yellow
        Write-Host "$VenvPath\Scripts\Activate.ps1"
        Write-Host "CMD: " -NoNewline -ForegroundColor $Yellow 
        Write-Host "$VenvPath\Scripts\activate.bat"
    } else {
        Write-Host "Bash/Zsh: " -NoNewline -ForegroundColor $Yellow
        Write-Host "source $VenvPath/bin/activate"
    }
    
    Write-Host "`nNext steps:" -ForegroundColor $Yellow
    Write-Host "• Run demos: " -NoNewline -ForegroundColor $Yellow
    Write-Host "./scripts/run-demos.ps1"
    Write-Host "• Start Jupyter: " -NoNewline -ForegroundColor $Yellow
    Write-Host "$VenvPython -m jupyter lab"
    Write-Host "• Run tests: " -NoNewline -ForegroundColor $Yellow
    Write-Host "./scripts/test.ps1 -TestType python"
}

function Main {
    Write-Host "🐍 Mogadishu Python Environment Setup" -ForegroundColor $Cyan
    Write-Host "=====================================" -ForegroundColor $Cyan
    
    $pythonCmd = Test-PythonAvailability
    
    if ($Reinstall) {
        Remove-ExistingEnvironment
    }
    
    if (-not (Test-Path $VenvPath)) {
        Create-VirtualEnvironment $pythonCmd
    } else {
        Write-SetupStatus "Virtual environment already exists at $VenvPath" $Yellow
    }
    
    $venvPython = Get-VenvPython
    
    Install-CoreDependencies $venvPython
    
    if (-not $NoDemos) {
        Install-SEntropyDependencies $venvPython
    }
    
    Install-DevelopmentDependencies $venvPython
    
    Create-RequirementsFile
    Create-DemoStructure
    
    $testPassed = Test-Installation $venvPython
    
    if ($testPassed) {
        Show-SetupSummary $venvPython
        Write-SetupSuccess "`n🎉 Python environment setup completed successfully!"
    } else {
        Write-SetupError "`n💥 Python environment setup failed!"
        exit 1
    }
}

Main
