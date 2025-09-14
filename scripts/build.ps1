#!/usr/bin/env pwsh

<#
.SYNOPSIS
Build script for Mogadishu S-Entropy Framework

.DESCRIPTION
Comprehensive build system for the Rust framework and Python demo environment.
Supports development, release, and feature-specific builds.

.PARAMETER Configuration  
Build configuration: dev, release, or test (default: dev)

.PARAMETER AllFeatures
Enable all optional features during build

.PARAMETER Features
Comma-separated list of specific features to enable

.PARAMETER Clean
Clean build artifacts before building

.PARAMETER Verbose
Enable verbose build output

.EXAMPLE
./scripts/build.ps1 -Configuration release -AllFeatures

.EXAMPLE  
./scripts/build.ps1 -Features "oxygen-enhanced,quantum-transport" -Clean
#>

param(
    [ValidateSet("dev", "release", "test")]
    [string]$Configuration = "dev",
    
    [switch]$AllFeatures,
    
    [string]$Features,
    
    [switch]$Clean,
    
    [switch]$Verbose
)

# Script configuration
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Build paths
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RustSrcPath = Join-Path $ProjectRoot "src"
$PythonDemosPath = Join-Path $ProjectRoot "demos"
$BuildOutputPath = Join-Path $ProjectRoot "target"

# Colors for output
$Green = "Green"
$Yellow = "Yellow" 
$Red = "Red"
$Cyan = "Cyan"

function Write-Status {
    param([string]$Message, [string]$Color = "White")
    Write-Host "🔧 $Message" -ForegroundColor $Color
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor $Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️ $Message" -ForegroundColor $Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor $Red
}

function Test-Prerequisites {
    Write-Status "Checking prerequisites..."
    
    # Check Rust installation
    try {
        $rustVersion = rustc --version
        Write-Success "Rust found: $rustVersion"
    }
    catch {
        Write-Error "Rust not found. Please install Rust from https://rustup.rs/"
        exit 1
    }
    
    # Check Cargo
    try {
        $cargoVersion = cargo --version
        Write-Success "Cargo found: $cargoVersion"
    }
    catch {
        Write-Error "Cargo not found. Please install Rust toolchain."
        exit 1
    }
    
    # Check Python (for demo validation)
    try {
        $pythonVersion = python --version
        Write-Success "Python found: $pythonVersion"
    }
    catch {
        Write-Warning "Python not found. Python demos will not be available."
    }
    
    # Check Git
    try {
        $gitVersion = git --version
        Write-Success "Git found: $gitVersion"
    }
    catch {
        Write-Warning "Git not found. Version information may be incomplete."
    }
}

function Clean-BuildArtifacts {
    Write-Status "Cleaning build artifacts..." $Cyan
    
    if (Test-Path $BuildOutputPath) {
        Remove-Item -Recurse -Force $BuildOutputPath
        Write-Success "Cleaned Rust build artifacts"
    }
    
    # Clean Python cache
    $pythonCachePaths = @(
        (Join-Path $PythonDemosPath "__pycache__"),
        (Join-Path $PythonDemosPath "*.pyc"),
        (Join-Path $ProjectRoot "*.egg-info")
    )
    
    foreach ($path in $pythonCachePaths) {
        if (Test-Path $path) {
            Remove-Item -Recurse -Force $path -ErrorAction SilentlyContinue
        }
    }
    
    Write-Success "Cleaned Python cache files"
}

function Build-RustFramework {
    Write-Status "Building Mogadishu Rust framework..." $Cyan
    
    Push-Location $ProjectRoot
    
    try {
        # Determine build flags
        $buildFlags = @()
        
        if ($Configuration -eq "release") {
            $buildFlags += "--release"
            Write-Status "Building in release mode"
        }
        elseif ($Configuration -eq "test") {
            $buildFlags += "--tests"
            Write-Status "Building with tests"
        }
        else {
            Write-Status "Building in development mode"
        }
        
        # Handle features
        if ($AllFeatures) {
            $buildFlags += "--all-features"
            Write-Status "Enabling all features"
        }
        elseif ($Features) {
            $buildFlags += "--features", $Features
            Write-Status "Enabling features: $Features"
        }
        else {
            # Default features for S-entropy framework
            $defaultFeatures = "oxygen-enhanced,quantum-transport,atp-constraints"
            $buildFlags += "--features", $defaultFeatures
            Write-Status "Using default features: $defaultFeatures"
        }
        
        # Verbose output
        if ($Verbose) {
            $buildFlags += "--verbose"
        }
        
        # Execute build
        Write-Status "Running: cargo build $($buildFlags -join ' ')"
        & cargo build @buildFlags
        
        if ($LASTEXITCODE -ne 0) {
            throw "Rust build failed with exit code $LASTEXITCODE"
        }
        
        Write-Success "Rust framework build completed successfully"
        
        # Build Python bindings if feature enabled
        if ($AllFeatures -or $Features -match "python-bindings") {
            Write-Status "Building Python bindings..."
            & cargo build --features python-bindings @buildFlags
            
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Python bindings build failed"
            }
            else {
                Write-Success "Python bindings built successfully"
            }
        }
    }
    finally {
        Pop-Location
    }
}

function Setup-PythonEnvironment {
    Write-Status "Setting up Python demo environment..." $Cyan
    
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Warning "Python not found. Skipping demo environment setup."
        return
    }
    
    Push-Location $ProjectRoot
    
    try {
        # Create virtual environment
        $venvPath = Join-Path $ProjectRoot "venv"
        if (-not (Test-Path $venvPath)) {
            Write-Status "Creating Python virtual environment..."
            python -m venv venv
        }
        
        # Activate virtual environment
        $activateScript = if ($IsWindows -or $env:OS -eq "Windows_NT") {
            Join-Path $venvPath "Scripts\Activate.ps1"
        } else {
            Join-Path $venvPath "bin/Activate.ps1"
        }
        
        if (Test-Path $activateScript) {
            & $activateScript
            Write-Success "Python virtual environment activated"
        }
        
        # Install demo dependencies
        $requirementsPath = Join-Path $PythonDemosPath "requirements.txt"
        if (Test-Path $requirementsPath) {
            Write-Status "Installing Python demo dependencies..."
            python -m pip install --upgrade pip
            python -m pip install -r $requirementsPath
            Write-Success "Python dependencies installed"
        }
        
    }
    finally {
        Pop-Location
    }
}

function Validate-Build {
    Write-Status "Validating build..." $Cyan
    
    # Check Rust build outputs
    $rustTargetPath = Join-Path $ProjectRoot "target"
    $buildPath = if ($Configuration -eq "release") {
        Join-Path $rustTargetPath "release"
    } else {
        Join-Path $rustTargetPath "debug" 
    }
    
    # Check for library
    $libPattern = if ($IsWindows -or $env:OS -eq "Windows_NT") {
        "mogadishu.dll"
    } else {
        "libmogadishu.so"
    }
    
    $libPath = Join-Path $buildPath $libPattern
    if (Test-Path $libPath) {
        $libSize = (Get-Item $libPath).Length / 1MB
        Write-Success "Library built: $libPattern ($($libSize.ToString('F2')) MB)"
    }
    
    # Check for CLI binary
    $binaryName = if ($IsWindows -or $env:OS -eq "Windows_NT") {
        "mogadishu-cli.exe"
    } else {
        "mogadishu-cli"
    }
    
    $binaryPath = Join-Path $buildPath $binaryName
    if (Test-Path $binaryPath) {
        Write-Success "CLI binary built: $binaryName"
        
        # Test CLI
        try {
            $versionOutput = & $binaryPath --version
            Write-Success "CLI test passed: $versionOutput"
        }
        catch {
            Write-Warning "CLI test failed, but binary exists"
        }
    }
    
    # Validate Python setup
    if (Get-Command python -ErrorAction SilentlyContinue) {
        try {
            $pythonTest = python -c "import numpy, matplotlib, scipy, json; print('Python environment OK')"
            Write-Success "Python demo environment validated"
        }
        catch {
            Write-Warning "Python demo environment validation failed"
        }
    }
}

function Show-BuildSummary {
    Write-Host "`n" -NoNewline
    Write-Host "🎯 Build Summary" -ForegroundColor $Cyan
    Write-Host "================" -ForegroundColor $Cyan
    
    Write-Host "Configuration: " -NoNewline
    Write-Host $Configuration -ForegroundColor $Green
    
    Write-Host "Features: " -NoNewline
    if ($AllFeatures) {
        Write-Host "All enabled" -ForegroundColor $Green
    }
    elseif ($Features) {
        Write-Host $Features -ForegroundColor $Green
    }
    else {
        Write-Host "Default (oxygen-enhanced, quantum-transport, atp-constraints)" -ForegroundColor $Green
    }
    
    # Show build artifacts
    $artifactPath = if ($Configuration -eq "release") {
        Join-Path $ProjectRoot "target/release"
    } else {
        Join-Path $ProjectRoot "target/debug"
    }
    
    if (Test-Path $artifactPath) {
        $totalSize = (Get-ChildItem -Recurse $artifactPath | Measure-Object -Property Length -Sum).Sum / 1MB
        Write-Host "Build size: " -NoNewline
        Write-Host "$($totalSize.ToString('F2')) MB" -ForegroundColor $Green
    }
    
    Write-Host "`nNext steps:" -ForegroundColor $Yellow
    Write-Host "• Run tests: " -NoNewline -ForegroundColor $Yellow
    Write-Host "./scripts/test.ps1"
    Write-Host "• Start demos: " -NoNewline -ForegroundColor $Yellow  
    Write-Host "./scripts/run-demos.ps1"
    Write-Host "• Benchmarks: " -NoNewline -ForegroundColor $Yellow
    Write-Host "./scripts/benchmark.ps1"
}

# Main execution
function Main {
    Write-Host "🚀 Mogadishu S-Entropy Framework Build System" -ForegroundColor $Cyan
    Write-Host "=============================================" -ForegroundColor $Cyan
    
    Test-Prerequisites
    
    if ($Clean) {
        Clean-BuildArtifacts
    }
    
    Build-RustFramework
    Setup-PythonEnvironment  
    Validate-Build
    Show-BuildSummary
    
    Write-Success "`n🎉 Build completed successfully!"
}

# Execute main function
Main
