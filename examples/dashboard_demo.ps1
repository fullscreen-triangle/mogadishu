# Mogadishu Dashboard Demonstration Script
# Launches the S-Entropy Process Dashboard with example parameters

Write-Host "🌐 Mogadishu S-Entropy Process Dashboard Demo" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""

# Check if Mogadishu CLI is built
if (-not (Test-Path "target/release/mogadishu-cli.exe") -and -not (Test-Path "target/debug/mogadishu-cli.exe")) {
    Write-Host "❌ Mogadishu CLI not found. Building..." -ForegroundColor Red
    Write-Host "Running: cargo build --release"
    cargo build --release
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Build failed. Exiting." -ForegroundColor Red
        exit 1
    }
}

# Determine CLI path
$cliPath = if (Test-Path "target/release/mogadishu-cli.exe") {
    "target/release/mogadishu-cli.exe"
} else {
    "target/debug/mogadishu-cli.exe"
}

Write-Host "✅ Using CLI: $cliPath" -ForegroundColor Green
Write-Host ""

# Dashboard configuration
$port = 3000
$updateInterval = 1000
$autoRefresh = $true

Write-Host "Dashboard Configuration:" -ForegroundColor Cyan
Write-Host "  Port: $port"
Write-Host "  Update interval: ${updateInterval}ms"  
Write-Host "  Auto-refresh: $autoRefresh"
Write-Host ""

Write-Host "🚀 Starting dashboard server..." -ForegroundColor Yellow
Write-Host "📊 Real-time Mermaid process diagrams will be available at:"
Write-Host "   http://localhost:$port" -ForegroundColor Green
Write-Host ""
Write-Host "📈 Available diagram types:"
Write-Host "  • System Overview - Complete S-entropy system state"
Write-Host "  • Cellular Network - ATP-constrained observer network"  
Write-Host "  • Miraculous Status - Tri-dimensional miracle dynamics"
Write-Host "  • Pipeline Flow - Processing pipeline status"
Write-Host ""

Write-Host "Press Ctrl+C to stop the dashboard server" -ForegroundColor Yellow
Write-Host ""

# Start dashboard with parameters
$dashboardArgs = @(
    "dashboard",
    "--port", $port,
    "--update-interval", $updateInterval,
    "--auto-refresh"
)

# Execute dashboard command
try {
    & $cliPath $dashboardArgs
}
catch {
    Write-Host "❌ Dashboard failed to start: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ Dashboard session completed." -ForegroundColor Green
