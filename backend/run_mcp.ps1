# MCP Server Startup Script for Z-Image-Turbo
# This script starts the MCP server with configurable transport modes

# Parse parameters - MUST be first in PowerShell scripts
param(
    [switch]$Stdio,
    [switch]$Http,
    [string]$HostAddress,
    [int]$Port = 0,
    [switch]$Help
)

# Stop on errors
$ErrorActionPreference = "Stop"

# Colors for output
$Colors = @{
    Red = 'Red'
    Green = 'Green'
    Yellow = 'Yellow'
    Blue = 'Cyan'
}

# Function to print colored output
function Print-Info {
    param([string]$Message)
    Write-Host "[INFO] " -ForegroundColor $Colors.Blue -NoNewline
    Write-Host $Message
}

function Print-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] " -ForegroundColor $Colors.Green -NoNewline
    Write-Host $Message
}

function Print-Warning {
    param([string]$Message)
    Write-Host "[WARNING] " -ForegroundColor $Colors.Yellow -NoNewline
    Write-Host $Message
}

function Print-Error {
    param([string]$Message)
    Write-Host "[ERROR] " -ForegroundColor $Colors.Red -NoNewline
    Write-Host $Message
}

# Help message
function Show-Help {
    @"
Z-Image-Turbo MCP Server Launcher

Usage: .\run_mcp.ps1 [OPTIONS]

Options:
    -Stdio              Run with stdio transport (for local clients like Claude Desktop)
    -Http               Run with HTTP/SSE transport (for web clients)
    -HostAddress <HOST> Set the host address for HTTP mode (default: 0.0.0.0)
    -Port <PORT>        Set the port for HTTP mode (default: 8001)
    -Help               Show this help message

Examples:
    # Run with stdio (default, for Claude Desktop)
    .\run_mcp.ps1 -Stdio

    # Run with HTTP on default port 8001
    .\run_mcp.ps1 -Http

    # Run with HTTP on custom port
    .\run_mcp.ps1 -Http -Port 8080

    # Run with HTTP on specific host
    .\run_mcp.ps1 -Http -HostAddress 127.0.0.1 -Port 9000

"@
}

# Show help if requested
if ($Help) {
    Show-Help
    exit 0
}

# Change to backend directory (where this script is located)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Print-Info "Starting Z-Image-Turbo MCP Server..."

# Build command arguments
$CmdArgs = @()

# Determine transport mode
if ($Stdio) {
    $CmdArgs += "--transport", "stdio"
    $TransportMode = "stdio"
}
elseif ($Http) {
    $CmdArgs += "--transport", "streamable-http"
    $TransportMode = "streamable-http"
}
else {
    $TransportMode = "(from mcp_config.json)"
}

# Add host if specified
if ($HostAddress -and $HostAddress -ne "") {
    $CmdArgs += "--host", $HostAddress
    $HostValue = $HostAddress
}
else {
    $HostValue = "(from mcp_config.json, default: 0.0.0.0)"
}

# Add port if specified
if ($Port -gt 0) {
    $CmdArgs += "--port", $Port.ToString()
    $PortValue = $Port.ToString()
}
else {
    $PortValue = "(from mcp_config.json, default: 8001)"
}

# Print configuration
Print-Info "Configuration:"
Write-Host "  Transport: $TransportMode"

if ($Http -or $TransportMode -eq "streamable-http") {
    Write-Host "  Host: $HostValue"
    Write-Host "  Port: $PortValue"
}

Write-Host ""
Print-Success "Launching MCP server..."
Write-Host ""

# Run the server
if ($CmdArgs.Count -gt 0) {
    & python mcp_server.py $CmdArgs
}
else {
    & python mcp_server.py
}
