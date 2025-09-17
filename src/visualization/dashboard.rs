//! # Real-time Process Dashboard
//!
//! Web-based dashboard for displaying dynamic Mermaid diagrams

use super::{SystemVisualizationState, MermaidDiagramGenerator};
use crate::cellular::Cell;
use crate::miraculous::MiraculousDynamics;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::time::{interval, Duration};

/// Real-time dashboard server
pub struct ProcessDashboard {
    /// Current system state
    state: Arc<Mutex<SystemVisualizationState>>,
    /// Mermaid diagram generator
    diagram_generator: MermaidDiagramGenerator,
    /// Update interval in milliseconds
    update_interval_ms: u64,
    /// Port to serve on
    port: u16,
}

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Update interval in milliseconds
    pub update_interval_ms: u64,
    /// Server port
    pub port: u16,
    /// Auto-refresh browser
    pub auto_refresh: bool,
    /// Enable animations
    pub enable_animations: bool,
    /// Diagram types to display
    pub diagram_types: Vec<DiagramType>,
}

/// Types of diagrams available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiagramType {
    SystemOverview,
    CellularNetwork,
    MiraculousStatus,
    PipelineFlow,
    SSpaceNavigation,
}

/// Dashboard response for API endpoints
#[derive(Debug, Serialize, Deserialize)]
pub struct DashboardResponse {
    /// Timestamp of the response
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Diagram type
    pub diagram_type: String,
    /// Mermaid diagram source
    pub mermaid_source: String,
    /// System metrics
    pub metrics: SystemMetrics,
}

/// Key system metrics for monitoring
#[derive(Debug, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// Number of active cells
    pub active_cells: usize,
    /// Average ATP level
    pub avg_atp_level: f64,
    /// Overall system efficiency
    pub system_efficiency: f64,
    /// Active miracles count
    pub active_miracles: usize,
    /// Global viability status
    pub globally_viable: bool,
    /// Current pipeline stage
    pub pipeline_stage: String,
    /// Distance to S-space target
    pub distance_to_target: f64,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            update_interval_ms: 1000,
            port: 3000,
            auto_refresh: true,
            enable_animations: true,
            diagram_types: vec![
                DiagramType::SystemOverview,
                DiagramType::CellularNetwork,
                DiagramType::MiraculousStatus,
                DiagramType::PipelineFlow,
            ],
        }
    }
}

impl ProcessDashboard {
    /// Create new dashboard
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            state: Arc::new(Mutex::new(SystemVisualizationState::new())),
            diagram_generator: MermaidDiagramGenerator::new(),
            update_interval_ms: config.update_interval_ms,
            port: config.port,
        }
    }

    /// Start the dashboard server
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        let state = self.state.clone();
        let diagram_generator = self.diagram_generator.clone();
        let port = self.port;

        // Start background state updater
        let state_updater = state.clone();
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(1000));
            loop {
                interval.tick().await;
                // In real implementation, this would update from actual system state
                // For now, simulate some state changes
                Self::simulate_state_update(&state_updater).await;
            }
        });

        // Start HTTP server
        self.start_http_server(state, diagram_generator, port).await
    }

    /// Update dashboard state from system components
    pub async fn update_state(&self, cells: &HashMap<String, Cell>, miraculous: &MiraculousDynamics) {
        let mut state = self.state.lock().unwrap();
        state.update_from_system(cells, miraculous);
    }

    /// Get current system metrics
    pub async fn get_metrics(&self) -> SystemMetrics {
        let state = self.state.lock().unwrap();
        SystemMetrics {
            active_cells: state.active_cells.len(),
            avg_atp_level: if !state.active_cells.is_empty() {
                state.active_cells.values().map(|c| c.atp_level).sum::<f64>() / state.active_cells.len() as f64
            } else {
                0.0
            },
            system_efficiency: if !state.active_cells.is_empty() {
                state.active_cells.values().map(|c| c.efficiency).sum::<f64>() / state.active_cells.len() as f64
            } else {
                0.0
            },
            active_miracles: state.miraculous_state.active_miracles.len(),
            globally_viable: state.miraculous_state.globally_viable,
            pipeline_stage: state.pipeline_state.current_stage.clone(),
            distance_to_target: state.navigation_state.distance_remaining,
        }
    }

    /// Generate HTML dashboard page
    pub async fn generate_dashboard_html(&self) -> String {
        let state = self.state.lock().unwrap();
        let overview_diagram = self.diagram_generator.generate_system_overview(&state);
        let cellular_diagram = self.diagram_generator.generate_cellular_network(&state);
        let miraculous_diagram = self.diagram_generator.generate_miraculous_status(&state);
        let pipeline_diagram = self.diagram_generator.generate_pipeline_flow(&state);
        
        format!(r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>S-Entropy Bioreactor Process Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .dashboard-container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .dashboard-header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .diagram-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .diagram-panel {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .diagram-panel h3 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #007acc;
            padding-bottom: 10px;
        }}
        .mermaid {{
            text-align: center;
        }}
        .status-bar {{
            display: flex;
            justify-content: space-around;
            background: #333;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .status-item {{
            text-align: center;
        }}
        .status-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #00ff88;
        }}
        .auto-refresh {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: #007acc;
            color: white;
            padding: 10px;
            border-radius: 20px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="auto-refresh">🔄 Auto-refresh: ON</div>
    
    <div class="dashboard-container">
        <div class="dashboard-header">
            <h1>🧬 S-Entropy Bioreactor Process Dashboard</h1>
            <p>Real-time visualization of cellular observer networks and miraculous dynamics</p>
            <p><strong>Last Updated:</strong> {}</p>
        </div>

        <div class="status-bar">
            <div class="status-item">
                <div class="status-value">{}</div>
                <div>Active Cells</div>
            </div>
            <div class="status-item">
                <div class="status-value">{:.1}</div>
                <div>Avg ATP (mM)</div>
            </div>
            <div class="status-item">
                <div class="status-value">{:.1}%</div>
                <div>Efficiency</div>
            </div>
            <div class="status-item">
                <div class="status-value">{}</div>
                <div>Active Miracles</div>
            </div>
            <div class="status-item">
                <div class="status-value">{}</div>
                <div>Global Viability</div>
            </div>
        </div>

        <div class="diagram-grid">
            <div class="diagram-panel">
                <h3>🌐 System Overview</h3>
                <div class="mermaid">
{}
                </div>
            </div>

            <div class="diagram-panel">
                <h3>🔬 Cellular Network</h3>
                <div class="mermaid">
{}
                </div>
            </div>

            <div class="diagram-panel">
                <h3>⚡ Miraculous Dynamics</h3>
                <div class="mermaid">
{}
                </div>
            </div>

            <div class="diagram-panel">
                <h3>⚙️ Processing Pipeline</h3>
                <div class="mermaid">
{}
                </div>
            </div>
        </div>
    </div>

    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
        
        // Auto-refresh every 2 seconds
        setTimeout(() => {{
            location.reload();
        }}, 2000);
    </script>
</body>
</html>
        "#,
            state.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            state.active_cells.len(),
            if !state.active_cells.is_empty() {
                state.active_cells.values().map(|c| c.atp_level).sum::<f64>() / state.active_cells.len() as f64
            } else { 0.0 },
            if !state.active_cells.is_empty() {
                state.active_cells.values().map(|c| c.efficiency * 100.0).sum::<f64>() / state.active_cells.len() as f64
            } else { 0.0 },
            state.miraculous_state.active_miracles.len(),
            if state.miraculous_state.globally_viable { "✅" } else { "⚠️" },
            overview_diagram,
            cellular_diagram,
            miraculous_diagram,
            pipeline_diagram
        )
    }

    /// Start HTTP server for dashboard
    async fn start_http_server(
        &self,
        state: Arc<Mutex<SystemVisualizationState>>,
        diagram_generator: MermaidDiagramGenerator,
        port: u16,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::convert::Infallible;
        use std::net::SocketAddr;
        use tokio::net::TcpListener;

        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        let listener = TcpListener::bind(addr).await?;

        println!("🚀 S-Entropy Dashboard running at http://localhost:{}", port);
        println!("📊 Real-time process diagrams available");
        
        loop {
            let (stream, _) = listener.accept().await?;
            let state = state.clone();
            let generator = diagram_generator.clone();
            
            tokio::spawn(async move {
                if let Err(e) = Self::handle_connection(stream, state, generator).await {
                    eprintln!("Error handling connection: {}", e);
                }
            });
        }
    }

    /// Handle individual HTTP connections
    async fn handle_connection(
        mut stream: tokio::net::TcpStream,
        state: Arc<Mutex<SystemVisualizationState>>,
        diagram_generator: MermaidDiagramGenerator,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        
        let mut buffer = [0; 1024];
        stream.read(&mut buffer).await?;

        let request = String::from_utf8_lossy(&buffer[..]);
        
        let response = if request.starts_with("GET / ") {
            // Serve main dashboard
            let dashboard = ProcessDashboard::generate_dashboard_html_static(&state, &diagram_generator).await;
            format!("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\n\r\n{}", dashboard.len(), dashboard)
        } else if request.starts_with("GET /api/metrics ") {
            // Serve metrics API
            let metrics = ProcessDashboard::get_metrics_static(&state).await;
            let json = serde_json::to_string(&metrics).unwrap();
            format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}", json.len(), json)
        } else {
            // 404 response
            let body = "Not Found";
            format!("HTTP/1.1 404 NOT FOUND\r\nContent-Length: {}\r\n\r\n{}", body.len(), body)
        };

        stream.write_all(response.as_bytes()).await?;
        stream.flush().await?;
        
        Ok(())
    }

    /// Static version of dashboard HTML generation
    async fn generate_dashboard_html_static(
        state: &Arc<Mutex<SystemVisualizationState>>,
        diagram_generator: &MermaidDiagramGenerator,
    ) -> String {
        let state = state.lock().unwrap();
        let overview_diagram = diagram_generator.generate_system_overview(&state);
        
        format!(r#"
<!DOCTYPE html>
<html><head><title>S-Entropy Dashboard</title></head>
<body>
<h1>S-Entropy Bioreactor Dashboard</h1>
<div class="mermaid">{}</div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad: true}});</script>
</body></html>
        "#, overview_diagram)
    }

    /// Static version of metrics generation
    async fn get_metrics_static(state: &Arc<Mutex<SystemVisualizationState>>) -> SystemMetrics {
        let state = state.lock().unwrap();
        SystemMetrics {
            active_cells: state.active_cells.len(),
            avg_atp_level: 0.0,
            system_efficiency: 0.0,
            active_miracles: state.miraculous_state.active_miracles.len(),
            globally_viable: state.miraculous_state.globally_viable,
            pipeline_stage: state.pipeline_state.current_stage.clone(),
            distance_to_target: state.navigation_state.distance_remaining,
        }
    }

    /// Simulate state updates for demonstration
    async fn simulate_state_update(state: &Arc<Mutex<SystemVisualizationState>>) {
        let mut state = state.lock().unwrap();
        state.timestamp = chrono::Utc::now();
        
        // Simulate some state changes for demo purposes
        state.pipeline_state.throughput = 50.0 + (chrono::Utc::now().timestamp() as f64 % 100.0);
        state.navigation_state.distance_remaining = 
            (state.navigation_state.distance_remaining - 1.0).max(0.0);
    }
}

/// Convenience function to start dashboard
pub async fn start_process_dashboard(
    config: DashboardConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let dashboard = ProcessDashboard::new(config);
    dashboard.start().await
}
