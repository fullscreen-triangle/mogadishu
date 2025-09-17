//! # Mermaid Diagram Generation
//!
//! Dynamic Mermaid diagram generation for real-time process visualization

use super::{SystemVisualizationState, CellVisualizationState, MiraculousDynamicsState};
use crate::s_entropy::SSpace;
use std::fmt::Write;

/// Mermaid diagram generator
pub struct MermaidDiagramGenerator {
    /// Diagram style configuration
    pub style_config: DiagramStyle,
}

/// Diagram styling configuration
#[derive(Debug, Clone)]
pub struct DiagramStyle {
    /// Colors for different states
    pub colors: DiagramColors,
    /// Node shapes for different components
    pub shapes: DiagramShapes,
    /// Animation settings
    pub animation: AnimationConfig,
}

/// Color scheme for diagrams
#[derive(Debug, Clone)]
pub struct DiagramColors {
    pub active_cell: String,
    pub inactive_cell: String,
    pub miraculous_active: String,
    pub miraculous_inactive: String,
    pub s_space_positive: String,
    pub s_space_negative: String,
    pub atp_high: String,
    pub atp_low: String,
    pub processing_membrane: String,
    pub processing_dna: String,
}

/// Node shapes for different components
#[derive(Debug, Clone)]
pub struct DiagramShapes {
    pub cell: String,
    pub observer: String,
    pub miracle: String,
    pub process: String,
    pub navigation: String,
}

/// Animation configuration
#[derive(Debug, Clone)]
pub struct AnimationConfig {
    pub enable_transitions: bool,
    pub update_interval_ms: u64,
    pub pulse_miraculous: bool,
}

impl Default for DiagramStyle {
    fn default() -> Self {
        Self {
            colors: DiagramColors {
                active_cell: "#00ff88".to_string(),
                inactive_cell: "#666666".to_string(),
                miraculous_active: "#ff6600".to_string(),
                miraculous_inactive: "#cccccc".to_string(),
                s_space_positive: "#0088ff".to_string(),
                s_space_negative: "#ff0088".to_string(),
                atp_high: "#88ff00".to_string(),
                atp_low: "#ff8800".to_string(),
                processing_membrane: "#00ffff".to_string(),
                processing_dna: "#ffff00".to_string(),
            },
            shapes: DiagramShapes {
                cell: "circle".to_string(),
                observer: "rect".to_string(),
                miracle: "diamond".to_string(),
                process: "round".to_string(),
                navigation: "hexagon".to_string(),
            },
            animation: AnimationConfig {
                enable_transitions: true,
                update_interval_ms: 1000,
                pulse_miraculous: true,
            },
        }
    }
}

impl MermaidDiagramGenerator {
    /// Create new generator with default styling
    pub fn new() -> Self {
        Self {
            style_config: DiagramStyle::default(),
        }
    }

    /// Create new generator with custom styling
    pub fn with_style(style: DiagramStyle) -> Self {
        Self {
            style_config: style,
        }
    }

    /// Generate complete system overview diagram
    pub fn generate_system_overview(&self, state: &SystemVisualizationState) -> String {
        let mut diagram = String::new();
        
        writeln!(diagram, "graph TB").unwrap();
        writeln!(diagram, "    %% S-Entropy Bioreactor System Overview").unwrap();
        writeln!(diagram, "    %% Generated at: {}", state.timestamp).unwrap();
        writeln!(diagram).unwrap();

        // S-Space Navigation
        self.add_s_space_navigation(&mut diagram, state);
        
        // Cellular Network
        self.add_cellular_network(&mut diagram, state);
        
        // Miraculous Dynamics
        self.add_miraculous_dynamics(&mut diagram, state);
        
        // Processing Pipeline
        self.add_processing_pipeline(&mut diagram, state);
        
        // Styling
        self.add_diagram_styling(&mut diagram);

        diagram
    }

    /// Generate detailed cellular network diagram
    pub fn generate_cellular_network(&self, state: &SystemVisualizationState) -> String {
        let mut diagram = String::new();
        
        writeln!(diagram, "graph LR").unwrap();
        writeln!(diagram, "    %% Cellular Observer Network").unwrap();
        writeln!(diagram, "    %% {} Active Cells", state.active_cells.len()).unwrap();
        writeln!(diagram).unwrap();

        // Central Observer Hub
        writeln!(diagram, "    ObsHub[\"🌐 Observer Hub<br/>S-Position: {:.1},{:.3},{:.1}\"]", 
                state.s_position.knowledge, state.s_position.time, state.s_position.entropy).unwrap();

        // Individual cells
        for (i, (id, cell)) in state.active_cells.iter().enumerate() {
            let atp_indicator = if cell.atp_level > 3.0 { "🔋" } else { "🪫" };
            let processing_indicator = match cell.processing_method.as_str() {
                "Membrane Quantum" => "⚛️",
                "DNA Consultation" => "🧬",
                _ => "⚙️",
            };
            
            writeln!(diagram, "    Cell{}[\"{}Cell {}<br/>{} ATP: {:.1}mM<br/>{}Efficiency: {:.1}%<br/>Challenges: {}\"]", 
                    i, atp_indicator, id, processing_indicator, cell.atp_level, 
                    processing_indicator, cell.efficiency * 100.0, cell.active_challenges).unwrap();
            
            writeln!(diagram, "    ObsHub --> Cell{}", i).unwrap();
            
            // Color coding based on state
            let color = if cell.atp_level > 3.0 && cell.efficiency > 0.95 {
                &self.style_config.colors.active_cell
            } else {
                &self.style_config.colors.inactive_cell
            };
            
            writeln!(diagram, "    Cell{} --> |\"S-Pos: {:.0},{:.3},{:.0}\"| ObsHub", 
                    i, cell.s_position.knowledge, cell.s_position.time, cell.s_position.entropy).unwrap();
        }

        // Styling
        self.add_diagram_styling(&mut diagram);
        
        diagram
    }

    /// Generate miraculous dynamics status diagram
    pub fn generate_miraculous_status(&self, state: &SystemVisualizationState) -> String {
        let mut diagram = String::new();
        
        writeln!(diagram, "graph TD").unwrap();
        writeln!(diagram, "    %% Miraculous Dynamics Status").unwrap();
        writeln!(diagram, "    %% Global Viability: {}", state.miraculous_state.globally_viable).unwrap();
        writeln!(diagram).unwrap();

        // Viability Center
        let viability_symbol = if state.miraculous_state.globally_viable { "✅" } else { "⚠️" };
        writeln!(diagram, "    Viability[\"{} Global S-Viability<br/>Status: {}\"]", 
                viability_symbol, 
                if state.miraculous_state.globally_viable { "VIABLE" } else { "VIOLATED" }).unwrap();

        // Knowledge Miracle
        let knowledge_symbol = if state.miraculous_state.knowledge_miracle > 1000.0 { "🧠⚡" } else { "🧠" };
        writeln!(diagram, "    Knowledge[\"{} Knowledge<br/>Level: {:.1}<br/>{}\"]", 
                knowledge_symbol, 
                state.miraculous_state.knowledge_miracle,
                if state.miraculous_state.knowledge_miracle > 1000.0 { "MIRACULOUS" } else { "Normal" }).unwrap();

        // Time Miracle  
        let time_symbol = if state.miraculous_state.time_miracle < 0.001 { "⚡⏰" } else { "⏰" };
        writeln!(diagram, "    Time[\"{} Time<br/>Factor: {:.6}<br/>{}\"]", 
                time_symbol, 
                state.miraculous_state.time_miracle,
                if state.miraculous_state.time_miracle < 0.001 { "MIRACULOUS" } else { "Normal" }).unwrap();

        // Entropy Miracle
        let entropy_symbol = if state.miraculous_state.entropy_miracle < 0.0 { "❄️⚡" } else { "🌡️" };
        writeln!(diagram, "    Entropy[\"{} Entropy<br/>Level: {:.1}<br/>{}\"]", 
                entropy_symbol, 
                state.miraculous_state.entropy_miracle,
                if state.miraculous_state.entropy_miracle < 0.0 { "MIRACULOUS" } else { "Normal" }).unwrap();

        // Connections
        writeln!(diagram, "    Knowledge --> Viability").unwrap();
        writeln!(diagram, "    Time --> Viability").unwrap();
        writeln!(diagram, "    Entropy --> Viability").unwrap();

        // Active miracles list
        if !state.miraculous_state.active_miracles.is_empty() {
            writeln!(diagram, "    ActiveMiracles[\"🎯 Active Miracles:").unwrap();
            for miracle in &state.miraculous_state.active_miracles {
                writeln!(diagram, "    • {}", miracle).unwrap();
            }
            writeln!(diagram, "    \"]").unwrap();
            writeln!(diagram, "    Viability --> ActiveMiracles").unwrap();
        }

        self.add_diagram_styling(&mut diagram);
        
        diagram
    }

    /// Generate processing pipeline flow diagram  
    pub fn generate_pipeline_flow(&self, state: &SystemVisualizationState) -> String {
        let mut diagram = String::new();
        
        writeln!(diagram, "flowchart LR").unwrap();
        writeln!(diagram, "    %% Processing Pipeline Flow").unwrap();
        writeln!(diagram, "    %% Current Stage: {}", state.pipeline_state.current_stage).unwrap();
        writeln!(diagram).unwrap();

        // Completed stages
        for (i, stage) in state.pipeline_state.completed_stages.iter().enumerate() {
            writeln!(diagram, "    S{}[\"✅ {}\"]", i, stage).unwrap();
            writeln!(diagram, "    S{} --> S{}", i, i + 1).unwrap();
        }

        // Current stage
        let current_idx = state.pipeline_state.completed_stages.len();
        writeln!(diagram, "    S{}[\"🔄 {}\"]", current_idx, state.pipeline_state.current_stage).unwrap();
        
        if current_idx > 0 {
            writeln!(diagram, "    S{} --> S{}", current_idx - 1, current_idx).unwrap();
        }

        // Pending stages
        for (i, stage) in state.pipeline_state.pending_stages.iter().enumerate() {
            let pending_idx = current_idx + i + 1;
            writeln!(diagram, "    S{}[\"⏳ {}\"]", pending_idx, stage).unwrap();
            writeln!(diagram, "    S{} --> S{}", pending_idx - 1, pending_idx).unwrap();
        }

        // Throughput indicator
        writeln!(diagram, "    Throughput[\"📊 Throughput<br/>{:.1} units/sec\"]", 
                state.pipeline_state.throughput).unwrap();
        writeln!(diagram, "    S{} -.-> Throughput", current_idx).unwrap();

        // Bottlenecks
        if !state.pipeline_state.bottlenecks.is_empty() {
            writeln!(diagram, "    Bottlenecks[\"⚠️ Bottlenecks:").unwrap();
            for bottleneck in &state.pipeline_state.bottlenecks {
                writeln!(diagram, "    • {}", bottleneck).unwrap();
            }
            writeln!(diagram, "    \"]").unwrap();
            writeln!(diagram, "    S{} -.-> Bottlenecks", current_idx).unwrap();
        }

        self.add_diagram_styling(&mut diagram);
        
        diagram
    }

    /// Add S-space navigation to diagram
    fn add_s_space_navigation(&self, diagram: &mut String, state: &SystemVisualizationState) {
        let nav = &state.navigation_state;
        
        writeln!(diagram, "    %% S-Space Navigation").unwrap();
        writeln!(diagram, "    SSpaceCurrent[\"📍 Current S-Position<br/>K:{:.1} T:{:.3} E:{:.1}\"]", 
                nav.current_position.knowledge, nav.current_position.time, nav.current_position.entropy).unwrap();
        writeln!(diagram, "    SSpaceTarget[\"🎯 Target S-Position<br/>K:{:.1} T:{:.3} E:{:.1}\"]", 
                nav.target_position.knowledge, nav.target_position.time, nav.target_position.entropy).unwrap();
        writeln!(diagram, "    SSpaceDist[\"📏 Distance: {:.1}<br/>Strategy: {}\"]", 
                nav.distance_remaining, nav.strategy).unwrap();
        
        writeln!(diagram, "    SSpaceCurrent --> SSpaceDist").unwrap();
        writeln!(diagram, "    SSpaceDist --> SSpaceTarget").unwrap();
        writeln!(diagram).unwrap();
    }

    /// Add cellular network to diagram
    fn add_cellular_network(&self, diagram: &mut String, state: &SystemVisualizationState) {
        writeln!(diagram, "    %% Cellular Observer Network").unwrap();
        writeln!(diagram, "    CellNetwork[\"🔬 {} Active Cells<br/>Avg ATP: {:.1}mM\"]", 
                state.active_cells.len(), 
                state.active_cells.values().map(|c| c.atp_level).sum::<f64>() / state.active_cells.len().max(1) as f64).unwrap();
        
        writeln!(diagram, "    SSpaceCurrent --> CellNetwork").unwrap();
        writeln!(diagram).unwrap();
    }

    /// Add miraculous dynamics to diagram
    fn add_miraculous_dynamics(&self, diagram: &mut String, state: &SystemVisualizationState) {
        writeln!(diagram, "    %% Miraculous Dynamics").unwrap();
        let miracle_status = if state.miraculous_state.globally_viable { "✅" } else { "⚠️" };
        let active_count = state.miraculous_state.active_miracles.len();
        
        writeln!(diagram, "    Miracles[\"{} Miraculous Dynamics<br/>{} Active Miracles<br/>Global Viability: {}\"]", 
                miracle_status, active_count, state.miraculous_state.globally_viable).unwrap();
        
        writeln!(diagram, "    CellNetwork --> Miracles").unwrap();
        writeln!(diagram).unwrap();
    }

    /// Add processing pipeline to diagram
    fn add_processing_pipeline(&self, diagram: &mut String, state: &SystemVisualizationState) {
        writeln!(diagram, "    %% Processing Pipeline").unwrap();
        writeln!(diagram, "    Pipeline[\"⚙️ {}<br/>Throughput: {:.1}/sec<br/>{}/{} Complete\"]", 
                state.pipeline_state.current_stage,
                state.pipeline_state.throughput,
                state.pipeline_state.completed_stages.len(),
                state.pipeline_state.completed_stages.len() + 1 + state.pipeline_state.pending_stages.len()).unwrap();
        
        writeln!(diagram, "    Miracles --> Pipeline").unwrap();
        writeln!(diagram).unwrap();
    }

    /// Add diagram styling
    fn add_diagram_styling(&self, diagram: &mut String) {
        writeln!(diagram, "    %% Styling").unwrap();
        writeln!(diagram, "    classDef active fill:{},stroke:#000,stroke-width:2px", self.style_config.colors.active_cell).unwrap();
        writeln!(diagram, "    classDef inactive fill:{},stroke:#000,stroke-width:1px", self.style_config.colors.inactive_cell).unwrap();
        writeln!(diagram, "    classDef miraculous fill:{},stroke:#ff0000,stroke-width:3px", self.style_config.colors.miraculous_active).unwrap();
        writeln!(diagram, "    classDef sspace fill:{},stroke:#0000ff,stroke-width:2px", self.style_config.colors.s_space_positive).unwrap();
        writeln!(diagram).unwrap();
    }
}

impl Default for MermaidDiagramGenerator {
    fn default() -> Self {
        Self::new()
    }
}
