//! Phase 3: Deterministic Compatibility Scorer
//!
//! Evaluates a matrix of (quantization × context × backend × run_mode)
//! configurations for each model, selects the highest quality safe
//! configuration, and assigns a deterministic fit category.
//!
//! All classifications emerge from HardwareProfile + ModelMetadata +
//! formulas. Zero GPU-specific or model-specific branches.

use crate::model_recommendation::traits::*;
use crate::model_recommendation::estimator;
use crate::model_recommendation::runtime;

/// Context checkpoints to evaluate (in tokens)
const CONTEXT_CHECKPOINTS: &[u32] = &[2048, 4096, 8192, 16384, 32768, 65536, 131072];

/// Evaluate all configurations for a single model against the memory budget.
/// Returns the best ModelRecommendation, or None if the model cannot fit at all.
pub fn evaluate_model(
    model: &ModelMetadata,
    budget: &MemoryBudget,
    estimator_config: &EstimatorConfig,
) -> Option<ModelRecommendation> {
    let quant_hierarchy = estimator::quantization_hierarchy();
    let mut best_config: Option<(EvaluatedConfiguration, f64)> = None;
    let mut max_possible_context: u32 = 0;

    // Evaluate each quantization from highest to lowest quality
    for quant in &quant_hierarchy {
        // Evaluate each context checkpoint (ascending)
        for &context in CONTEXT_CHECKPOINTS {
            if context > model.max_context_length {
                continue;
            }

            let (w, kv, oh, total) = estimator::estimate_total_memory(
                model, quant, context, estimator_config,
            );

            // Try each GPU for PureGpu mode
            for gpu_budget in &budget.gpu_budgets {
                let backends = runtime::compatible_backends(gpu_budget);
                if backends.is_empty() {
                    continue;
                }

                // PureGpu: try fitting entirely in dedicated VRAM
                if gpu_budget.usable_dedicated_vram > 0 && total <= gpu_budget.usable_dedicated_vram {
                    let headroom = (gpu_budget.usable_dedicated_vram - total) as f64
                        / gpu_budget.usable_dedicated_vram as f64;
                    let backend = runtime::select_preferred_backend(&backends, gpu_budget.cuda_available);
                    let config = EvaluatedConfiguration {
                        quantization: quant.clone(),
                        context_length: context,
                        run_mode: RunMode::PureGpu { gpu_index: gpu_budget.gpu_index },
                        backend: backend.clone(),
                        weight_memory_bytes: w,
                        kv_cache_memory_bytes: kv,
                        overhead_memory_bytes: oh,
                        total_memory_bytes: total,
                        vram_required_bytes: total,
                        ram_required_bytes: 0,
                        shared_mem_required_bytes: 0,
                        headroom_ratio: headroom,
                        fits: true,
                    };
                    let score = compute_score(&config, quant, model);
                    if context > max_possible_context {
                        max_possible_context = context;
                    }
                    if best_config.as_ref().map_or(true, |(_, bs)| score > *bs) {
                        best_config = Some((config, score));
                    }
                }

                // GpuWithCpuOffload: use VRAM for weights, overflow KV+overhead to RAM
                // Only if backend supports offloading
                let offload_backend = runtime::select_preferred_backend(&backends, gpu_budget.cuda_available);
                if offload_backend.supports_cpu_offload() && gpu_budget.usable_dedicated_vram > 0 {
                    let vram_available = gpu_budget.usable_dedicated_vram;
                    if w <= vram_available && total > vram_available {
                        let ram_needed = total - vram_available;
                        // Also consider shared memory for backends that support it
                        let shared_available = if offload_backend.supports_shared_memory() {
                            gpu_budget.usable_shared_memory
                        } else {
                            0
                        };
                        let total_offload_pool = budget.system_ram.usable_for_inference + shared_available;
                        if ram_needed <= total_offload_pool {
                            let offload_frac = ram_needed as f64 / total as f64;
                            let binding_pool = vram_available + total_offload_pool;
                            let headroom = (binding_pool - total) as f64 / binding_pool as f64;
                            let shared_used = ram_needed.min(shared_available);
                            let ram_used = ram_needed.saturating_sub(shared_used);
                            let config = EvaluatedConfiguration {
                                quantization: quant.clone(),
                                context_length: context,
                                run_mode: RunMode::GpuWithCpuOffload {
                                    gpu_index: gpu_budget.gpu_index,
                                    offload_fraction: offload_frac,
                                },
                                backend: offload_backend.clone(),
                                weight_memory_bytes: w,
                                kv_cache_memory_bytes: kv,
                                overhead_memory_bytes: oh,
                                total_memory_bytes: total,
                                vram_required_bytes: vram_available.min(total),
                                ram_required_bytes: ram_used,
                                shared_mem_required_bytes: shared_used,
                                headroom_ratio: headroom,
                                fits: true,
                            };
                            let score = compute_score(&config, quant, model);
                            if context > max_possible_context {
                                max_possible_context = context;
                            }
                            if best_config.as_ref().map_or(true, |(_, bs)| score > *bs) {
                                best_config = Some((config, score));
                            }
                        }
                    }
                }
            }

            // PureCpu: use only system RAM
            if total <= budget.system_ram.usable_for_inference {
                let headroom = (budget.system_ram.usable_for_inference - total) as f64
                    / budget.system_ram.usable_for_inference as f64;
                let cpu_backends = runtime::cpu_only_backends();
                let backend = cpu_backends.first().cloned().unwrap_or(InferenceBackend::LlamaCppGguf);
                let config = EvaluatedConfiguration {
                    quantization: quant.clone(),
                    context_length: context,
                    run_mode: RunMode::PureCpu,
                    backend,
                    weight_memory_bytes: w,
                    kv_cache_memory_bytes: kv,
                    overhead_memory_bytes: oh,
                    total_memory_bytes: total,
                    vram_required_bytes: 0,
                    ram_required_bytes: total,
                    shared_mem_required_bytes: 0,
                    headroom_ratio: headroom,
                    fits: true,
                };
                let score = compute_score(&config, quant, model);
                if context > max_possible_context {
                    max_possible_context = context;
                }
                // Only use CPU path if no GPU path was found with higher score
                if best_config.as_ref().map_or(true, |(_, bs)| score > *bs) {
                    best_config = Some((config, score));
                }
            }
        }
    }

    // Build the recommendation from the best configuration
    best_config.map(|(config, score)| {
        let category = assign_category(&config);
        let explanation = generate_explanation(model, &config, &category, &budget);
        let warnings = generate_warnings(&config, &category);
        let arch_label = match &model.architecture {
            ModelArchitecture::Dense => "Dense".to_string(),
            ModelArchitecture::MixtureOfExperts { num_experts, active_experts } =>
                format!("MoE ({}×, {} active)", num_experts, active_experts),
        };

        ModelRecommendation {
            model_id: model.id.clone(),
            model_name: model.name.clone(),
            model_family: model.family.clone(),
            provider_id: None,
            quantization: config.quantization.label.clone(),
            quantization_bits_per_weight: config.quantization.bits_per_weight,
            recommended_context: config.context_length,
            max_possible_context: max_possible_context,
            backend: config.backend.display_name().to_string(),
            run_mode: runtime::format_run_mode(&config.run_mode),
            estimated_vram_bytes: config.vram_required_bytes,
            estimated_ram_bytes: config.ram_required_bytes,
            estimated_shared_mem_bytes: config.shared_mem_required_bytes,
            estimated_total_memory_bytes: config.total_memory_bytes,
            headroom_percent: config.headroom_ratio * 100.0,
            fit_score: score,
            category,
            confidence: if config.headroom_ratio >= 0.15 { "High".into() } else if config.headroom_ratio >= 0.05 { "Medium".into() } else { "Low".into() },
            explanation,
            warnings,
            architecture: arch_label,
            total_parameters: model.total_parameters,
            active_parameters: model.active_parameters,
            estimated_tokens_per_sec: None,
            performance_note: "Unknown — no benchmark data available".to_string(),
        }
    })
}

/// Compute a deterministic score for a configuration.
/// Score = (memory_fit × 0.35) + (quant_quality × 0.25) + (context × 0.20) + (accel × 0.20)
fn compute_score(config: &EvaluatedConfiguration, quant: &QuantizationSpec, model: &ModelMetadata) -> f64 {
    // Memory fit: headroom ratio scaled (50% headroom = 1.0)
    let memory_fit = (config.headroom_ratio * 2.0).min(1.0).max(0.0);

    // Quantization quality: normalized from quality_rank
    let quant_quality = quant.quality_rank as f64 / 10.0;

    // Context score: log-scaled relative to model max
    let context_score = if model.max_context_length > 2048 {
        let log_ctx = (config.context_length as f64 / 2048.0).log2().max(0.0);
        let log_max = (model.max_context_length as f64 / 2048.0).log2().max(1.0);
        (log_ctx / log_max).min(1.0)
    } else {
        1.0
    };

    // Acceleration score: GPU > Offload > CPU
    let accel_score = match &config.run_mode {
        RunMode::PureGpu { .. } => 1.0,
        RunMode::GpuWithCpuOffload { offload_fraction, .. } => 0.8 - offload_fraction * 0.3,
        RunMode::MultiGpu { .. } => 0.9,
        RunMode::PureCpu => 0.3,
    };

    (memory_fit * 0.35 + quant_quality * 0.25 + context_score * 0.20 + accel_score * 0.20) * 100.0
}

/// Assign a FitCategory based on configuration characteristics.
fn assign_category(config: &EvaluatedConfiguration) -> FitCategory {
    let is_pure_gpu = matches!(&config.run_mode, RunMode::PureGpu { .. });
    let is_light_offload = matches!(&config.run_mode,
        RunMode::GpuWithCpuOffload { offload_fraction, .. } if *offload_fraction <= 0.20
    );

    if config.headroom_ratio >= 0.20 && (is_pure_gpu || is_light_offload) && config.quantization.quality_rank >= 4 {
        FitCategory::Recommended
    } else if config.headroom_ratio >= 0.05 && config.quantization.quality_rank >= 3 {
        FitCategory::Compatible
    } else {
        FitCategory::MayRun
    }
}

/// Generate a human-readable explanation for a recommendation.
fn generate_explanation(
    model: &ModelMetadata,
    config: &EvaluatedConfiguration,
    category: &FitCategory,
    budget: &MemoryBudget,
) -> String {
    let mem_gb = |bytes: u64| format!("{:.1} GB", bytes as f64 / 1_073_741_824.0);

    let run_desc = match &config.run_mode {
        RunMode::PureGpu { gpu_index } => {
            let gpu_name = budget.gpu_budgets.get(*gpu_index)
                .map(|g| g.gpu_model.as_str())
                .unwrap_or("GPU");
            format!("Fits entirely in {}'s VRAM", gpu_name)
        }
        RunMode::GpuWithCpuOffload { gpu_index, offload_fraction } => {
            let gpu_name = budget.gpu_budgets.get(*gpu_index)
                .map(|g| g.gpu_model.as_str())
                .unwrap_or("GPU");
            format!("Runs on {} with {:.0}% offloaded to CPU/RAM", gpu_name, offload_fraction * 100.0)
        }
        RunMode::MultiGpu { .. } => "Distributed across multiple GPUs".to_string(),
        RunMode::PureCpu => "Runs entirely on CPU using system RAM".to_string(),
    };

    let category_qualifier = match category {
        FitCategory::Recommended => "comfortably",
        FitCategory::Compatible => "with acceptable resource usage",
        FitCategory::MayRun => "with tight resources — performance not guaranteed",
    };

    format!(
        "{} at {} quantization with {} token context. {} {} ({} total, {:.0}% headroom).",
        model.name,
        config.quantization.label,
        config.context_length,
        run_desc,
        category_qualifier,
        mem_gb(config.total_memory_bytes),
        config.headroom_ratio * 100.0,
    )
}

/// Generate warnings for edge cases.
fn generate_warnings(config: &EvaluatedConfiguration, category: &FitCategory) -> Vec<String> {
    let mut warnings = Vec::new();

    if *category == FitCategory::MayRun {
        warnings.push("May run on this system; performance and stability are not guaranteed.".into());
    }

    if matches!(&config.run_mode, RunMode::PureCpu) {
        warnings.push("CPU-only inference is significantly slower than GPU-accelerated.".into());
    }

    if let RunMode::GpuWithCpuOffload { offload_fraction, .. } = &config.run_mode {
        if *offload_fraction > 0.50 {
            warnings.push("More than 50% of model is offloaded to CPU; expect reduced speed.".into());
        }
    }

    if config.quantization.quality_rank <= 3 {
        warnings.push(format!("Using {} quantization may noticeably reduce output quality.", config.quantization.label));
    }

    if config.headroom_ratio < 0.10 {
        warnings.push("Tight memory headroom. Other running applications may cause issues.".into());
    }

    warnings
}

/// Generate recommendations for all models in the catalog.
pub fn generate_all_recommendations(
    models: &[ModelMetadata],
    budget: &MemoryBudget,
    estimator_config: &EstimatorConfig,
) -> Vec<ModelRecommendation> {
    let mut recommendations: Vec<ModelRecommendation> = models
        .iter()
        .filter_map(|m| evaluate_model(m, budget, estimator_config))
        .collect();

    // Sort by fit_score descending
    recommendations.sort_by(|a, b| b.fit_score.partial_cmp(&a.fit_score).unwrap_or(std::cmp::Ordering::Equal));

    log::info!(
        "[RECOMMENDATION] Generated {} recommendations ({} Recommended, {} Compatible, {} May Run)",
        recommendations.len(),
        recommendations.iter().filter(|r| r.category == FitCategory::Recommended).count(),
        recommendations.iter().filter(|r| r.category == FitCategory::Compatible).count(),
        recommendations.iter().filter(|r| r.category == FitCategory::MayRun).count(),
    );

    recommendations
}

// ─── Unit Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_recommendation::catalog::bootstrap_models;

    const GB: u64 = 1_073_741_824;

    fn make_budget_8gb_dgpu() -> MemoryBudget {
        MemoryBudget {
            gpu_budgets: vec![GpuMemoryBudget {
                gpu_index: 0, gpu_model: "RTX 5060".into(), gpu_type: GpuType::Dedicated,
                total_dedicated_vram: 8 * GB, usable_dedicated_vram: 7 * GB,
                total_shared_memory: 12 * GB, usable_shared_memory: 6 * GB,
                cuda_available: true, rocm_available: false, vulkan_available: true,
                directml_available: true, compute_capability: Some("12.0".into()),
            }],
            system_ram: SystemRamBudget {
                total_bytes: 24 * GB, available_bytes: 14 * GB,
                usable_for_inference: 12 * GB, ram_speed_mts: Some(5600),
            },
        }
    }

    fn make_budget_cpu_only_8gb() -> MemoryBudget {
        MemoryBudget {
            gpu_budgets: vec![],
            system_ram: SystemRamBudget {
                total_bytes: 8 * GB, available_bytes: 4 * GB,
                usable_for_inference: 3 * GB, ram_speed_mts: Some(3200),
            },
        }
    }

    #[test]
    fn test_category_recommended_7b() {
        let models = bootstrap_models();
        let model = models.iter().find(|m| m.id == "Qwen/Qwen2.5-Coder-7B").unwrap();
        let budget = make_budget_8gb_dgpu();
        let rec = evaluate_model(model, &budget, &EstimatorConfig::default());
        assert!(rec.is_some());
        let rec = rec.unwrap();
        assert_eq!(rec.category, FitCategory::Recommended);
        assert!(rec.headroom_percent > 15.0);
    }

    #[test]
    fn test_category_compatible_14b() {
        let models = bootstrap_models();
        let model = models.iter().find(|m| m.id == "Qwen/Qwen2.5-14B").unwrap();
        let budget = make_budget_8gb_dgpu();
        let rec = evaluate_model(model, &budget, &EstimatorConfig::default());
        assert!(rec.is_some());
        let rec = rec.unwrap();
        // 14B should need offload or tighter fit on 8GB VRAM
        assert!(rec.category == FitCategory::Compatible || rec.category == FitCategory::MayRun);
    }

    #[test]
    fn test_category_filtered_insufficient() {
        // 70B-scale model should not fit in 3GB RAM CPU-only
        let model = ModelMetadata {
            id: "test/test-70b".into(), name: "Test 70B".into(), family: "Test".into(),
            architecture: ModelArchitecture::Dense, total_parameters: 70_000_000_000,
            active_parameters: None, num_layers: 80, num_attention_heads: 64, num_kv_heads: 8,
            head_dimension: 128, hidden_size: 8192, max_context_length: 8192, vocab_size: 32000,
            default_dtype: "bf16".into(), use_cases: vec![], catalog_version: "1.0".into(),
        };
        let budget = make_budget_cpu_only_8gb();
        let rec = evaluate_model(&model, &budget, &EstimatorConfig::default());
        // Should be None (filtered out — doesn't fit)
        assert!(rec.is_none(), "70B model should not fit in 3GB usable RAM");
    }

    #[test]
    fn test_no_hardcoded_results() {
        let models = bootstrap_models();
        let budget_big = make_budget_8gb_dgpu();
        let budget_small = make_budget_cpu_only_8gb();
        let recs_big = generate_all_recommendations(&models, &budget_big, &EstimatorConfig::default());
        let recs_small = generate_all_recommendations(&models, &budget_small, &EstimatorConfig::default());
        // Different hardware profiles should produce different results
        assert_ne!(recs_big.len(), recs_small.len());
    }

    #[test]
    fn test_quantization_selects_highest_quality() {
        let models = bootstrap_models();
        let model = models.iter().find(|m| m.id == "meta-llama/Llama-3.2-1B").unwrap();
        let budget = make_budget_8gb_dgpu();
        let rec = evaluate_model(model, &budget, &EstimatorConfig::default()).unwrap();
        // 1.2B model should fit at very high quantization on 7GB VRAM
        assert!(rec.quantization_bits_per_weight >= 5.0,
            "1.2B model should use Q5_K_M or better on 7GB VRAM, got {}", rec.quantization);
    }

    #[test]
    fn test_moe_uses_total_params_for_memory() {
        let models = bootstrap_models();
        let mixtral = models.iter().find(|m| m.id == "mistralai/Mixtral-8x7B-v0.1").unwrap();
        let q4 = estimator::quantization_hierarchy().iter().find(|q| q.label == "Q4_K_M").unwrap().clone();
        let weight_mem = estimator::estimate_weight_memory(mixtral, &q4);
        assert!(weight_mem > 25 * GB, "MoE should use total params (46.7B) for weight memory");
    }

    #[test]
    fn test_physical_pc_validation() {
        use crate::system_analyzer::get_system_analyzer_manager;
        let analyzer = get_system_analyzer_manager();
        let profile = analyzer.analyze_system().expect("Physical system scan failed");
        let recs = crate::model_recommendation::generate_recommendations(&profile);
        assert!(!recs.is_empty(), "Recommendation engine must return recommendations for physical PC");
        
        println!("\n=== PHYSICAL PC RECOMMENDATION RESULTS ===");
        println!("PC Hardware: {}", profile.cpu.current().model);
        for g in profile.gpus.current() {
            println!("GPU: {} ({}, Dedicated: {} GB, Shared: {} GB)", 
                g.model, g.gpu_type, 
                g.dedicated_video_memory_bytes / 1073741824,
                g.shared_system_memory_bytes / 1073741824);
        }
        println!("System RAM: {} GB total, {} GB available",
            profile.memory.current().total_bytes / 1073741824,
            profile.memory.current().available_bytes / 1073741824);
        println!("------------------------------------------");
        for r in &recs {
            println!("[{:?}] {} ({}) -> {} | Context: {} | Headroom: {:.1}%",
                r.category, r.model_name, r.architecture, r.quantization, r.recommended_context, r.headroom_percent);
        }
        println!("==========================================\n");
    }
}
