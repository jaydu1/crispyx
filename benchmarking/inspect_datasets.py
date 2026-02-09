#!/usr/bin/env python
"""Helper script to inspect dataset structures and recommend configurations."""

from __future__ import annotations

import sys
from pathlib import Path
import warnings
import anndata as ad
import pandas as pd
import numpy as np
import yaml
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crispyx.data import resolve_control_label, calculate_adaptive_qc_thresholds, calculate_optimal_chunk_size


def inspect_dataset(dataset_path: Path, memory_limit_gb: float | None = None) -> dict:
    """Inspect a single dataset and return configuration recommendations.
    
    Parameters
    ----------
    dataset_path : Path
        Path to the h5ad dataset file
    memory_limit_gb : float | None
        Memory limit in GB to use for chunk size calculation. If None, uses available system memory.
    """
    print(f"\n{'='*80}")
    print(f"Inspecting: {dataset_path.name}")
    print(f"{'='*80}")
    
    try:
        # Read dataset
        adata = ad.read_h5ad(dataset_path, backed='r')
        
        info = {
            'dataset_name': dataset_path.stem,
            'dataset_path': str(dataset_path),
            'n_obs': adata.n_obs,
            'n_vars': adata.n_vars,
            'obs_columns': list(adata.obs.columns),
            'var_columns': list(adata.var.columns) if hasattr(adata.var, 'columns') else [],
        }
        
        print(f"Shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
        print(f"\nobs columns: {info['obs_columns']}")
        print(f"var columns: {info['var_columns'][:10]}{'...' if len(info['var_columns']) > 10 else ''}")
        
        # Detect perturbation column
        perturbation_candidates = ['perturbation', 'gene', 'gene_target', 'condition', 'treatment']
        perturbation_column = None
        for candidate in perturbation_candidates:
            if candidate in adata.obs.columns:
                perturbation_column = candidate
                info['perturbation_column'] = candidate
                print(f"\n✓ Found perturbation column: '{candidate}'")
                
                # Show unique values and counts
                pert_counts = adata.obs[candidate].value_counts()
                n_unique = len(pert_counts)
                print(f"  - {n_unique} unique perturbations")
                print(f"  - Cells per perturbation: min={pert_counts.min()}, "
                      f"median={pert_counts.median():.0f}, max={pert_counts.max()}")
                
                # Try to detect control label
                try:
                    labels = adata.obs[candidate].astype(str).to_numpy()
                    control_label = resolve_control_label(labels, None, verbose=False)
                    info['control_label'] = control_label
                    n_control = (labels == control_label).sum()
                    print(f"  - Auto-detected control: '{control_label}' ({n_control} cells)")
                except Exception as e:
                    info['control_label'] = None
                    print(f"  - Control auto-detection failed: {e}")
                    # Show top 5 most common labels
                    print(f"  - Top 5 labels: {list(pert_counts.head(5).index)}")
                
                break
        
        if perturbation_column is None:
            info['perturbation_column'] = None
            print(f"\n✗ No perturbation column found in candidates: {perturbation_candidates}")
        
        # Detect gene name column
        gene_candidates = ['gene_symbols', 'gene_symbol', 'gene_name', 'gene_names', 'symbol']
        gene_name_column = None
        for candidate in gene_candidates:
            if candidate in adata.var.columns:
                gene_name_column = candidate
                info['gene_name_column'] = candidate
                print(f"\n✓ Found gene name column: '{candidate}'")
                # Check if var index is same as column
                if hasattr(adata.var, 'index') and candidate in adata.var.columns:
                    first_n = min(100, len(adata.var))
                    same_as_index = (adata.var.index[:first_n] == adata.var[candidate].values[:first_n]).all()
                    if same_as_index:
                        print(f"  - Column matches var.index (can use gene_name_column: null)")
                break
        
        if gene_name_column is None:
            info['gene_name_column'] = None
            print(f"\n✓ No gene column found, will use var.index")
            print(f"  - var.index sample: {list(adata.var.index[:5])}")
        
        # Calculate adaptive QC parameters
        if perturbation_column:
            try:
                adaptive_params = calculate_adaptive_qc_thresholds(
                    adata, 
                    perturbation_column, 
                    mode='conservative'
                )
                info['adaptive_qc'] = adaptive_params
                
                print(f"\n✓ Adaptive QC parameters (will be used at runtime):")
                print(f"  - min_genes: {adaptive_params['min_genes']}")
                print(f"  - min_cells_per_perturbation: {adaptive_params['min_cells_per_perturbation']}")
                print(f"  - min_cells_per_gene: {adaptive_params['min_cells_per_gene']}")
                
                # Calculate chunk size with memory constraint
                if memory_limit_gb is not None and memory_limit_gb > 0:
                    chunk_size = calculate_optimal_chunk_size(
                        adata.n_obs,
                        adata.n_vars,
                        available_memory_gb=memory_limit_gb
                    )
                    print(f"  - chunk_size: {chunk_size} (memory-constrained: {memory_limit_gb} GB)")
                    info['adaptive_qc']['chunk_size'] = chunk_size
                    info['memory_constrained_chunk'] = True
                else:
                    print(f"  - chunk_size: {adaptive_params['chunk_size']} (system memory)")
                    info['memory_constrained_chunk'] = False
                    
            except Exception as e:
                print(f"\n⚠ Could not calculate adaptive QC parameters: {e}")
                info['adaptive_qc'] = None
        
        adata.file.close()
        return info
        
    except Exception as e:
        print(f"✗ Error inspecting {dataset_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'dataset_name': dataset_path.stem,
            'dataset_path': str(dataset_path),
            'error': str(e)
        }


def generate_yaml_config(dataset_infos: list[dict], output_path: Path, memory_limit_gb: float | None = None, n_cores: int | None = None):
    """Generate individual YAML configuration files for each dataset.
    
    Parameters
    ----------
    dataset_infos : list[dict]
        List of dataset inspection results
    output_path : Path
        Directory path to write individual YAML config files
    memory_limit_gb : float | None
        Memory limit from reference config (if any)
    n_cores : int | None
        Number of cores from reference config (if any)
    """
    
    # Create output directory
    output_dir = output_path if output_path.is_dir() else output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Generating individual config files in: {output_dir}")
    print(f"{'='*80}")
    
    generated_files = []
    
    for info in dataset_infos:
        if 'error' in info:
            print(f"⊗ Skipping {info['dataset_name']} - Error: {info['error']}")
            continue
        
        dataset_name = info['dataset_name']
        config_file = output_dir / f"{dataset_name}.yaml"
        
        yaml_lines = [
            f"# Benchmark Configuration for {dataset_name}",
            f"# Auto-generated by inspect_datasets.py",
            "",
            f"dataset_path: \"{info['dataset_path']}\"",
            f"output_dir: \"benchmarking/results/{dataset_name}\"",
            "",
            "# Dataset column configuration",
        ]
        
        if info.get('perturbation_column'):
            yaml_lines.append(f"perturbation_column: \"{info['perturbation_column']}\"")
        else:
            yaml_lines.append(f"perturbation_column: \"perturbation\"  # TODO: VERIFY THIS")
        
        if info.get('control_label'):
            yaml_lines.append(f"control_label: null  # Auto-detects '{info['control_label']}'")
        else:
            yaml_lines.append(f"control_label: null  # TODO: VERIFY AUTO-DETECTION")
        
        if info.get('gene_name_column'):
            yaml_lines.append(f"gene_name_column: \"{info['gene_name_column']}\"")
        else:
            yaml_lines.append(f"gene_name_column: null  # Uses var.index")
        
        yaml_lines.extend([
            "",
            "# Quality control parameters - set to null to use adaptive calculation",
            "qc_params: null  # Will be calculated adaptively based on data distribution",
            "",
            "# Resource limits",
            "resource_limits:",
            "  time_limit: 36000  # 10 hours per method",
        ])
        
        if memory_limit_gb is not None and memory_limit_gb > 0:
            yaml_lines.append(f"  memory_limit: {memory_limit_gb}  # GB per method")
        else:
            yaml_lines.append("  memory_limit: 0  # No limit")
        
        yaml_lines.extend([
            "",
            "# Parallelization configuration",
            "parallel_config:",
        ])
        
        if n_cores is not None and n_cores > 0:
            yaml_lines.append(f"  n_cores: {n_cores}")
        else:
            yaml_lines.append("  n_cores: null  # Auto-detect available cores")
        
        yaml_lines.extend([
            "",
            "# Adaptive QC mode",
            "force_restandardize: false  # Set to true to regenerate standardized files",
            "adaptive_qc_mode: conservative  # or 'aggressive'",
            "",
            "# Methods to run (null = run all available methods)",
            "methods_to_run: null",
            "",
            "# Progress and output options",
            "show_progress: true",
            "quiet: false",
        ])
        
        with open(config_file, 'w') as f:
            f.write('\n'.join(yaml_lines))
        
        generated_files.append(config_file)
        print(f"✓ Generated: {config_file.name}")
    
    print(f"\n{'='*80}")
    print(f"Generated {len(generated_files)} config files")
    print(f"{'='*80}")
    
    return generated_files


def load_memory_limit_from_config(config_path: Path | None) -> float | None:
    """Load memory limit from reference YAML configuration.
    
    Parameters
    ----------
    config_path : Path | None
        Path to reference YAML config file
        
    Returns
    -------
    float | None
        Memory limit in GB, or None if not specified or config not found
    """
    if config_path is None or not config_path.exists():
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Try different paths in config
        if isinstance(config, dict):
            # Single dataset mode
            if 'resource_limits' in config and 'memory_limit' in config['resource_limits']:
                mem_limit = config['resource_limits']['memory_limit']
                if mem_limit and mem_limit > 0:
                    return float(mem_limit)
            
            # Multi-dataset mode
            if 'shared_config' in config and 'resource_limits' in config['shared_config']:
                mem_limit = config['shared_config']['resource_limits'].get('memory_limit')
                if mem_limit and mem_limit > 0:
                    return float(mem_limit)
        
        return None
    except Exception as e:
        print(f"⚠ Warning: Could not load memory limit from {config_path}: {e}")
        return None


def load_n_cores_from_config(config_path: Path | None) -> int | None:
    """Load n_cores from reference YAML configuration.
    
    Parameters
    ----------
    config_path : Path | None
        Path to reference YAML config file
        
    Returns
    -------
    int | None
        Number of cores, or None if not specified or config not found
    """
    if config_path is None or not config_path.exists():
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Try different paths in config
        if isinstance(config, dict):
            # Single dataset mode
            if 'parallel_config' in config and 'n_cores' in config['parallel_config']:
                n_cores = config['parallel_config']['n_cores']
                if n_cores is not None and n_cores > 0:
                    return int(n_cores)
            
            # Multi-dataset mode
            if 'shared_config' in config and 'parallel_config' in config['shared_config']:
                n_cores = config['shared_config']['parallel_config'].get('n_cores')
                if n_cores is not None and n_cores > 0:
                    return int(n_cores)
        
        return None
    except Exception as e:
        print(f"⚠ Warning: Could not load n_cores from {config_path}: {e}")
        return None


def main():
    """Inspect all datasets and generate configuration."""
    
    parser = argparse.ArgumentParser(description="Inspect CRISPR screen datasets and generate configuration")
    parser.add_argument(
        '--reference-config',
        type=Path,
        default=PROJECT_ROOT / "benchmarking" / "benchmark_config.yaml",
        help="Reference YAML config to read memory limit from (default: benchmark_config.yaml)"
    )
    parser.add_argument(
        '--memory-limit',
        type=float,
        default=None,
        help="Override memory limit in GB (overrides reference config)"
    )
    args = parser.parse_args()
    
    # Determine memory limit
    memory_limit_gb = args.memory_limit
    if memory_limit_gb is None:
        memory_limit_gb = load_memory_limit_from_config(args.reference_config)
        if memory_limit_gb is not None:
            print(f"Using memory limit from {args.reference_config}: {memory_limit_gb} GB")
    else:
        print(f"Using memory limit from command line: {memory_limit_gb} GB")
    
    if memory_limit_gb is None or memory_limit_gb == 0:
        print("No memory limit specified - using system memory for chunk size calculation")
    
    # Determine n_cores
    n_cores = load_n_cores_from_config(args.reference_config)
    if n_cores is not None:
        print(f"Using n_cores from {args.reference_config}: {n_cores}")
    else:
        print("No n_cores specified - will use auto-detect")
    
    dataset_names = [
        "Adamson", "Frangieh",
        "Replogle-GW-k562", "Replogle-E-k562", "Replogle-E-rpe1",
        "Tian-crispra", "Tian-crispri",
        "Nadig-HEPG2", "Nadig-JURKAT",
        "Feng-ts", "Feng-gwsf", "Feng-gwsnf",
        "Huang-HCT116", "Huang-HEK293T",
    ]
    
    data_dir = Path("/data/projects/SeqExpDesign/data/origin/")
    
    print(f"\nInspecting {len(dataset_names)} datasets from {data_dir}")
    
    dataset_infos = []
    for name in dataset_names:
        dataset_path = data_dir / f"{name}.h5ad"
        if not dataset_path.exists():
            print(f"\n✗ File not found: {dataset_path}")
            dataset_infos.append({
                'dataset_name': name,
                'dataset_path': str(dataset_path),
                'error': 'File not found'
            })
            continue
        
        info = inspect_dataset(dataset_path, memory_limit_gb=memory_limit_gb)
        dataset_infos.append(info)
    
    # Generate individual YAML configuration files
    output_dir = PROJECT_ROOT / "benchmarking" / "config"
    generated_files = generate_yaml_config(dataset_infos, output_dir, memory_limit_gb=memory_limit_gb, n_cores=n_cores)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    successful = [info for info in dataset_infos if 'error' not in info]
    failed = [info for info in dataset_infos if 'error' in info]
    
    print(f"\nSuccessfully inspected: {len(successful)}/{len(dataset_infos)} datasets")
    if failed:
        print(f"\nFailed datasets:")
        for info in failed:
            print(f"  - {info['dataset_name']}: {info['error']}")
    
    if successful:
        print(f"\nDataset size summary:")
        total_cells = sum(info['n_obs'] for info in successful)
        total_genes_avg = sum(info['n_vars'] for info in successful) / len(successful)
        print(f"  - Total cells across all datasets: {total_cells:,}")
        print(f"  - Average genes per dataset: {total_genes_avg:,.0f}")
    
    print(f"\nNext steps:")
    print(f"1. Review generated configs in: {output_dir}")
    print(f"2. Verify perturbation columns and control labels marked with TODO")
    print(f"3. QC parameters will be calculated adaptively at runtime (no manual config needed)")
    if memory_limit_gb and memory_limit_gb > 0:
        print(f"4. Chunk sizes calculated with memory constraint: {memory_limit_gb} GB")
    print(f"\nRun benchmarks:")
    print(f"  Single dataset:   ./run_single_dataset.sh config/Adamson.yaml")
    print(f"  Multiple datasets: ./run_multiple_datasets.sh config/Adamson.yaml config/Frangieh.yaml")
    print(f"  All datasets:      ./run_multiple_datasets.sh config/*.yaml")


if __name__ == "__main__":
    main()
