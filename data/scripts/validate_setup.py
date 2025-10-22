"""
Validation script to check if all paths and dependencies are properly configured.

Usage:
    python validate_setup.py
"""

import os
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print(f"  ❌ Python {major}.{minor} detected. Python 3.8+ required.")
        return False
    print(f"  ✓ Python {major}.{minor} detected")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    required = [
        'hydra',
        'omegaconf',
        'pandas',
        'numpy',
        'scipy',
        'plotly',
        'tqdm'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ❌ {package} not found")
            missing.append(package)
    
    if missing:
        print(f"\n  Install missing packages with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def check_config_file():
    """Check if config file exists."""
    print("\nChecking configuration file...")
    config_path = Path(__file__).parent.parent.parent / "configs" / "data" / "prepare_mimic.yaml"
    
    if not config_path.exists():
        print(f"  ❌ Config file not found: {config_path}")
        return False
    
    print(f"  ✓ Config file found: {config_path}")
    return True


def check_mimic_paths():
    """Check if MIMIC-III paths are accessible (requires config to be loaded)."""
    print("\nChecking MIMIC-III paths...")
    print("  Note: This requires proper configuration in prepare_mimic.yaml")
    
    try:
        from omegaconf import OmegaConf
        config_path = Path(__file__).parent.parent.parent / "configs" / "data" / "prepare_mimic.yaml"
        cfg = OmegaConf.load(config_path)
        
        paths_to_check = {
            'mimic_path': cfg.mimic_path,
            'path_root': cfg.path_root,
            'path_ihm': cfg.path_ihm,
            'path_phe': cfg.path_phe,
        }
        
        all_exist = True
        for name, path in paths_to_check.items():
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                print(f"  ✓ {name}: {expanded_path}")
            else:
                print(f"  ❌ {name} not found: {expanded_path}")
                all_exist = False
        
        return all_exist
    
    except Exception as e:
        print(f"  ⚠ Could not validate paths: {e}")
        print(f"  Please manually verify paths in the config file")
        return None


def check_scripts():
    """Check if all script files exist."""
    print("\nChecking script files...")
    scripts = [
        'step1_read_data.py',
        'step2_create_tokens.py',
        'step3_discretize.py',
        'run_all.py',
        'example_usage.py'
    ]
    
    script_dir = Path(__file__).parent
    all_exist = True
    
    for script in scripts:
        script_path = script_dir / script
        if script_path.exists():
            print(f"  ✓ {script}")
        else:
            print(f"  ❌ {script} not found")
            all_exist = False
    
    return all_exist


def check_output_directory():
    """Check if output directory exists or can be created."""
    print("\nChecking output directory...")
    
    try:
        from omegaconf import OmegaConf
        config_path = Path(__file__).parent.parent.parent / "configs" / "data" / "prepare_mimic.yaml"
        cfg = OmegaConf.load(config_path)
        
        output_dir = Path(cfg.path_data)
        
        if output_dir.exists():
            print(f"  ✓ Output directory exists: {output_dir}")
        else:
            print(f"  ℹ Output directory will be created: {output_dir}")
            # Try to create it
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"  ✓ Successfully created output directory")
                return True
            except Exception as e:
                print(f"  ❌ Cannot create output directory: {e}")
                return False
        
        return True
    
    except Exception as e:
        print(f"  ⚠ Could not validate output directory: {e}")
        return None


def main():
    """Run all validation checks."""
    print("=" * 80)
    print("MIMIC-III Data Preparation - Setup Validation")
    print("=" * 80)
    
    checks = {
        "Python version": check_python_version(),
        "Dependencies": check_dependencies(),
        "Config file": check_config_file(),
        "Scripts": check_scripts(),
        "Output directory": check_output_directory(),
        "MIMIC-III paths": check_mimic_paths(),
    }
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    for check_name, result in checks.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠ WARNING"
        print(f"{status:12} {check_name}")
    
    # Determine overall status
    failed_checks = [name for name, result in checks.items() if result is False]
    warning_checks = [name for name, result in checks.items() if result is None]
    
    print("\n" + "=" * 80)
    
    if failed_checks:
        print("❌ VALIDATION FAILED")
        print("\nFailed checks:")
        for check in failed_checks:
            print(f"  - {check}")
        print("\nPlease fix the issues above before running the pipeline.")
        return 1
    elif warning_checks:
        print("⚠ VALIDATION PASSED WITH WARNINGS")
        print("\nWarning checks:")
        for check in warning_checks:
            print(f"  - {check}")
        print("\nYou may proceed, but please verify the warnings manually.")
        return 0
    else:
        print("✓ VALIDATION PASSED")
        print("\nAll checks passed! You can now run the pipeline:")
        print("  python run_all.py")
        return 0


if __name__ == "__main__":
    sys.exit(main())
