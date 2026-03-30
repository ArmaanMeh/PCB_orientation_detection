"""
Quick Start Guide for HOG+SVM PCB Orientation Detection
Simple script to run all steps in sequence - OPTIMIZED
"""

import os
import sys
import subprocess
import gc
from pathlib import Path

# Memory optimization
gc.enable()

def print_header(text):
    """Print formatted header - optimized."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_section(text):
    """Print formatted section."""
    print(f"\n{text}")
    print("-" * 70)

def check_requirements():
    """Check if all required packages are installed - optimized."""
    print_header("CHECKING REQUIREMENTS")
    
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'tensorflow': 'tensorflow',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'tqdm': 'tqdm'
    }
    
    missing = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print("\nInstalling Missing Packages")
        print("-" * 70)
        print("\nRun the following command to install missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All requirements installed")
    return True


def check_data():
    """Check if data directory exists and contains images - optimized."""
    print_header("CHECKING DATA")
    
    data_dirs = {
        'Data/Processed_data/Pass_data': 'Pass training data',
        'Data/Processed_data/Fail_data': 'Fail training data'
    }
    
    all_exist = True
    
    for path, description in data_dirs.items():
        path_obj = Path(path)
        if path_obj.exists():
            # Fast count using glob instead of os.listdir
            img_count = sum(1 for _ in path_obj.glob('*.jpg')) + \
                       sum(1 for _ in path_obj.glob('*.jpeg')) + \
                       sum(1 for _ in path_obj.glob('*.png')) + \
                       sum(1 for _ in path_obj.glob('*.PNG')) + \
                       sum(1 for _ in path_obj.glob('*.JPG'))
            print(f"  ✓ {description}: {img_count} images")
        else:
            print(f"  ✗ {description}: MISSING")
            all_exist = False
    
    if not all_exist:
        print("\nSetting up Data")
        print("-" * 70)
        print("\nPlease ensure your data is organized as:")
        print("  Data/Processed_data/")
        print("    ├── Pass_data/    (images with correct orientation)")
        print("    └── Fail_data/    (images with incorrect orientation)")
        return False
    
    print("\n✓ Data structure is correct")
    return True


def print_menu():
    """Print main menu."""
    print_header("QUICK START MENU - HOG+SVM MODEL")
    
    print("""
  1. Train HOG+SVM Model
     - Complete training pipeline with hyperparameter optimization
     - Generates confusion matrices and ROC curves
     - Saves trained model for later use
  
  2. Live Classification (HOG+SVM)
     - Real-time video classification using trained model
     - Webcam input with predictions overlay
     - Press 'q' to quit, 's' to save frame
  
  3. Compare CNN vs HOG+SVM
     - Compare performance of both models
     - Generate side-by-side comparisons
     - Visualize performance differences
  
  4. Model Utilities & Statistics
     - Interactive menu for model analysis
     - Dataset statistics
     - Inference speed testing
     - Performance reports
  
  5. Live Classification (CNN)
     - Real-time classification using pre-trained CNN
     - Webcam input with predictions overlay
  
  0. Exit
    """)
    
    return input("Select option (0-5): ").strip()


def run_training():
    """Run model training - optimized."""
    print("\n" + "-" * 70)
    print("Starting HOG+SVM Training")
    print("-" * 70)
    print("This may take several minutes depending on dataset size...")
    print("Training will:")
    print("  • Load all images from Data/Processed_data/")
    print("  • Extract HOG features")
    print("  • Perform hyperparameter optimization")
    print("  • Evaluate on train/validation/test sets")
    
    input("\nPress Enter to continue...")
    
    try:
        # Use subprocess instead of os.system for better control
        result = subprocess.call([sys.executable, "hog_svm_train.py"])
        if result == 0:
            print("\n" + "="*70)
            print("  ✓ Training completed successfully!")
            print("  Model saved to: Export/hog_svm_model.pkl")
            print("="*70)
            return True
        else:
            print(f"\n✗ Training failed with code {result}")
            return False
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        return False
    finally:
        gc.collect()  # Force garbage collection after heavy operation


def run_live_hog_svm():
    """Run live HOG+SVM classification - optimized."""
    print("\n" + "-" * 70)
    print("Starting Live HOG+SVM Classification")
    print("-" * 70)
    print("Controls:")
    print("  • 'q' - Quit application")
    print("  • 's' - Save current frame")
    print("\nMake sure your webcam is connected and available.")
    
    input("\nPress Enter to start webcam...")
    
    try:
        subprocess.call([sys.executable, "hog_svm_live.py"])
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        gc.collect()


def run_live_cnn():
    """Run live CNN classification - optimized."""
    print("\n" + "-" * 70)
    print("Starting Live CNN Classification")
    print("-" * 70)
    print("Controls:")
    print("  • 'q' - Quit application")
    
    input("\nPress Enter to start webcam...")
    
    try:
        subprocess.call([sys.executable, "live_classification.py"])
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        gc.collect()


def run_comparison():
    """Run model comparison - optimized."""
    print("\n" + "-" * 70)
    print("Starting Model Comparison (CNN vs HOG+SVM)")
    print("-" * 70)
    print("This will:")
    print("  • Load test data")
    print("  • Generate predictions from both models")
    print("  • Compare performance metrics")
    
    input("\nPress Enter to continue...")
    
    try:
        subprocess.call([sys.executable, "compare_models.py"])
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        gc.collect()


def run_utilities():
    """Run model utilities - optimized."""
    print("\n" + "-" * 70)
    print("Starting Model Utilities")
    print("-" * 70)
    
    try:
        subprocess.call([sys.executable, "model_utils.py"])
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        gc.collect()


def print_menu():
    """Print main menu - optimized."""
    print_header("QUICK START MENU - HOG+SVM MODEL")
    
    print("""
  1. Train HOG+SVM Model
     - Complete training pipeline with hyperparameter optimization
     - Generates confusion matrices and ROC curves
  
  2. Live Classification (HOG+SVM)
     - Real-time video classification using trained model
     - Webcam input with predictions overlay
  
  3. Compare CNN vs HOG+SVM
     - Compare performance of both models
  
  4. Model Utilities & Statistics
     - Interactive menu for model analysis
  
  5. Live Classification (CNN)
     - Real-time classification using pre-trained CNN
  
  0. Exit
    """)
    
    return input("Select option (0-5): ").strip()


def main():
    """Main quick-start application - optimized."""
    print_header("PCB ORIENTATION DETECTION - QUICK START")
    
    # Check requirements
    if not check_requirements():
        print("\nPlease install missing packages and try again.")
        input("Press Enter to exit...")
        return
    
    # Check data
    if not check_data():
        print("\nPlease set up your data and try again.")
        input("Press Enter to exit...")
        return
    
    # Main loop
    while True:
        choice = print_menu()
        
        if choice == '1':
            run_training()
        elif choice == '2':
            run_live_hog_svm()
        elif choice == '3':
            run_comparison()
        elif choice == '4':
            run_utilities()
        elif choice == '5':
            run_live_cnn()
        elif choice == '0':
            print("\n" + "="*70)
            print("  Thank you for using PCB Orientation Detection!")
            print("="*70 + "\n")
            break
        else:
            print("\n✗ Invalid option. Please select 0-5.")
        
        # Ask if continue
        if choice != '0':
            input("\nPress Enter to return to menu...")


def run_single(script_name):
    """Run a single script directly - optimized."""
    scripts = {
        'train': 'hog_svm_train.py',
        'live_hog': 'hog_svm_live.py',
        'live_cnn': 'live_classification.py',
        'compare': 'compare_models.py',
        'utils': 'model_utils.py'
    }
    
    if script_name in scripts:
        print_header(f"Running {script_name}")
        try:
            subprocess.call([sys.executable, scripts[script_name]])
        except Exception as e:
            print(f"Error: {e}")
        finally:
            gc.collect()
    else:
        print(f"Unknown script: {script_name}")
        print(f"Available: {', '.join(scripts.keys())}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line mode
        run_single(sys.argv[1])
    else:
        # Interactive mode
        main()
