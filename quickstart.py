"""
Quick Start Guide for HOG+SVM PCB Orientation Detection
Simple script to run all steps in sequence
"""

import os
import sys
import time

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_section(text):
    """Print formatted section."""
    print(f"\n{text}")
    print("-" * 70)

def check_requirements():
    """Check if all required packages are installed."""
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
        print_section("Installing Missing Packages")
        print("\nRun the following command to install missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All requirements installed")
    return True


def check_data():
    """Check if data directory exists and contains images."""
    print_header("CHECKING DATA")
    
    data_dirs = {
        'Data/Processed_data/Pass_data': 'Pass training data',
        'Data/Processed_data/Fail_data': 'Fail training data'
    }
    
    all_exist = True
    
    for path, description in data_dirs.items():
        if os.path.exists(path):
            img_count = len([f for f in os.listdir(path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  ✓ {description}: {img_count} images")
        else:
            print(f"  ✗ {description}: MISSING")
            all_exist = False
    
    if not all_exist:
        print_section("Setting up Data")
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
    """Run model training."""
    print_section("Starting HOG+SVM Training")
    print("This may take several minutes depending on dataset size...")
    print("Training will:")
    print("  • Load all images from Data/Processed_data/")
    print("  • Extract HOG features")
    print("  • Perform hyperparameter optimization")
    print("  • Evaluate on train/validation/test sets")
    print("  • Perform 5-fold cross-validation")
    print("  • Generate visualizations")
    print("  • Save trained model to Export/")
    
    input("\nPress Enter to continue...")
    
    try:
        os.system("python hog_svm_train.py")
        print("\n" + "="*70)
        print("  ✓ Training completed successfully!")
        print("  Model saved to: Export/hog_svm_model.pkl")
        print("  Scaler saved to: Export/hog_svm_scaler.pkl")
        print("="*70)
        return True
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        return False


def run_live_hog_svm():
    """Run live HOG+SVM classification."""
    print_section("Starting Live HOG+SVM Classification")
    print("Controls:")
    print("  • 'q' - Quit application")
    print("  • 's' - Save current frame")
    print("  • 'r' - Reset FPS counter")
    print("\nMake sure your webcam is connected and available.")
    
    input("\nPress Enter to start webcam...")
    
    try:
        os.system("python hog_svm_live.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def run_live_cnn():
    """Run live CNN classification."""
    print_section("Starting Live CNN Classification")
    print("Controls:")
    print("  • 'q' - Quit application")
    
    input("\nPress Enter to start webcam...")
    
    try:
        os.system("python live_classification.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def run_comparison():
    """Run model comparison."""
    print_section("Starting Model Comparison (CNN vs HOG+SVM)")
    print("This will:")
    print("  • Load test data")
    print("  • Generate predictions from both models")
    print("  • Compare performance metrics")
    print("  • Generate visualization comparisons")
    
    input("\nPress Enter to continue...")
    
    try:
        os.system("python compare_models.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def run_utilities():
    """Run model utilities."""
    print_section("Starting Model Utilities")
    
    try:
        os.system("python model_utils.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def print_next_steps():
    """Print next steps guide."""
    print_header("NEXT STEPS")
    
    print("""
  After selecting an option:
  
  AFTER TRAINING:
  • Your model is saved in Export/hog_svm_model.pkl
  • Check confusion matrices in Export/
  • Review performance metrics in console output
  • Run Live Classification to test on webcam
  
  AFTER LIVE CLASSIFICATION:
  • Frames are saved in Export/ (if you pressed 's')
  • Review predictions and confidence scores
  • Check FPS for performance
  
  FOR FURTHER ANALYSIS:
  • Use Model Utilities for detailed statistics
  • Use Model Comparison to see CNN vs HOG+SVM
  • Check HOG_SVM_README.md for advanced usage
  
  TIPS:
  • The first training run may take longer (feature extraction)
  • Live classification works best with adequate lighting
  • GPU acceleration (if available) will speed up CNN
  • HOG+SVM is faster but may need more CPU for feature extraction
    """)


def main():
    """Main quick-start application."""
    print("\n" * 2)
    print_header("PCB ORIENTATION DETECTION - QUICK START")
    
    # Check requirements
    if not check_requirements():
        print("\nPlease install missing packages and try again.")
        input("Press Enter to exit...")
        return
    
    time.sleep(1)
    
    # Check data
    if not check_data():
        print("\nPlease set up your data and try again.")
        input("Press Enter to exit...")
        return
    
    time.sleep(1)
    
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
            time.sleep(1)
        
        # Ask if continue
        if choice != '0':
            input("\nPress Enter to return to menu...")


def run_single(script_name):
    """Run a single script directly."""
    """
    Usage: python quickstart.py [train|live_hog|live_cnn|compare|utils]
    """
    scripts = {
        'train': 'hog_svm_train.py',
        'live_hog': 'hog_svm_live.py',
        'live_cnn': 'live_classification.py',
        'compare': 'compare_models.py',
        'utils': 'model_utils.py'
    }
    
    if script_name in scripts:
        print_header(f"Running {script_name}")
        os.system(f"python {scripts[script_name]}")
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
