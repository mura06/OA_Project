import os
import zipfile
import sys

def package_project(group_number="99"):
    zip_name = f"Group_{group_number}.zip"
    print(f"Packaging project deliverables into {zip_name}...")
    
    # Files to include in the root of the folder inside zip
    root_files = [
        "run_experiments.py",
        "generate_report.py",
        "package_delivery.py",
        "parkinsons_preprocessed.csv",
        "Report.pdf"
    ]
    
    # Subdirectories to include
    subdirs = [
        "optimization_lib"
    ]
    
    # Check if Report.pdf exists
    if not os.path.exists("Report.pdf"):
        print("[WARNING] Report.pdf not found! Run generate_report.py first.")
        # Proceed anyway so the user can package if they printed manually
        
    folder_prefix = f"Group_{group_number}"
    
    try:
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add root files
            for f in root_files:
                if os.path.exists(f):
                    archive_path = os.path.join(folder_prefix, f)
                    zipf.write(f, archive_path)
                    print(f"Added {f} -> {archive_path}")
                else:
                    print(f"[ERROR] Required file {f} is missing!")
            
            # Add subdirectories recursively
            for subdir in subdirs:
                if os.path.exists(subdir):
                    for root, dirs, files in os.walk(subdir):
                        for file in files:
                            # Skip cached compiled files
                            if "__pycache__" in root or file.endswith(".pyc"):
                                continue
                            file_path = os.path.join(root, file)
                            archive_path = os.path.join(folder_prefix, file_path)
                            zipf.write(file_path, archive_path)
                            print(f"Added {file_path} -> {archive_path}")
                            
        print(f"\n[SUCCESS] Deliverable archive created: {zip_name}")
    except Exception as e:
        print(f"[ERROR] Failed to package project: {e}")

if __name__ == "__main__":
    group_num = "99"
    if len(sys.argv) > 1:
        group_num = sys.argv[1]
    package_project(group_num)
