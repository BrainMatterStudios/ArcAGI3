import subprocess
import time
import sys

def main():
    while True:
        status_res = subprocess.run(["kaggle", "kernels", "status", "ahmedmobasher86/arc-agi-3-duck-sft"], capture_output=True, text=True)
        print(f"Status: {status_res.stdout.strip()}")
        if "COMPLETE" in status_res.stdout:
            print("Kernel completed! Submitting version 1 to competition...")
            submit_cmd = [
                "kaggle", "competitions", "submit",
                "arc-prize-2026-arc-agi-3",
                "-k", "ahmedmobasher86/arc-agi-3-duck-sft",
                "-v", "1",
                "-f", "submission.parquet",
                "-m", "Duck-SFT (Run-8 Fine-Tuned Adapter with In-Kernel Merge & Serving Assertions)"
            ]
            result = subprocess.run(submit_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("Submitted successfully!")
                print(result.stdout)
                break
            else:
                print(f"Submission failed: {result.stderr} \n {result.stdout}")
                break
        elif "ERROR" in status_res.stdout:
            print("Kernel errored! Cannot submit.")
            break
        print("Kernel is still running... waiting 30 seconds.")
        time.sleep(30)

if __name__ == "__main__":
    main()
