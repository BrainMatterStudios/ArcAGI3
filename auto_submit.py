import os
import subprocess
import time
import re
from datetime import datetime, timezone

def wait_for_reset():
    # Wait until 00:05 UTC
    while True:
        now = datetime.now(timezone.utc)
        print(f"Current UTC time: {now.strftime('%H:%M:%S')}")
        if now.hour == 0 and now.minute >= 5:
            break
        # Sleep for 60 seconds
        time.sleep(60)

def main():
    print("Waiting for Kaggle quota reset (00:05 UTC)...")
    wait_for_reset()

    print("Quota reset reached! Pushing duck-effects kernel...")
    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", "submission/_duck_effects"],
        capture_output=True, text=True, check=True
    )
    print(result.stdout)
    
    # Parse version
    version_match = re.search(r"Kernel version (\d+) successfully pushed", result.stdout)
    if not version_match:
        print("Failed to parse kernel version from output. Cannot submit.")
        print(result.stdout)
        return
        
    version = version_match.group(1)
    print(f"Detected new kernel version: {version}")
    
    print("Sleeping 90 seconds to allow Kaggle to finish commit run before submission...")
    time.sleep(90)
    
    print("Submitting to competition...")
    submit_cmd = [
        "kaggle", "competitions", "submit",
        "arc-prize-2026-arc-agi-3",
        "-k", "ahmedmobasher86/arc-agi-3-duck-effects",
        "-v", version,
        "-f", "submission.parquet",
        "-m", "Duck-Effects (Base Model + Grounded Effect Memory & Click Productivity Grid)"
    ]
    submit_result = subprocess.run(submit_cmd, capture_output=True, text=True, check=True)
    print(submit_result.stdout)
    print("Auto-submit complete!")

if __name__ == "__main__":
    main()
