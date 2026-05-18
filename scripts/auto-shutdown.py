#!/usr/bin/env python3
import subprocess
import time
import sys

def get_gpu_utilizations():
    """Get current GPU utilization percentages for all visible GPUs."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True,
        check=True
    )
    utilizations = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            utilizations.append(int(line))

    if not utilizations:
        raise RuntimeError('nvidia-smi returned no GPU utilization data')

    return utilizations

def auto_shutdown(idle_minutes=30, threshold=5):
    """Shutdown instance if all GPUs are idle for the specified minutes."""
    idle_count = 0
    check_interval = 60  # Check every minute

    while True:
        utilizations = get_gpu_utilizations()
        util_summary = ', '.join(f'GPU {index}: {util}%' for index, util in enumerate(utilizations))
        # All GPUs' utilization must be below the threshold to 
        # count as idle
        if all(util < threshold for util in utilizations):
            idle_count += 1
            print(f"All GPUs idle ({util_summary}). Idle count: {idle_count}/{idle_minutes}")

            if idle_count >= idle_minutes:
                print("Shutting down due to inactivity...")
                subprocess.run(['sudo', 'shutdown', '-h', 'now'])
                sys.exit(0)
        else:
            idle_count = 0
            print(f"GPU activity detected ({util_summary}). Reset idle counter.")

        time.sleep(check_interval)

if __name__ == "__main__":
    auto_shutdown(idle_minutes=15, threshold=5)
    