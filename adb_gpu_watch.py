import subprocess
import time
import threading
import argparse

def adb_shell(adb_path, device_id, cmd):
    try:
        result = subprocess.check_output(
            [adb_path, "-s", device_id, "shell"] + cmd.split(),
            stderr=subprocess.DEVNULL
        )
        return result.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        return "N/A"

def monitor_gpu(adb_path, device_id, interval):
    print(f"{'Time':<10} {'Clock Usage (%)':<18} {'GPU Busy (%)':<15}")
    print("-" * 45)
    start = time.time()
    while not stop_event.is_set():
        cur_freq = adb_shell(adb_path, device_id, "cat /sys/class/kgsl/kgsl-3d0/devfreq/cur_freq")
        max_freq = adb_shell(adb_path, device_id, "cat /sys/class/kgsl/kgsl-3d0/devfreq/max_freq")
        busy_pct = adb_shell(adb_path, device_id, "cat /sys/class/kgsl/kgsl-3d0/gpu_busy_percentage")

        try:
            cur_freq = int(cur_freq)
            max_freq = int(max_freq)
            clock_usage_pct = (cur_freq / max_freq) * 100 if max_freq else 0
        except ValueError:
            clock_usage_pct = 0
            busy_pct = "N/A"

        timestamp = f"{(time.time() - start):.1f}s"
        print(f"{timestamp:<10} {clock_usage_pct:>6.1f}%{'':<10} {busy_pct:<15}")
        time.sleep(interval)

def wait_for_exit():
    input("\nPress Enter to stop monitoring...\n")
    stop_event.set()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor GPU frequency and busy percentage over ADB.")
    parser.add_argument("-i", "--interval", type=float, default=1.0,
                        help="Polling interval in seconds (default: 1.0)")
    parser.add_argument("-a", "--adb-path", type=str, default="adb",
                        help="Path to adb executable (default: 'adb' in PATH)")
    parser.add_argument("-d", "--device", type=str, required=True,
                        help="ADB device ID (e.g., serial number from `adb devices`)")

    args = parser.parse_args()

    if args.interval <= 0:
        print("Interval must be greater than 0.")
        exit(1)

    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_gpu, args=(args.adb_path, args.device, args.interval))
    monitor_thread.start()

    try:
        wait_for_exit()
    except KeyboardInterrupt:
        print("\nStopping due to keyboard interrupt.")
        stop_event.set()

    monitor_thread.join()
    print("Monitoring stopped.")
