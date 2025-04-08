import subprocess
import argparse

def run_adb_command(command, adb_path="adb"):
    """Run an adb command and return the output."""
    full_command = f"{adb_path} {command}"
    result = subprocess.run(full_command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        exit(1)
    return result.stdout.strip()

def measure_battery_temp(device_id, adb_path):
    """Measure battery temperature."""
    command = f"-s {device_id} shell cat /sys/class/power_supply/battery/temp"
    temp = run_adb_command(command, adb_path)
    return int(temp) / 10  # Convert millidegree Celsius to Celsius

def main():
    parser = argparse.ArgumentParser(description="Run performance test on adb.")
    parser.add_argument("-d", "--device", required=True, help="ADB device ID")
    parser.add_argument("-m", "--model", required=True, help="Model name for the performance test")
    parser.add_argument("-e", "--execution_provider", choices=["cpu", "webgpu"], default="cpu", help="Execution provider")
    parser.add_argument("-w", "--workgroup", default="8,8,1", help="Workgroup size as X,Y,Z (default: 8,8,1)")
    parser.add_argument("-t", "--elements_per_thread", default="1,1,1", help="Elements per thread as X,Y,Z (default: 1,1,1)")
    parser.add_argument("-i", "--tile_inner", type=int, help="Value for ORT_WEBGPU_MATMUL_TILE_INNER (optional)")
    parser.add_argument("--adb_path", default="adb", help="Path to the adb executable (default: 'adb')")
    args = parser.parse_args()

    adb_path = args.adb_path
    device_id = args.device
    model_name = args.model
    execution_provider = args.execution_provider
    workgroup = args.workgroup.split(",")
    elements_per_thread = args.elements_per_thread.split(",")
    tile_inner = args.tile_inner

    # Start adb shell
    run_adb_command(f"-s {device_id} shell", adb_path)

    # Navigate to the directory
    run_adb_command(f"-s {device_id} shell cd /data/local/tmp/vraspar", adb_path)

    # Set environment variables
    run_adb_command(f"-s {device_id} shell export LD_LIBRARY_PATH=/data/local/tmp/vraspar", adb_path)
    if workgroup[0] and workgroup[1] and workgroup[2]:
        run_adb_command(f"-s {device_id} shell export ORT_MATMUL_WORKGROUP_X={workgroup[0]}", adb_path)
        run_adb_command(f"-s {device_id} shell export ORT_MATMUL_WORKGROUP_Y={workgroup[1]}", adb_path)
        run_adb_command(f"-s {device_id} shell export ORT_MATMUL_WORKGROUP_Z={workgroup[2]}", adb_path)
    if elements_per_thread[0] and elements_per_thread[1] and elements_per_thread[2]:
        run_adb_command(f"-s {device_id} shell export ORT_MATMUL_ELEMENTS_PER_THREAD_X={elements_per_thread[0]}", adb_path)
        run_adb_command(f"-s {device_id} shell export ORT_MATMUL_ELEMENTS_PER_THREAD_Y={elements_per_thread[1]}", adb_path)
        run_adb_command(f"-s {device_id} shell export ORT_MATMUL_ELEMENTS_PER_THREAD_Z={elements_per_thread[2]}", adb_path)
    if tile_inner is not None:
        run_adb_command(f"-s {device_id} shell export ORT_WEBGPU_MATMUL_TILE_INNER={tile_inner}", adb_path)

    # Measure battery temperature before the test
    temp_before = measure_battery_temp(device_id, adb_path)
    print(f"Battery temperature before test: {temp_before}°C")

    # Run the performance test
    run_adb_command(f"-s {device_id} shell ./onnxruntime_perf_test -I -r 20 -e {execution_provider} {model_name}", adb_path)

    # Measure battery temperature after the test
    temp_after = measure_battery_temp(device_id, adb_path)
    print(f"Battery temperature after test: {temp_after}°C")

if __name__ == "__main__":
    main()
