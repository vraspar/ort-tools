import subprocess
import argparse

def run_adb_command(command, adb_path="adb"):
    """Run an adb command and return the output."""
    print(f"Running command: {adb_path} {command}")
    full_command = f"{adb_path} {command}"
    result = subprocess.run(full_command, shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print(f"Command stdout: {result.stdout.strip()}")
    if result.stderr.strip():
        print(f"Command stderr: {result.stderr.strip()}")
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        exit(1)
    return result.stdout.strip()

def measure_battery_temp(device_id, adb_path):
    """Measure battery temperature."""
    command = f"-s {device_id} shell dumpsys battery | grep temperature"
    output = run_adb_command(command, adb_path)
    # Extract the temperature value from the output
    try:
        temp_line = output.strip()
        temp = int(temp_line.split(":")[1].strip())  # Extract the numeric value after "temperature:"
        return temp / 10  # Convert from tenths of a degree Celsius to Celsius
    except (IndexError, ValueError):
        print("Error: Failed to parse battery temperature.")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run performance test on adb.")
    parser.add_argument("-d", "--device", required=True, help="ADB device ID")
    parser.add_argument("-m", "--model", required=True, help="Model name for the performance test")
    parser.add_argument("-e", "--execution_provider", choices=["cpu", "webgpu"], default="cpu", help="Execution provider")
    parser.add_argument("-w", "--workgroup", help="Workgroup size as X,Y,Z (optional)")
    parser.add_argument("-t", "--elements_per_thread", help="Elements per thread as X,Y,Z (optional)")
    parser.add_argument("-i", "--tile_inner", type=int, help="Value for ORT_WEBGPU_MATMUL_TILE_INNER (optional)")
    parser.add_argument("--adb_path", default="adb", help="Path to the adb executable (default: 'adb')")
    args = parser.parse_args()

    adb_path = args.adb_path
    device_id = args.device
    model_name = args.model
    execution_provider = args.execution_provider
    workgroup = args.workgroup.split(",") if args.workgroup else None
    elements_per_thread = args.elements_per_thread.split(",") if args.elements_per_thread else None
    tile_inner = args.tile_inner

    commands = [
        "cd /data/local/tmp/vraspar",
        "export LD_LIBRARY_PATH=/data/local/tmp/vraspar"
    ]
    if workgroup:
        commands.append(f"export ORT_MATMUL_WORKGROUP_X={workgroup[0]}")
        commands.append(f"export ORT_MATMUL_WORKGROUP_Y={workgroup[1]}")
        commands.append(f"export ORT_MATMUL_WORKGROUP_Z={workgroup[2]}")
    if elements_per_thread:
        commands.append(f"export ORT_MATMUL_ELEMENTS_PER_THREAD_X={elements_per_thread[0]}")
        commands.append(f"export ORT_MATMUL_ELEMENTS_PER_THREAD_Y={elements_per_thread[1]}")
        commands.append(f"export ORT_MATMUL_ELEMENTS_PER_THREAD_Z={elements_per_thread[2]}")
    if tile_inner is not None:
        commands.append(f"export ORT_WEBGPU_MATMUL_TILE_INNER={tile_inner}")
    commands.append(f"./onnxruntime_perf_test -I -r 20 -e {execution_provider} {model_name}")

    full_command = " && ".join(commands)

    temp_before = measure_battery_temp(device_id, adb_path)
    print(f"Battery temperature before test: {temp_before}°C")

    print(f"Executing onnxruntime_perf_test with model {model_name}...")
    run_adb_command(f"-s {device_id} shell \"{full_command}\"", adb_path)

    temp_after = measure_battery_temp(device_id, adb_path)
    print(f"Battery temperature after test: {temp_after}°C")

if __name__ == "__main__":
    main()
