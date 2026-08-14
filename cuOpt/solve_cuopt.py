#!/usr/bin/env python3
"""
solve_cuopt.py

Runs the NVIDIA cuOpt GPU routing solver on every CVRPTW instance in a
testcase folder, once per requested time limit, and appends one row per
(instance, time limit) to a CSV file.

Instance files are expected in the standard Solomon VRPTW format:

    <instance name>

    VEHICLE
    NUMBER     CAPACITY
      <K>        <Q>

    CUSTOMER
    CUST NO.  XCOORD.  YCOORD.  DEMAND  READY TIME  DUE DATE  SERVICE TIME

       0      40         50          0          0       1236          0
       1      45         68         10        912        967         90
       ...

This is the same format the project's own testcase/ files use (the same
Solomon 1000-customer VRPTW set referenced in README.md), so no conversion
is needed.

Usage:
    python3 solve_cuopt.py \
        --testcase_dir testcase \
        --output_csv outputs_cuopt/cuopt_results.csv \
        --timeouts 60 10 5 2

Requires the cuopt Python package (and cudf) to be installed and a CUDA
GPU to be visible to the process. See submit_cuopt_job.sh for how this is
wired up on the cluster.
"""

import argparse
import csv
import glob
import math
import os
import sys
import time

# cuopt / cudf are only imported once we actually need them, so that
# --help and argument errors don't require a GPU environment.
def _import_cuopt():
    try:
        import cudf
        from cuopt import routing
    except ImportError as exc:
        sys.stderr.write(
            "ERROR: could not import cudf / cuopt. Make sure you are running "
            "inside an environment with cuopt installed and a GPU available.\n"
            f"Original error: {exc}\n"
        )
        sys.exit(1)
    return cudf, routing


# Solver status codes, per cuOpt's routing.Assignment.get_status():
#   0 - SUCCESS, 1 - FAIL, 2 - TIMEOUT, 3 - EMPTY
STATUS_NAMES = {0: "SUCCESS", 1: "FAIL", 2: "TIMEOUT", 3: "EMPTY"}


def parse_solomon_instance(filepath):
    """Parse a Solomon-format VRPTW instance file.

    Returns a dict with:
        coords:   list of (x, y) tuples, index 0 is the depot
        demand:   list of int demands, index 0 is the depot (0)
        ready:    list of float ready times
        due:      list of float due times
        service:  list of float service times
        n_vehicles: int, fleet size given in the file
        capacity:   int, per-vehicle capacity given in the file
    """
    with open(filepath, "r") as f:
        raw_lines = [line.rstrip("\n") for line in f]

    # Keep only non-blank lines for locating section headers, but remember
    # original content for the data rows.
    lines = [line.strip() for line in raw_lines if line.strip() != ""]

    if "VEHICLE" not in lines:
        raise ValueError("Could not find VEHICLE section")
    if "CUSTOMER" not in lines:
        raise ValueError("Could not find CUSTOMER section")

    veh_idx = lines.index("VEHICLE")
    # line after "VEHICLE" is the "NUMBER  CAPACITY" header,
    # the line after that holds the actual numbers
    veh_values = lines[veh_idx + 2].split()
    n_vehicles = int(veh_values[0])
    capacity = int(veh_values[1])

    cust_idx = lines.index("CUSTOMER")
    # line after "CUSTOMER" is the column header line; data starts after that
    data_start = cust_idx + 2

    coords, demand, ready, due, service = [], [], [], [], []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            _cust_no = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            dem = int(float(parts[3]))
            ready_t = float(parts[4])
            due_t = float(parts[5])
            serv_t = float(parts[6])
        except ValueError:
            # Not a data row (stray text, footer, etc.) - skip it.
            continue
        coords.append((x, y))
        demand.append(dem)
        ready.append(ready_t)
        due.append(due_t)
        service.append(serv_t)

    if len(coords) < 2:
        raise ValueError("Parsed fewer than 2 locations (depot + customers)")

    return {
        "coords": coords,
        "demand": demand,
        "ready": ready,
        "due": due,
        "service": service,
        "n_vehicles": n_vehicles,
        "capacity": capacity,
    }


def build_distance_matrix(coords):
    """Euclidean distance matrix as a list of lists (n x n)."""
    n = len(coords)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d = math.hypot(xi - xj, yi - yj)
            mat[i][j] = d
            mat[j][i] = d
    return mat


def solve_instance(cudf, routing, instance, time_limit):
    """Solve one parsed instance with cuOpt at the given time limit (seconds).

    Returns a dict with status, status_name, total_cost, vehicles_used,
    solve_time_s. total_cost / vehicles_used are None if no solution was
    produced.
    """
    coords = instance["coords"]
    n_locations = len(coords)
    n_vehicles = instance["n_vehicles"]
    capacity = instance["capacity"]

    dist_matrix = build_distance_matrix(coords)
    cost_matrix = cudf.DataFrame(dist_matrix, dtype="float32")

    data_model = routing.DataModel(n_locations, n_vehicles)
    data_model.add_cost_matrix(cost_matrix)
    data_model.add_transit_time_matrix(cost_matrix.copy(deep=True))

    demand_series = cudf.Series(instance["demand"])
    capacity_series = cudf.Series([capacity] * n_vehicles)
    data_model.add_capacity_dimension("demand", demand_series, capacity_series)

    earliest = cudf.Series(instance["ready"])
    latest = cudf.Series(instance["due"])
    data_model.set_order_time_windows(earliest, latest)
    data_model.set_order_service_times(cudf.Series(instance["service"]))

    depot_ready = instance["ready"][0]
    depot_due = instance["due"][0]
    vehicle_earliest = cudf.Series([depot_ready] * n_vehicles)
    vehicle_latest = cudf.Series([depot_due] * n_vehicles)
    data_model.set_vehicle_time_windows(vehicle_earliest, vehicle_latest)

    solver_settings = routing.SolverSettings()
    solver_settings.set_time_limit(float(time_limit))

    t0 = time.perf_counter()
    solution = routing.Solve(data_model, solver_settings)
    elapsed = time.perf_counter() - t0

    status = solution.get_status()
    status_name = STATUS_NAMES.get(status, f"UNKNOWN({status})")

    total_cost = None
    vehicles_used = None
    if status in (0, 2):  # SUCCESS or TIMEOUT may still carry a solution
        try:
            total_cost = float(solution.get_total_objective())
            vehicles_used = int(solution.get_vehicle_count())
        except Exception:
            pass

    return {
        "status": status,
        "status_name": status_name,
        "total_cost": total_cost,
        "vehicles_used": vehicles_used,
        "solve_time_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--testcase_dir", default="testcase",
        help="Directory containing Solomon-format instance files (default: testcase)",
    )
    parser.add_argument(
        "--output_csv", default="outputs_cuopt/cuopt_results.csv",
        help="CSV file to append results to (default: outputs_cuopt/cuopt_results.csv)",
    )
    parser.add_argument(
        "--timeouts", type=float, nargs="+", default=[60, 10, 5, 2],
        help="Time limits in seconds to run for each instance (default: 60 10 5 2)",
    )
    args = parser.parse_args()

    cudf, routing = _import_cuopt()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    write_header = not os.path.exists(args.output_csv)

    files = sorted(
        p for p in glob.glob(os.path.join(args.testcase_dir, "*"))
        if os.path.isfile(p)
    )
    if not files:
        sys.stderr.write(f"No files found in {args.testcase_dir}\n")
        sys.exit(1)

    fieldnames = [
        "instance", "n_customers", "n_vehicles_available", "capacity",
        "timeout_s", "status", "status_name", "total_cost",
        "vehicles_used", "solve_time_s",
    ]

    with open(args.output_csv, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for filepath in files:
            instance_name = os.path.basename(filepath)
            try:
                instance = parse_solomon_instance(filepath)
            except Exception as exc:
                print(f"[SKIP] {instance_name}: failed to parse ({exc})")
                continue

            n_customers = len(instance["coords"]) - 1
            print(f"=== {instance_name} "
                  f"({n_customers} customers, {instance['n_vehicles']} vehicles, "
                  f"capacity {instance['capacity']}) ===")

            for timeout in args.timeouts:
                row = {
                    "instance": instance_name,
                    "n_customers": n_customers,
                    "n_vehicles_available": instance["n_vehicles"],
                    "capacity": instance["capacity"],
                    "timeout_s": timeout,
                }
                try:
                    result = solve_instance(cudf, routing, instance, timeout)
                    row.update(result)
                    print(
                        f"  timeout={timeout:>5}s  status={result['status_name']:<8} "
                        f"cost={result['total_cost']}  vehicles={result['vehicles_used']}  "
                        f"wall_time={result['solve_time_s']:.2f}s"
                    )
                except Exception as exc:
                    row.update({
                        "status": "", "status_name": "ERROR", "total_cost": "",
                        "vehicles_used": "", "solve_time_s": "",
                    })
                    print(f"  timeout={timeout:>5}s  ERROR: {exc}")

                writer.writerow(row)
                csvfile.flush()

    print(f"\nDone. Results written to {args.output_csv}")


if __name__ == "__main__":
    main()
