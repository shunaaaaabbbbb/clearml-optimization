import sys
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from clearml import Task, Logger
from pyscipopt import Model, quicksum, SCIP_PARAMSETTING, SCIP_STATUS, Eventhdlr, SCIP_EVENTTYPE

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VRP_Optimizer")

# -------------------------------------------------------------------------
# 1. Pydantic Configuration & Validation
# -------------------------------------------------------------------------

class VRPConfig(BaseModel):
    """Configuration for VRP Optimization Experiment."""
    num_customers: int = Field(default=5, ge=3, description="Number of customers")
    vehicle_capacity: float = Field(default=100.0, gt=0, description="Capacity of each vehicle")
    num_vehicles: int = Field(default=2, ge=1, description="Number of vehicles available")
    
    # Random seed for reproducibility
    seed: int = Field(default=42, description="Random seed")
    
    # Optimization weights
    weight_distance: float = Field(default=1.0, ge=0)
    weight_time: float = Field(default=0.0, ge=0)
    
    # Problem domain parameters (for generation)
    max_demand: float = Field(default=20.0, gt=0)
    area_size: float = Field(default=100.0, gt=0)

    @model_validator(mode='after')
    def check_total_capacity(self):
        # Heuristic check: Total capacity vs Expected Total Demand
        # Expected total demand approx: num_customers * (max_demand / 2)
        # We enforce a stricter check if we had concrete demands, but here we validate config logic.
        # Let's enforce that max_demand <= vehicle_capacity to ensure at least one customer fits.
        if self.max_demand > self.vehicle_capacity:
            raise ValueError(f"Max demand ({self.max_demand}) cannot exceed vehicle capacity ({self.vehicle_capacity})")
        return self

# -------------------------------------------------------------------------
# 2. Data Generation & Structures
# -------------------------------------------------------------------------

class Customer:
    def __init__(self, id: int, x: float, y: float, demand: float):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand

def generate_data(config: VRPConfig) -> Tuple[Customer, List[Customer]]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # Depot at (50, 50) roughly center
    depot = Customer(0, config.area_size/2, config.area_size/2, 0)
    
    customers = []
    total_demand = 0
    for i in range(1, config.num_customers + 1):
        x = random.uniform(0, config.area_size)
        y = random.uniform(0, config.area_size)
        demand = random.uniform(1, config.max_demand)
        customers.append(Customer(i, x, y, demand))
        total_demand += demand
        
    # Validation: Logic to fail Validatoin if Total Demand > Total Capacity
    total_capacity = config.num_vehicles * config.vehicle_capacity
    if total_demand > total_capacity:
        logger.error(f"Validation Error: Total Demand ({total_demand:.2f}) exceeds Total Fleet Capacity ({total_capacity:.2f})")
        # In a real scenario, we might want to raise an error to stop execution
        # The user requested: "ClearML Task to Failed"
        # We will handle this in the main block
        raise ValueError(f"Total Demand ({total_demand:.2f}) > Total Capacity ({total_capacity:.2f})")

    return depot, customers

def dist(c1: Customer, c2: Customer) -> float:
    return np.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)

# -------------------------------------------------------------------------
# 3. ClearML Custom Event Handler for PySCIPOpt
# -------------------------------------------------------------------------

class ClearMLEventHandler(Eventhdlr):
    def __init__(self, clearml_logger: Logger):
        self.clearml_logger = clearml_logger
        self.iter_count = 0

    def eventexec(self, event):
        try:
            # Safely attempt to get bounds/gap
            primal = self.model.getPrimalbound()
            dual = self.model.getDualbound()
            gap = self.model.getGap()
            
            # SCIP might return infinity for bounds initially
            if abs(primal) < 1e15:
                self.clearml_logger.report_scalar("Optimization", "Primal Bound", primal, self.iter_count)
            if abs(dual) < 1e15:
                self.clearml_logger.report_scalar("Optimization", "Dual Bound", dual, self.iter_count)
            
            self.clearml_logger.report_scalar("Optimization", "Gap", gap * 100, self.iter_count) # Gap in %
            self.iter_count += 1
        except Exception as e:
            pass
            # logger.warning(f"Event Handler Error: {e}")

# -------------------------------------------------------------------------
# 4. Solver Logic (PySCIPOpt)
# -------------------------------------------------------------------------

def solve_vrp(config: VRPConfig, depot: Customer, customers: List[Customer], clearml_task: Task):
    # Initialize Model
    model = Model("CVRP_MTZ")
    
    # ------------------
    # Data Preparation
    # ------------------
    all_nodes = [depot] + customers
    N = len(all_nodes)
    customers_indices = [c.id for c in customers]
    all_indices = [n.id for n in all_nodes]
    
    # Distance Matrix
    c = {}
    for i in all_indices:
        for j in all_indices:
            if i != j:
                # Cost function: w1 * dist + w2 * time
                # Modeling time as proportional to dist for simplicity
                d = dist(all_nodes[i], all_nodes[j])
                cost = config.weight_distance * d + config.weight_time * (d / 1.0) # Assume speed=1
                c[i,j] = cost
    
    # ------------------
    # Variables
    # ------------------
    # x[i,j] = 1 if vehicle goes from i to j
    x = {}
    for i in all_indices:
        for j in all_indices:
            if i != j:
                x[i,j] = model.addVar(vtype="B", name=f"x_{i}_{j}")
    
    # u[i] = cumulative load (or position) at node i for MTZ
    # u[i] represents the load of the vehicle AFTER visiting customer i
    u = {}
    for i in customers_indices:
        u[i] = model.addVar(lb=all_nodes[i].demand, ub=config.vehicle_capacity, vtype="C", name=f"u_{i}")

    # ------------------
    # Objective
    # ------------------
    model.setObjective(quicksum(c[i,j] * x[i,j] for i in all_indices for j in all_indices if i != j), "minimize")

    # ------------------
    # Constraints
    # ------------------
    
    # 1. Each customer is visited exactly once
    for i in customers_indices:
        model.addCons(quicksum(x[i,j] for j in all_indices if i != j) == 1, name=f"leave_{i}")
        model.addCons(quicksum(x[j,i] for j in all_indices if i != j) == 1, name=f"enter_{i}")
        
    # 2. Flow conservation at depot
    # Outgoing = Number of Vehicles (or <= Num Vehicles if fleet is upper bound)
    # Let's assume exactly Num Vehicles used, or at most. 
    # Standard VRP usually minimizes cost, so we allow <= K vehicles.
    # But usually sum(x[0,j]) == sum(x[j,0])
    model.addCons(quicksum(x[0,j] for j in customers_indices) <= config.num_vehicles, name="depot_out")
    model.addCons(quicksum(x[j,0] for j in customers_indices) <= config.num_vehicles, name="depot_in")
    model.addCons(quicksum(x[0,j] for j in customers_indices) == quicksum(x[j,0] for j in customers_indices), name="depot_flow")

    # 3. MTZ Subtour Elimination & Capacity Constraints
    for i in customers_indices:
        for j in customers_indices:
            if i != j:
                # u_j >= u_i + d_j - Q(1 - x_ij)
                # u_i - u_j + Q*x_ij <= Q - d_j
                demand_j = all_nodes[j].demand
                capacity = config.vehicle_capacity
                model.addCons(u[i] - u[j] + capacity * x[i,j] <= capacity - demand_j, name=f"mtz_{i}_{j}")
    
    # ------------------
    # Logging Setup
    # ------------------
    # Attach Event Handler for real-time logging
    eventhdlr = ClearMLEventHandler(clearml_task.logger)
    model.includeEventhdlr(eventhdlr, "ClearMLEventHdlr", "Logs gap and primal bound to ClearML")
    
    # ------------------
    # Optimization
    # ------------------
    logger.info("Starting Optimization...")
    model.optimize()
    
    status = model.getStatus()
    logger.info(f"Optimization finished with status: {status}")
    
    if status == "infeasible":
        logger.error("Model is Infeasible! Computing IIS...")
        # PySCIPOpt IIS handling
        # Note: PySCIPOpt might not print IIS to strings effectively in all versions, 
        # but we can try to rely on its internal print or use an alternative if needed.
        # model.computeIIS() usually prints to stdout. We can capture it.
        
        # NOTE: Not all PySCIPOpt builds include IIS support (requires LPS check).
        # Assuming it works.
        try:
            model.computeIIS()
            # Since computeIIS prints to stdout, we can check if we can redirect or just let it print
            # and the user sees it in Console. 
            # To upload to ClearML artifact, we might need to capture stdout.
            # For now, we will assume console log is captured by ClearML automatically.
        except Exception as e:
            logger.error(f"Failed to compute IIS: {e}")
            
        return None, None

    if model.getSols():
        sol_routes = []
        # Extract solution
        # Iterate over all x variables to find active edges
        active_edges = []
        for key, var in x.items():
            if model.getVal(var) > 0.5:
                active_edges.append(key)
        
        # Build routes from edges
        # Start from depot (0)
        # Find all edges starting at 0
        starts = [j for (i, j) in active_edges if i == 0]
        
        routes = []
        for s in starts:
            route = [0, s]
            curr = s
            while curr != 0:
                # Find next node
                found = False
                for (i, j) in active_edges:
                    if i == curr:
                        route.append(j)
                        curr = j
                        found = True
                        break
                if not found:
                    break # Should not happen in valid flow
            routes.append(route)
            
        return routes, active_edges
    
    return None, None

def plot_routes(depot: Customer, customers: List[Customer], routes: List[List[int]]):
    plt.figure(figsize=(10, 8))
    
    # Plot Depot
    plt.scatter(depot.x, depot.y, c='red', marker='s', s=100, label='Depot')
    
    # Plot Customers
    for c in customers:
        plt.scatter(c.x, c.y, c='blue', s=50)
        plt.text(c.x+1, c.y+1, str(c.id), fontsize=9)
        
    # Plot Routes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
    for route, color in zip(routes, colors):
        # Gather coords
        all_nodes = {0: depot}
        for c in customers:
            all_nodes[c.id] = c
            
        path_x = [all_nodes[idx].x for idx in route]
        path_y = [all_nodes[idx].y for idx in route]
        
        plt.plot(path_x, path_y, c=color, linewidth=2, label=f'Route {routes.index(route)}')
        
        # Arrows
        for k in range(len(route)-1):
            p1 = all_nodes[route[k]]
            p2 = all_nodes[route[k+1]]
            mid_x = (p1.x + p2.x) / 2
            mid_y = (p1.y + p2.y) / 2
            dx = (p2.x - p1.x)
            dy = (p2.y - p1.y)
            plt.arrow(mid_x, mid_y, dx*0.1, dy*0.1, head_width=2, color=color)

    plt.title("VRP Optimization Result")
    plt.legend()
    plt.grid(True)
    
    # Save to file
    plt.savefig("vrp_solution.png")
    
    # Close
    plt.close()

# -------------------------------------------------------------------------
# 5. Main Execution
# -------------------------------------------------------------------------

def main():
    # 1. Initialize ClearML Task
    task = Task.init(project_name="VRP_Optimization", task_name="VRP_MTZ_Experiment")
    
    # 2. Config & Pydantic
    # Load defaults first
    config_dict = {
        "num_customers": 10,
        "vehicle_capacity": 50,
        "num_vehicles": 4,
        "weight_distance": 1.0,
        "weight_time": 0.5,
        "seed": 42,
        "max_demand": 10
    }
    
    # Connect to ClearML to allow overriding
    # We pass the dictionary to task.connect to make it editable in UI
    # Then we load it back into Pydantic to validate
    params = task.connect(config_dict)
    
    try:
        config = VRPConfig(**params)
    except Exception as e:
        logger.error(f"Configuration Error: {e}")
        task.mark_failed(status_reason=f"Config Error: {e}")
        sys.exit(1)
        
    logger.info(f"Configuration Loaded: {config}")

    # 3. Generate Data
    try:
        depot, customers = generate_data(config)
    except ValueError as e:
        logger.error(f"Data Generation/Validation Failed: {e}")
        task.mark_failed(status_reason=str(e))
        sys.exit(1)

    # 4. Optimization
    routes, edges = solve_vrp(config, depot, customers, task)
    
    if routes:
        logger.info(f"Solution Found! Routes: {routes}")
        
        # 5. Visualization
        plot_routes(depot, customers, routes)
        
        # Report Image to ClearML
        task.logger.report_image("Result", "VRP Routes", local_path="vrp_solution.png")
        
    else:
        logger.warning("No solution found or Infeasible.")
        # If IIS was computed, it's irrelevant to plot routes
        if not routes:
             # We might optionally fail the task or just complete it with no result
             pass
    
    logger.info("Experiment Completed.")

if __name__ == "__main__":
    main()
