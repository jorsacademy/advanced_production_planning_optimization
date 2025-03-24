import pandas as pd
import numpy as np
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpInteger

class AdvancedProductionOptimizer:
    """
    Advanced Production Planning and Optimization module to handle various
    manufacturing scenarios including multi-level planning, capable-to-promise,
    and different production strategies.
    """
    
    def __init__(self, plant_data=None, orders=None):
        self.plant_data = plant_data
        self.orders = orders
        self.production_plan = None
        self.resources = {}
        self.products = None
        self.bom = None
        self.routing = None
        self.inventory = None
        self.constraints = []
        self.heuristic_mode = "standard"  # can be "standard", "custom", or specific heuristic name
        
    def set_heuristic(self, heuristic_type):
        """Configure planning heuristic approach"""
        self.heuristic_mode = heuristic_type
        return True
        
    def load_resources(self, resource_file):
        """Load manufacturing resource data"""
        try:
            self.resources = pd.read_excel(resource_file)
            return True
        except Exception as e:
            print(f"Error loading resources: {e}")
            return False
    
    def plan_multi_level_production(self, horizon_days=30):
        """
        Create a multi-level production plan based on demand, inventory and capacity
        
        Parameters:
        horizon_days (int): Planning horizon in days
        
        Returns:
        DataFrame: Production plan with detailed scheduling
        """
        # Initialize model
        model = LpProblem(name="multi_level_production", sense=LpMaximize)
        
        # Mock data for demonstration - would be replaced with actual data in production
        products = ["ProductA", "ProductB", "ProductC", "SubassemblyA", "SubassemblyB", "RawMaterial"]
        resources = ["Machine1", "Machine2", "Assembly", "Labor"]
        periods = range(1, horizon_days + 1)
        
        # Create decision variables
        production_vars = {}
        for product in products:
            for period in periods:
                production_vars[(product, period)] = LpVariable(
                    name=f"Produce_{product}_P{period}", 
                    lowBound=0, 
                    cat=LpInteger
                )
        
        # Objective function: Maximize on-time delivery or profit
        model += lpSum([production_vars[(p, t)] for p in products for t in periods])
        
        # Add constraints based on capacity, materials, etc.
        # This is a simplified example - real implementation would include 
        # detailed resource/material constraints and BOM logic
        
        for resource in resources:
            for period in periods:
                # Simplified resource constraint
                model += lpSum([
                    production_vars[(p, period)] * self._resource_consumption(p, resource) 
                    for p in products
                ]) <= self._available_capacity(resource, period)
        
        # Solve model
        model.solve()
        
        # Convert results to dataframe
        results = []
        for product in products:
            for period in periods:
                if production_vars[(product, period)].value() > 0:
                    results.append({
                        'product': product,
                        'period': period,
                        'quantity': production_vars[(product, period)].value(),
                        'resource': self._determine_main_resource(product)
                    })
                    
        self.production_plan = pd.DataFrame(results)
        return self.production_plan
    
    def available_to_promise(self, order_id, quantity, requested_date):
        """
        Calculate if an order can be promised by requested date
        
        Parameters:
        order_id (str): Unique order identifier
        quantity (int): Requested quantity
        requested_date (datetime): Customer requested date
        
        Returns:
        dict: Promise details including confirmation, date, and quantity
        """
        # Check inventory and planned production
        available = self._check_global_availability(order_id, quantity, requested_date)
        
        if available['can_fulfill']:
            return {
                'order_id': order_id,
                'confirmed': True,
                'promise_date': available['date'],
                'promise_quantity': quantity
            }
        else:
            # Try to find alternative date
            alternative = self._find_alternative_promise(order_id, quantity, requested_date)
            return {
                'order_id': order_id,
                'confirmed': alternative['can_fulfill'],
                'promise_date': alternative['date'],
                'promise_quantity': alternative['quantity']
            }
    
    def calculate_lot_sizes(self, product_id, total_demand, scheduling_period=30):
        """
        Calculate optimal lot sizes based on demand, setup costs, and holding costs
        
        Parameters:
        product_id (str): Product identifier
        total_demand (int): Total demand for the period
        scheduling_period (int): Scheduling period in days
        
        Returns:
        list: Recommended lot sizes and production dates
        """
        # Mock data - in production this would pull from configuration
        setup_cost = 500  # Cost per setup
        holding_cost = 0.1  # Daily holding cost per unit
        production_rate = 100  # Units per day
        
        # Simple Economic Order Quantity calculation
        eoq = np.sqrt((2 * total_demand * setup_cost) / holding_cost)
        number_of_lots = max(1, round(total_demand / eoq))
        
        # Distribute lots over scheduling period
        lot_size = total_demand / number_of_lots
        lots = []
        
        for i in range(number_of_lots):
            day = int(i * (scheduling_period / number_of_lots)) + 1
            lots.append({
                'product_id': product_id,
                'lot_number': i + 1,
                'lot_size': int(lot_size),
                'scheduled_day': day,
                'completion_day': day + int(lot_size / production_rate)
            })
            
        return lots
    
    def generate_production_schedule(self, manufacturing_type="make-to-stock"):
        """
        Generate production schedule based on manufacturing strategy
        
        Parameters:
        manufacturing_type (str): One of "make-to-stock", "make-to-order", 
                                  "engineer-to-order", "project", "flow"
        
        Returns:
        DataFrame: Detailed production schedule
        """
        if not self.production_plan:
            self.plan_multi_level_production()
            
        # Different scheduling logic based on manufacturing type
        if manufacturing_type == "make-to-stock":
            return self._schedule_make_to_stock()
        elif manufacturing_type == "make-to-order":
            return self._schedule_make_to_order()
        elif manufacturing_type == "engineer-to-order":
            return self._schedule_engineer_to_order()
        elif manufacturing_type == "project":
            return self._schedule_project_manufacturing()
        elif manufacturing_type == "flow":
            return self._schedule_flow_manufacturing()
        else:
            raise ValueError(f"Unsupported manufacturing type: {manufacturing_type}")
    
    def peg_orders(self):
        """
        Link customer orders to specific production lots or inventory
        
        Returns:
        DataFrame: Order pegging results
        """
        if not self.orders or not self.production_plan:
            raise ValueError("Orders and production plan must be loaded first")
            
        pegging_results = []
        
        # Simple FIFO pegging for demonstration
        for _, order in self.orders.iterrows():
            remaining = order['quantity']
            order_pegs = []
            
            # First check inventory
            inventory_pegged = min(remaining, self._get_available_inventory(order['product']))
            if inventory_pegged > 0:
                order_pegs.append({
                    'order_id': order['id'],
                    'source_type': 'inventory',
                    'source_id': f"INV-{order['product']}",
                    'quantity': inventory_pegged
                })
                remaining -= inventory_pegged
            
            # Then check production plan
            if remaining > 0:
                prod_plan = self.production_plan[
                    (self.production_plan['product'] == order['product']) & 
                    (self.production_plan['period'] <= order['due_period'])
                ].sort_values('period')
                
                for _, prod in prod_plan.iterrows():
                    available = min(remaining, self._get_unpegged_quantity(prod['product'], prod['period']))
                    if available > 0:
                        order_pegs.append({
                            'order_id': order['id'],
                            'source_type': 'production',
                            'source_id': f"PROD-{prod['product']}-P{prod['period']}",
                            'quantity': available
                        })
                        remaining -= available
                    
                    if remaining == 0:
                        break
            
            pegging_results.extend(order_pegs)
        
        return pd.DataFrame(pegging_results)
    
    # Internal helper methods
    def _resource_consumption(self, product, resource):
        """Determine how much of a resource is consumed by producing a product"""
        # Mock implementation - would connect to BOMs and routings in production
        consumption_rates = {
            "ProductA": {"Machine1": 2, "Machine2": 0, "Assembly": 1, "Labor": 3},
            "ProductB": {"Machine1": 0, "Machine2": 2, "Assembly": 1, "Labor": 2},
            "ProductC": {"Machine1": 1, "Machine2": 1, "Assembly": 2, "Labor": 4},
            "SubassemblyA": {"Machine1": 1, "Machine2": 0, "Assembly": 0, "Labor": 1},
            "SubassemblyB": {"Machine1": 0, "Machine2": 1, "Assembly": 0, "Labor": 1},
            "RawMaterial": {"Machine1": 0, "Machine2": 0, "Assembly": 0, "Labor": 0}
        }
        return consumption_rates.get(product, {}).get(resource, 0)
    
    def _available_capacity(self, resource, period):
        """Get available capacity for a resource in a period"""
        # Mock implementation - would connect to capacity calendar in production
        base_capacity = {"Machine1": 16, "Machine2": 16, "Assembly": 24, "Labor": 40}
        
        # Simulate some capacity variations over time
        variation = np.sin(period / 7 * np.pi) * 0.2 + 1  # +/- 20% variation
        
        return base_capacity.get(resource, 0) * variation
    
    def _determine_main_resource(self, product):
        """Determine the main resource used for a product"""
        resources = ["Machine1", "Machine2", "Assembly", "Labor"]
        consumption = [self._resource_consumption(product, r) for r in resources]
        if max(consumption) > 0:
            return resources[consumption.index(max(consumption))]
        return None
    
    def _check_global_availability(self, order_id, quantity, date):
        """Check global availability across plants and warehouses"""
        # Mock implementation - would check actual inventory and planned receipts
        return {
            'can_fulfill': np.random.random() > 0.3,  # 70% chance of fulfillment for demo
            'date': date,
            'quantity': quantity
        }
    
    def _find_alternative_promise(self, order_id, quantity, requested_date):
        """Find alternative promise dates and quantities"""
        # Mock implementation - would use actual planning data
        delay_days = int(np.random.exponential(5)) + 1
        alternative_date = requested_date + pd.Timedelta(days=delay_days)
        
        return {
            'can_fulfill': True,
            'date': alternative_date,
            'quantity': quantity
        }
    
    def _get_available_inventory(self, product):
        """Get available inventory for a product"""
        # Mock implementation - would connect to inventory management
        return np.random.randint(10, 100)
    
    def _get_unpegged_quantity(self, product, period):
        """Get unpegged production quantity"""
        # Mock implementation - would track actual pegging
        return np.random.randint(5, 50)
    
    def _schedule_make_to_stock(self):
        """Schedule for make-to-stock manufacturing"""
        # Implementation would include inventory targets, forecasts, etc.
        return self._create_generic_schedule("make-to-stock")
    
    def _schedule_make_to_order(self):
        """Schedule for make-to-order manufacturing"""
        # Implementation would include customer orders, lead times, etc.
        return self._create_generic_schedule("make-to-order")
    
    def _schedule_engineer_to_order(self):
        """Schedule for engineer-to-order manufacturing"""
        # Implementation would include engineering tasks, BOM development, etc.
        return self._create_generic_schedule("engineer-to-order")
    
    def _schedule_project_manufacturing(self):
        """Schedule for project manufacturing"""
        # Implementation would include project phases, milestones, etc.
        return self._create_generic_schedule("project")
    
    def _schedule_flow_manufacturing(self):
        """Schedule for flow manufacturing"""
        # Implementation would include flow rates, takt times, etc.
        return self._create_generic_schedule("flow")
    
    def _create_generic_schedule(self, manufacturing_type):
        """Create a generic schedule for demonstration"""
        if not self.production_plan:
            return pd.DataFrame()
            
        schedule = self.production_plan.copy()
        
        # Add scheduling details based on manufacturing type
        schedule['manufacturing_type'] = manufacturing_type
        schedule['start_time'] = schedule['period'] * 24 - np.random.randint(1, 24, size=len(schedule))
        schedule['end_time'] = schedule['start_time'] + np.random.randint(1, 48, size=len(schedule))
        
        return schedule



    # Hypothetical scenario for a manufacturing company
if __name__ == "__main__":
    # Create sample order data
    orders_data = pd.DataFrame({
        'id': ['ORD001', 'ORD002', 'ORD003', 'ORD004', 'ORD005'],
        'product': ['ProductA', 'ProductB', 'ProductC', 'ProductA', 'ProductB'],
        'quantity': [100, 50, 200, 75, 150],
        'due_date': pd.to_datetime(['2025-04-15', '2025-04-20', '2025-04-30', '2025-04-25', '2025-05-05']),
        'due_period': [15, 20, 30, 25, 35],
        'priority': [2, 1, 3, 2, 1],  # 1=high, 3=low
        'customer': ['CUST001', 'CUST002', 'CUST001', 'CUST003', 'CUST002']
    })
    
    # Create sample resource data
    resource_data = pd.DataFrame({
        'resource_id': ['MACH001', 'MACH002', 'MACH003', 'ASSY001', 'ASSY002', 'LAB001'],
        'resource_type': ['Machine', 'Machine', 'Machine', 'Assembly', 'Assembly', 'Labor'],
        'capacity_per_day': [16, 16, 8, 24, 16, 40],
        'setup_time': [60, 45, 90, 30, 45, 15],  # minutes
        'efficiency': [0.95, 0.92, 0.88, 0.9, 0.93, 0.85]
    })
    
    # Create sample product data
    product_data = pd.DataFrame({
        'product_id': ['ProductA', 'ProductB', 'ProductC', 'SubassemblyA', 'SubassemblyB', 'RawMaterial'],
        'description': ['Finished Product A', 'Finished Product B', 'Finished Product C', 
                       'Subassembly A', 'Subassembly B', 'Raw Material'],
        'manufacturing_type': ['make-to-order', 'make-to-stock', 'engineer-to-order', 
                              'make-to-stock', 'make-to-stock', 'make-to-stock'],
        'lead_time_days': [5, 3, 10, 2, 2, 1],
        'min_lot_size': [20, 25, 10, 50, 50, 100],
        'max_lot_size': [200, 250, 100, 500, 500, 1000],
        'setup_cost': [500, 450, 800, 200, 200, 100],
        'holding_cost_per_day': [0.5, 0.4, 0.8, 0.2, 0.2, 0.1]
    })
    
    # Create BOM (Bill of Materials) relationships
    bom_data = pd.DataFrame({
        'parent_id': ['ProductA', 'ProductA', 'ProductB', 'ProductB', 'ProductC', 'ProductC', 
                     'SubassemblyA', 'SubassemblyB'],
        'component_id': ['SubassemblyA', 'RawMaterial', 'SubassemblyB', 'RawMaterial', 
                        'SubassemblyA', 'SubassemblyB', 'RawMaterial', 'RawMaterial'],
        'quantity': [2, 5, 1, 8, 3, 2, 3, 4]
    })
    
    # Create routing data (operations sequence)
    routing_data = pd.DataFrame({
        'product_id': ['ProductA', 'ProductA', 'ProductB', 'ProductB', 'ProductC', 'ProductC', 'ProductC',
                      'SubassemblyA', 'SubassemblyB'],
        'operation_seq': [10, 20, 10, 20, 10, 20, 30, 10, 10],
        'operation_desc': ['Processing', 'Assembly', 'Processing', 'Assembly', 'Engineering', 
                          'Processing', 'Assembly', 'Processing', 'Processing'],
        'resource_id': ['MACH001', 'ASSY001', 'MACH002', 'ASSY001', 'LAB001', 
                       'MACH003', 'ASSY002', 'MACH001', 'MACH002'],
        'runtime_per_unit': [15, 20, 10, 15, 60, 30, 45, 8, 12]  # minutes
    })
    
    # Create inventory data
    inventory_data = pd.DataFrame({
        'product_id': ['ProductA', 'ProductB', 'ProductC', 'SubassemblyA', 'SubassemblyB', 'RawMaterial'],
        'on_hand': [30, 120, 15, 200, 180, 500],
        'allocated': [20, 50, 0, 100, 80, 200],
        'available': [10, 70, 15, 100, 100, 300],
        'safety_stock': [20, 50, 10, 100, 100, 200]
    })
    
    print("===== Advanced Production Planning and Optimization Demo =====")
    print("\nHypothetical Scenario: Manufacturing Company with Multi-level Products")
    
    # Initialize the optimizer with all data
    optimizer = AdvancedProductionOptimizer()
    optimizer.orders = orders_data
    optimizer.resources = resource_data
    optimizer.products = product_data
    optimizer.bom = bom_data
    optimizer.routing = routing_data
    optimizer.inventory = inventory_data
    
    print("\n1. Planning multi-level production for the next 30 days...")
    production_plan = optimizer.plan_multi_level_production(horizon_days=30)
    
    # Since we're demonstrating hypothetically, we'll create a sample production plan
    # that would normally be calculated by the optimizer
    sample_plan_data = []
    for product in ['ProductA', 'ProductB', 'ProductC', 'SubassemblyA', 'SubassemblyB']:
        for period in range(1, 31):
            if np.random.random() > 0.8:  # Only generate records for some product-period combinations
                quantity = np.random.randint(10, 100) if product.startswith('Product') else np.random.randint(50, 200)
                sample_plan_data.append({
                    'product': product,
                    'period': period,
                    'quantity': quantity,
                    'resource': optimizer._determine_main_resource(product)
                })
    
    # Create sample production plan dataframe
    optimizer.production_plan = pd.DataFrame(sample_plan_data)
    
    print("Multi-level Production Plan (Sample):")
    if len(optimizer.production_plan) > 0:
        print(optimizer.production_plan.head())
    else:
        print("No production planned based on current constraints.")
    
    print("\n2. Generating detailed schedule for make-to-order products...")
    # For demonstration, create a sample detailed schedule
    detailed_data = []
    for idx, row in optimizer.production_plan.iterrows():
        if row['product'] in ['ProductA', 'ProductC']:  # Make-to-order products
            start_hour = np.random.randint(0, 16)
            runtime_hours = np.random.randint(2, 8)
            detailed_data.append({
                'product': row['product'],
                'period': row['period'],
                'quantity': row['quantity'],
                'resource': row['resource'],
                'manufacturing_type': 'make-to-order',
                'start_time': row['period'] * 24 - start_hour,
                'end_time': row['period'] * 24 - start_hour + runtime_hours,
                'order_id': np.random.choice(['ORD001', 'ORD003', 'ORD004']) if row['product'] == 'ProductA' else 'ORD003'
            })
    
    detailed_schedule = pd.DataFrame(detailed_data)
    print("Detailed Production Schedule (Make-to-Order):")
    if len(detailed_schedule) > 0:
        print(detailed_schedule.head())
    else:
        print("No make-to-order products scheduled.")
    
    print("\n3. Calculating lot sizes for ProductA (total demand: 500 units)...")
    lots = optimizer.calculate_lot_sizes("ProductA", 500, 30)
    print("Lot Size Calculation:")
    for lot in lots[:3]:  # Show first 3 lots
        print(f"  Lot {lot['lot_number']}: {lot['lot_size']} units scheduled for day {lot['scheduled_day']}")
    
    print("\n4. Checking available-to-promise for a new order (75 units of ProductA)...")
    promise = optimizer.available_to_promise("ORD006", 75, pd.Timestamp("2025-04-25"))
    print("Available-to-Promise Result:")
    print(f"  Order: ORD006, Confirmed: {promise['confirmed']}")
    print(f"  Promise Date: {promise['promise_date']}")
    print(f"  Promise Quantity: {promise['promise_quantity']}")
    
    print("\n5. Pegging orders to production lots...")
    # For demonstration, create sample pegging results
    pegging_data = []
    for order_id in ['ORD001', 'ORD002', 'ORD003', 'ORD004']:
        order_row = orders_data[orders_data['id'] == order_id].iloc[0]
        product = order_row['product']
        quantity = order_row['quantity']
        
        # First check inventory
        inventory_row = inventory_data[inventory_data['product_id'] == product].iloc[0]
        inventory_available = max(0, min(quantity, inventory_row['available']))
        
        if inventory_available > 0:
            pegging_data.append({
                'order_id': order_id,
                'source_type': 'inventory',
                'source_id': f"INV-{product}",
                'quantity': inventory_available,
                'date': pd.Timestamp('today')
            })
            quantity -= inventory_available
        
        # Then check production lots
        if quantity > 0 and len(optimizer.production_plan) > 0:
            relevant_production = optimizer.production_plan[
                (optimizer.production_plan['product'] == product) & 
                (optimizer.production_plan['period'] <= order_row['due_period'])
            ].sort_values('period')
            
            for idx, prod_row in relevant_production.iterrows():
                if quantity <= 0:
                    break
                    
                peg_quantity = min(quantity, prod_row['quantity'])
                pegging_data.append({
                    'order_id': order_id,
                    'source_type': 'production',
                    'source_id': f"PROD-{prod_row['product']}-P{prod_row['period']}",
                    'quantity': peg_quantity,
                    'date': pd.Timestamp('today') + pd.Timedelta(days=prod_row['period'])
                })
                quantity -= peg_quantity
    
    pegging = pd.DataFrame(pegging_data)
    print("Order Pegging Results:")
    if len(pegging) > 0:
        print(pegging.head())
    else:
        print("No orders pegged to inventory or production.")
    
    print("\n6. Global available-to-promise check across multiple facilities...")
    # Simulate a global ATP check across multiple facilities
    facilities = ['PLANT1', 'PLANT2', 'DC_EAST', 'DC_WEST']
    print("Checking availability for ProductB, quantity 200, due date 2025-05-01")
    
    availability_results = []
    for facility in facilities:
        available = np.random.random() > 0.4
        if available:
            delay = np.random.randint(0, 5)
            available_qty = np.random.randint(50, 250)
            availability_results.append({
                'facility': facility,
                'available': available,
                'available_quantity': available_qty if available else 0,
                'available_date': pd.Timestamp('2025-05-01') + pd.Timedelta(days=delay) if available else None
            })
    
    availability_df = pd.DataFrame(availability_results)
    print("Global Availability Results:")
    print(availability_df)
    
    # Make a final ATP decision based on all facilities
    best_option = availability_df[availability_df['available']].sort_values('available_date').iloc[0] if len(availability_df[availability_df['available']]) > 0 else None
    
    if best_option is not None:
        print(f"\nBest fulfillment option: {best_option['facility']} can provide {best_option['available_quantity']} units by {best_option['available_date'].strftime('%Y-%m-%d')}")
    else:
        print("\nNo facilities can fulfill this request within the timeframe")
