# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 23:29:36 2025

@author: MaxSc
"""

#%% import modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
import time
import copy 
from scipy.special import erfinv

#%% distribution functions

def Weibull(shape, scale):
    return scale*(-math.log(random.uniform(0, 1)))**(1/shape)

def Exponential(shape, lamda):
    return Weibull(1, 1/lamda)


def Lognormal(mu, sigma):
    u = random.uniform(0, 1)
    z = math.sqrt(2) * erfinv(2*u - 1)
    return math.exp(mu + sigma * z)

def Battery_replacement(shape, scale):
    return 1

#%% global parameters

#see below for running the model


#----- Change only this !!! ---------
num_runs = 1000
max_time = 10000

min_operational = 11

num_diesel_locs = 4
num_hybrid_locs = 10
extra_BU = 0
extra_DGU = 0

reliability_params = {"diesel loc": {"Fshape": 1.2313,
                                     "Fscale": 348.788,
                                     "Distribution": Weibull
                                     },
                      "hybrid loc": {"Fshape": 1.2313,
                                     "Fscale": 400,
                                     "Distribution": Weibull
                                     },
                      "BU":         {"Fshape": 4.96,
                                     "Fscale": 16283,
                                     "Distribution": Weibull
                                     },
                      "DGU":        {"Fshape": 4.319,
                                     "Fscale": 350,
                                     "Distribution": Weibull
                                     }
                      }

maintainability_params = {"diesel loc": {"Mshape": 4.0392,
                                         "Mscale": 0.9688,
                                         "Distribution": Lognormal
                                         },
                          "hybrid loc": {"Mshape": 4.0392,
                                         "Mscale": 0.9688,
                                         "Distribution": Lognormal
                                         },
                          "BU":         {"Mshape": 1,
                                         "Mscale": 1,
                                         "Distribution": Battery_replacement
                                         },
                          "DGU":        {"Mshape": 2,
                                         "Mscale": 90,
                                         "Distribution": Weibull
                                         }
                          }

pm_loc_tinterval = 2 #weeks

#-------------------------------

num_BUs = num_hybrid_locs + extra_BU
num_DGUs = num_hybrid_locs + extra_DGU

pm_intervals = {"diesel loc": pm_loc_tinterval*7*24,
                "hybrid loc": pm_loc_tinterval*7*24,
                "BU": None,
                "DGU": pm_loc_tinterval*7*24
                      }


random.seed(0)

#%% classes
class Component:
    all_components = []
    in_maintenance = set()
    in_use = set()
    preventive_maintenance = set()
    
    def __init__(self, name):
        self.name = name
        self.operational = True
        self.in_use = False
        self.F_param1, self.F_param2, self.Fdistribution = [reliability_params[self.Type][p] for p in reliability_params[self.Type]]
        self.M_param1, self.M_param2, self.Mdistribution = [maintainability_params[self.Type][p] for p in maintainability_params[self.Type]]
        self.tt_failure = self.Fdistribution(self.F_param1, self.F_param2)
        self.pm_interval = pm_intervals[self.Type]
        self.next_pm = None
        cls = self.__class__
        if self.pm_interval:
            cls.preventive_maintenance.add(self) 
            Component.preventive_maintenance.add(self) 
        # cls.stand_by.add(self)
        Component.all_components.append(self)
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, operational={self.operational}, in_use={self.in_use})"
        
    def failed(self):
        if self.is_failed:
            raise Exception("Failed component cannot fail.")
        self.tt_maintenance = self.Mdistribution(self.M_param1, self.M_param2)
        self.operational = False
        self.in_use = False
        self.update_tracking()
        
    def fixed(self):
        if self.operational:
            raise Exception("Operational component cannot be maintenanced.")
        self.tt_failure = self.Fdistribution(self.F_param1, self.F_param2)
        self.operational = True
        self.update_tracking()
        
    def preventive(self):
        if self.is_failed:
            self.next_pm = self.pm_interval
        else:
            self.tt_maintenance = self.Mdistribution(self.M_param1, self.M_param2)
            self.operational = False
            self.in_use = False
            self.next_pm = self.pm_interval
            self.update_tracking()
        
    def use(self):
        if self.is_failed:
            raise Exception(f"Cannot use, {self.name} is not operational")
        self.in_use = True
        self.update_tracking()
        
    def update_tracking(self):
        cls = self.__class__
        # print(f"🔄 {self.name} tracking update — op: {self.operational}, in_use: {self.in_use}.")
        cls.in_maintenance.discard(self)
        Component.in_maintenance.discard(self)
        cls.stand_by.discard(self)
        cls.in_use.discard(self)
        Component.in_use.discard(self)

        if self.is_failed:
            # print(f"➕ {self.name} ➜ in_maintenance")
            cls.in_maintenance.add(self)
            Component.in_maintenance.add(self)
        elif self.in_use:
            # print(f"➕ {self.name} ➜ in_use")
            cls.in_use.add(self)
            Component.in_use.add(self)
        else:
            # print(f"➕ {self.name} ➜ stand_by")
            cls.stand_by.add(self)
            
    @classmethod
    def reset_tracking(cls):
        cls.in_maintenance.clear()
        cls.in_use.clear()
        if hasattr(cls, "stand_by"): cls.stand_by.clear()
        if hasattr(cls, "on_loc"): cls.on_loc.clear()
        if hasattr(cls, "no_modules"): cls.no_modules.clear()
        
    @classmethod
    def validate_tracking(cls, total_num_object):
        num_maintenace = len(cls.in_maintenance)
        num_inuse = len(cls.in_use)
        num_standby = len(cls.stand_by)
        num_nomodules = len(cls.no_modules) if hasattr(cls, "no_modules") else 0
        num_onloc = len(cls.on_loc) if hasattr(cls, "on_loc") else 0
        total_tracked = num_maintenace + num_inuse + num_standby + num_nomodules + num_onloc
        if total_tracked != total_num_object:
            raise ValueError(f"Tracking mismatch in {cls.__name__}: expected {total_num_object}, found {total_tracked}")
    
    @property
    def is_failed(self):
        return not self.operational
    
        
class Locomotive(Component):
    stand_by = set()
    in_use = set()
    preventive_maintenance = set()
    
    def __init__(self, name):
        super().__init__(name)
    
    def update_tracking(self):
        cls = self.__class__
        cls.in_maintenance.discard(self)
        Component.in_maintenance.discard(self)
        cls.stand_by.discard(self)
        Locomotive.stand_by.discard(self)
        cls.in_use.discard(self)
        Component.in_use.discard(self)
        Locomotive.in_use.discard(self)

        if self.is_failed:
            cls.in_maintenance.add(self)
            Component.in_maintenance.add(self)
        elif self.in_use:
            cls.in_use.add(self)
            Locomotive.in_use.add(self)
            Component.in_use.add(self)
        else:
            cls.stand_by.add(self)
            Locomotive.stand_by.add(self)
            
    @classmethod
    def num_operational(cls):
        return len(cls.stand_by) + len(cls.in_use)
    

class Module(Component):
    
    def __init__(self, name):
        super().__init__(name)
        self.loco = None
        
    def assign_to(self, loco):
        if self.loco is not None:
            raise Exception(f"❌ cannot assign: {self.name} is already assigned to {self.loco.name}!")
        if not self.operational:
            raise Exception(f"❌ cannot assign: {self.name} is not operational.")
        self.in_use = True
        self.loco = loco
        self.update_tracking()
        
    def unassign(self):
        if self.loco == None:
            raise Exception(f"❌ cannot unassing {self.name}, because it is not assigned.")
        setattr(self.loco, self.Type, None)
        self.loco.in_use = False
        self.loco.update_tracking()
        self.loco = None
        self.in_use = False
        if self.operational:
            self.update_tracking()
        
    def failed(self):
        super().failed()
        loco = self.loco
        self.unassign() 
        loco.fill()
        
    def fixed(self):
        super().fixed()
        HybridLoc.assign_all_modules()
        
    def update_tracking(self):
        cls = self.__class__
        cls.in_maintenance.discard(self)
        Component.in_maintenance.discard(self)
        cls.stand_by.discard(self)
        cls.in_use.discard(self)
        Component.in_use.discard(self)
        cls.on_loc.discard(self)
    
        if self.is_failed:
            cls.in_maintenance.add(self)
            Component.in_maintenance.add(self)
        elif self.is_on_loc:
            cls.on_loc.add(self)
        elif self.in_use:
            cls.in_use.add(self)
            Component.in_use.add(self)
        else:
            cls.stand_by.add(self)
        
    @property
    def is_on_loc(self):
        return self.loco is not None and not self.loco.in_use
    
###--------- subclasses ------------------------------------------- 
    
class DGU(Module):
    in_maintenance = set()
    preventive_maintenance = set()
    stand_by = set()
    on_loc = set()
    in_use = set()
    
    def __init__(self, name):
        self.Type = "DGU"
        super().__init__(name)
        self.update_tracking()
    

class BU(Module):
    in_maintenance = set()
    stand_by = set()
    on_loc = set()
    in_use = set()
    
    def __init__(self, name):
        self.Type = "BU"
        super().__init__(name)
        self.update_tracking()
        
    def preventive(self):
        return #no pm for BU
        
               
class DieselLoc(Locomotive):
    in_maintenance = set()
    stand_by = set()
    in_use = set()
    
    def __init__(self, name):
        self.Type = "diesel loc"
        super().__init__(name)
        self.update_tracking()
        

class HybridLoc(Locomotive):
    in_maintenance = set()
    no_modules = set()
    stand_by = set()
    in_use = set()
    
    def __init__(self, name):
        self.Type = "hybrid loc"
        super().__init__(name)
        self.DGU = None
        self.BU = None
        self.filling = False
        self.fill()
        self.update_tracking()
        
    def assign_module(self, module):
        if self.is_failed:
            raise Exception(f"❌ cannot assign {module.Type} because {self.name} is not operational.")
        elif module.Type == "DGU" and self.DGU is None:
            self.DGU = module
            module.assign_to(self)
        elif module.Type == "BU" and self.BU is None:
            self.BU = module
            module.assign_to(self)
        else:
            print(f"⚠️ Warning: {module.name} could not be assigned to {self.name}")
            return
        self.update_tracking()
        
    def fill(self):
        if self.is_failed:
            return
        if not DGU.stand_by and not BU.stand_by:
            return
        if self.filling:
            return
        self.filling = True
    
        # Detect which modules are missing
        needs_dgu = self.DGU is None
        needs_bu = self.BU is None
    
        available_dgu = next((d for d in DGU.stand_by if not d.in_use and d.operational), None) if needs_dgu else None
        available_bu = next((b for b in BU.stand_by if not b.in_use and b.operational), None) if needs_bu else None
    
        # Try to assign individually as needed
        if needs_dgu and available_dgu:
            self.assign_module(available_dgu)
        if needs_bu and available_bu:
            self.assign_module(available_bu)
    
        # If after attempting fill we still don't have both, rollback both
        if self.has_no_modules:
            if self.DGU:
                self.DGU.unassign()
                self.DGU = None
            if self.BU:
                self.BU.unassign()
                self.BU = None
            self.in_use = False
    
        self.update_tracking()
        self.filling = False

        
    def failed(self):
        super().failed()
        if self.DGU:
            self.DGU.unassign()
        if self.BU:
            self.BU.unassign()
            
    def use(self):
        if self.is_failed:
            raise Exception(f"❌ cannot use, {self.name} is not operational")
        if self.has_no_modules:
            raise Exception(f"❌ cannot use, {self.name} has no modules")
        self.in_use = True
        self.update_tracking()
        self.DGU.update_tracking()
        self.BU.update_tracking()
        
    @property
    def has_modules(self):
        return self.DGU is not None and self.BU is not None
    
    @property
    def has_no_modules(self):
        return self.DGU is None or self.BU is None
            
    def update_tracking(self):
        cls = self.__class__
        cls.in_maintenance.discard(self)
        Component.in_maintenance.discard(self)
        cls.no_modules.discard(self)
        cls.stand_by.discard(self)
        Locomotive.stand_by.discard(self)
        cls.in_use.discard(self)
        Component.in_use.discard(self)
        Locomotive.in_use.discard(self)

        if self.is_failed:
            cls.in_maintenance.add(self)
            Component.in_maintenance.add(self)
        elif self.has_no_modules:
            cls.no_modules.add(self)
            self.fill()
        elif self.in_use:
            cls.in_use.add(self)
            Locomotive.in_use.add(self)
            Component.in_use.add(self)
        else:
            cls.stand_by.add(self)
            Locomotive.stand_by.add(self)
            
    def validate(self):
        if self.has_no_modules and self.has_modules is True:
            raise Exception(f"{self.name} is missing a module.")
        if (self.DGU or self.BU) and self.is_failed:
            raise Exception(f"{self.name} has modules but is not operational.")
        if self.DGU.loco != self:
            raise Exception(f"{self.name} is matched to {self.DGU.name}, but {self.DGU.name} is assigned to {self.DGU.loco.name}")
        if self.BU.loco != self:
            raise Exception(f"{self.name} is matched to {self.BU.name}, but {self.BU.name} is assigned to {self.BU.loco.name}")
            
    @classmethod
    def assign_all_modules(cls):
        for h in cls.no_modules.copy():
            if not DGU.stand_by or not BU.stand_by:
                return
            else:
                h.fill()
                
#%% simulation funtions

def Generate_components(num_DGUs=num_DGUs, num_BUs=num_BUs):
    #reset all tracking sets
    for cls in [Component, Locomotive, DieselLoc, HybridLoc, BU, DGU]:
        cls.reset_tracking()
    Component.all_components.clear()

    diesellocs = [DieselLoc(f"DieselLoc{i+1}") for i in range(num_diesel_locs)]
    bus = [BU(f"BU{i+1}") for i in range(num_BUs)]
    dgus = [DGU(f"DGU{i+1}") for i in range(num_DGUs)]
    hybridlocs = [HybridLoc(f"HybridLoc{i+1}") for i in range(num_hybrid_locs)]
    
    for cls, total_num_object in zip([Locomotive, DGU], [num_diesel_locs + num_hybrid_locs, num_DGUs]):
        i = 1
        for comp in cls.preventive_maintenance:
            comp.next_pm = comp.pm_interval*(i/total_num_object)
            i += 1
        
    for cls, total_num_object in zip([DieselLoc, HybridLoc, BU, DGU], [num_diesel_locs, num_hybrid_locs, num_BUs, num_DGUs]):
        cls.validate_tracking(total_num_object)
        
    return diesellocs, bus, dgus, hybridlocs

def Test_component_tracking_behavior():
    print("\n===== 🔍 Running Component Tracking Tests =====")
    di, bu, dg, hy = Generate_components()

    # Pick test components
    hybrid = next(h for h in hy if h.has_modules)
    diesel = di[0]
    bu_mod = hybrid.BU
    dgu_mod = hybrid.DGU

    print(f"\n--- ⚠️ Test 1: HybridLoc Failure ({hybrid.name}) ---")
    hybrid.failed()
    print(f"✔ {hybrid.name} in HybridLoc.in_maintenance: {hybrid in HybridLoc.in_maintenance}")
    print(f"✔ Modules unassigned:")
    print(f"   - {bu_mod.name}.loco is None? {bu_mod.loco is None}")
    print(f"   - {dgu_mod.name}.loco is None? {dgu_mod.loco is None}")
    print(f"✔ Modules in stand_by:")
    print(f"   - BU: {bu_mod in BU.stand_by}")
    print(f"   - DGU: {dgu_mod in DGU.stand_by}")

    print(f"\n--- ⚙️ Test 2: Module Failure ({bu_mod.name}) ---")
    hybrid.fixed()  # this should reassign both modules
    bu_mod.failed()

    print(f"✔ {bu_mod.name} in BU.in_maintenance: {bu_mod in BU.in_maintenance}")
    print(f"✔ HybridLoc has module? {hybrid.has_modules}")
    print(f"✔ HybridLoc in no_modules: {hybrid in HybridLoc.no_modules}")
    print(f"✔ {bu_mod.name}.loco is None: {bu_mod.loco is None}")

    print(f"✔ {dgu_mod.name} still on loc? {dgu_mod.loco == hybrid}")
    print(f"✔ {dgu_mod.name} in DGU.on_loc: {dgu_mod in DGU.on_loc}")
    print(f"✔ {dgu_mod.name} not in DGU.in_use: {dgu_mod not in DGU.in_use}")
    print(f"✔ {dgu_mod.name} not in DGU.stand_by: {dgu_mod not in DGU.stand_by}")

    print(f"\n--- 🚂 Test 3: DieselLoc Failure ({diesel.name}) ---")
    diesel.failed()
    print(f"✔ {diesel.name} in DieselLoc.in_maintenance: {diesel in DieselLoc.in_maintenance}")
    diesel.fixed()
    print(f"✔ {diesel.name} in DieselLoc.stand_by after fix: {diesel in DieselLoc.stand_by}")

    print(f"\n--- 🧮 Global Component.in_use Consistency Check ---")
    all_okay = True
    for comp in Component.in_use:
        if not comp.operational:
            print(f"❌ ERROR: {comp.name} is in Component.in_use but is not operational.")
            all_okay = False
        if isinstance(comp, Module) and comp.loco and not comp.loco.in_use:
            print(f"❌ ERROR: {comp.name} is in Component.in_use but its loco is not in use.")
            all_okay = False
    if all_okay:
        print("✔ All components in Component.in_use are correctly tracked.")

    print(f"\n--- 🔄 Final Tracking Validation ---")
    for cls, expected_count in zip([DieselLoc, HybridLoc, BU, DGU], [num_diesel_locs, num_hybrid_locs, num_BUs, num_DGUs]):
        cls.validate_tracking(expected_count)
        print(f"✔ {cls.__name__} tracking totals validated ({expected_count})")

    print("\n✅ All tracking tests completed.\n")

def Next_event():
    min_obj = None
    min_value = float("inf")
    event_type = None

    # Check failures
    for obj in Component.in_use:
        if obj.tt_failure < min_value:
            min_obj = obj
            min_value = obj.tt_failure
            event_type = "failure"

    # Check repairs
    for obj in Component.in_maintenance:
        if obj.tt_maintenance < min_value:
            min_obj = obj
            min_value = obj.tt_maintenance
            event_type = "maintenance"
            
    # for obj in Component.preventive_maintenance:
    #     if obj.next_pm < min_value:
    #         min_obj = obj
    #         min_value = obj.next_pm
    #         event_type = "pm"
        

    return min_obj, min_value, event_type

def Advance_time_with(dt):
    for obj in Component.in_use:
        obj.tt_failure -= dt
        if obj.tt_failure < 0:
            raise ValueError(f"⚠️ Negative tt_failure for {obj.name}")

    for obj in Component.in_maintenance:
        obj.tt_maintenance -= dt
        if obj.tt_maintenance < 0:
            raise ValueError(f"⚠️ Negative tt_maintenance for {obj.name}")
            
    # for obj in Component.preventive_maintenance:
    #     obj.next_pm -= dt
    #     if obj.next_pm < 0:
    #         raise ValueError(f"⚠️ Negative next_pm for {obj.name}")

# def Simulate_to_failure(max_time):
    
#     start = time.time()
    
#     current_time = 0
    
#     diesellocs, bus, dgus, hybridlocs = Generate_components()
    
#     for i in range(min_operational):
#         Locomotive.stand_by.pop().use()
    
#     while Locomotive.num_operational() >= min_operational:
        
#         obj, dt, etype = Next_event()
        
#         current_time += dt
#         Advance_time_with(dt)
        
#         if etype == "failure":
#             obj.failed()
#         elif etype == "maintenance":
#             obj.fixed()
#         else:
#             obj.preventive()
        
#         if len(Locomotive.in_use) < min_operational:
#             if Locomotive.stand_by: 
#                 Locomotive.stand_by.pop().use()
#             else:
#                 break
        
#         if current_time > max_time:
#             break
        
#     end = time.time()
        
#     return {
#         "time_to_failure": current_time,
#         "simulation_time": end - start
#         }
            
def Simulate_to_failure(max_time, num_DGUs=num_DGUs, num_BUs=num_BUs):
    #run this function for a single simulation
    
    start = time.time()
    current_time = 0

    # Reset + generate components
    diesellocs, bus, dgus, hybridlocs = Generate_components(num_DGUs=num_DGUs, num_BUs=num_BUs)

    # Stats to track
    num_failures = 0
    diesel_failures = 0
    hybrid_failures = 0
    bu_failures = 0
    dgu_failures = 0

    # Start with minimum number of locomotives in use
    for _ in range(min_operational):
        Locomotive.stand_by.pop().use()

    while Locomotive.num_operational() >= min_operational:
        obj, dt, etype = Next_event()
        current_time += dt
        Advance_time_with(dt)

        if etype == "failure":
            num_failures += 1
            if isinstance(obj, DieselLoc):
                diesel_failures += 1
            elif isinstance(obj, HybridLoc):
                hybrid_failures += 1
            elif isinstance(obj, BU):
                bu_failures += 1
            elif isinstance(obj, DGU):
                dgu_failures += 1
            obj.failed()
        elif etype == "maintenance":
            obj.fixed()
        # else:
        #     obj.preventive()

        if len(Locomotive.in_use) < min_operational:
            if Locomotive.stand_by:
                Locomotive.stand_by.pop().use()
            else:
                break

        if current_time > max_time:
            break

    end = time.time()

    return {
        "time_to_failure": current_time,
        "simulation_time": end - start,
        "num_failures": num_failures,
        "diesel_failures": diesel_failures,
        "hybrid_failures": hybrid_failures,
        "bu_failures": bu_failures,
        "dgu_failures": dgu_failures
    }      
        
def Monte_Carlo_single(num_runs=num_runs, max_time=max_time):
    #run this for single monte carlo simulation
    
    
    print(f"\nStarting simulation")
    
    start = time.time()
    
    results = []
        
    t1 = time.time()
    dt = t1 - start
    
    sim_times = []

    for i in range(num_runs):
        if (i)%int(num_runs/6) == 0:
            avg = np.mean(sim_times) if sim_times else 0.01
            p = 100*(time.time() - t1)/((time.time() - t1) + avg*(num_runs-i))
            print(f"Simulating... ({p:.1f}%)")
            
        sim_data = Simulate_to_failure(max_time)
        results.append(sim_data)
        sim_times.append(sim_data["simulation_time"])
            
        
    t2 = time.time()
    dt = t2 - t1
    
    print(f"Finished simulation after {dt:.2f} sec")
    print(f"\nTime prediction for data analysis: 0.35 sec")
    print(f"Analysing data...")

    df = pd.DataFrame(results)

    # Running averages for convergence plots
    running_avg = df.cumsum() / (pd.Series(range(1, num_runs + 1)).values[:, None])

    # Plot time to failure separately
    plt.figure(figsize=(10, 5))
    plt.plot(running_avg['time_to_failure'], label='Time to Failure', color='blue')
    plt.title("Convergence of Time to Failure")
    plt.xlabel("Simulation Run")
    plt.ylabel("Running Average Time to Failure")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot all other metrics
    plt.figure(figsize=(12, 6))
    for col in ['num_failures', 'diesel_failures', 'hybrid_failures', 'bu_failures', 'dgu_failures']:
        plt.plot(running_avg[col], label=col.replace('_', ' ').title())

    plt.title("Convergence of Failure Statistics")
    plt.xlabel("Simulation Run")
    plt.ylabel("Running Average")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
        
    t3 = time.time()
    dt = t3 - t2
    
    ttf, nf, dif, hyf, buf, dgf = (df["time_to_failure"].mean(), df['num_failures'].mean(), df['diesel_failures'].mean(), df['hybrid_failures'].mean(), df['bu_failures'].mean(), df['dgu_failures'].mean())
    
    print(f"Finished data analysis after {dt:.2f} sec")
    print(f"\n-----------------RESULTS-------------------")
    print(f"Estimated values:")
    print(f"Time to failure (system):.........{ttf:.2f} hours")
    print(f"Total number of failures:.........{nf:.1f}")
    print(f"Num. of diesel loc failures:......{dif:.1f}")
    print(f"Num. of hybrid loc failures:......{hyf:.1f}")
    print(f"Num. of BU failures:..............{buf:.1f}")
    print(f"Num. of DGU failures:.............{dgf:.1f}")

    return df       
        

def Monte_Carlo(num_runs=num_runs, max_time=max_time,extra_mod=None, j=None, num_DGUs=num_DGUs, num_BUs=num_BUs):
    #dont run this
    
    print(f"Starting simulation {j}")
    
    start = time.time()
    
    results = []
        
    t1 = time.time()
    dt = t1 - start
    
    sim_times = []

    for i in range(num_runs):
        if (i)%int(num_runs/6) == 0:
            avg = np.mean(sim_times) if sim_times else 0.01
            p = 100*(((time.time() - t1)/((time.time() - t1) + avg*(num_runs-i)))/(extra_mod+1) + j/(extra_mod+1))
            print(f"Simulating... ({p:.1f}%)")
            
        sim_data = Simulate_to_failure(max_time, num_DGUs=num_DGUs, num_BUs=num_BUs)
        results.append(sim_data)
        sim_times.append(sim_data["simulation_time"])
            
        
    t2 = time.time()
    dt = t2 - t1
    
    print(f"Finished simulation {j} after {dt:.2f} sec")

    df = pd.DataFrame(results)

    return df   
    

def Monte_Carlo_multi(num_runs=num_runs, max_time=max_time, mod_type="DGU", extra_mod=3):
    #run this for MC simulation for a range of values for the number of modules
    
    print("\n")
    DF = []
    
    plt.figure(figsize=(10, 5))
    
    for j in range(extra_mod+1):
        if mod_type == "BU":
            num_BUs = num_hybrid_locs + j
            num_DGUs = num_hybrid_locs
        elif mod_type == "DGU":
            num_DGUs = num_hybrid_locs + j
            num_BUs = num_hybrid_locs
        else:
            raise Exception("Invalid argument")
         
        df = Monte_Carlo(num_runs=num_runs, max_time=max_time, extra_mod=extra_mod, j=j, num_DGUs=num_DGUs, num_BUs=num_BUs)
        DF.append(df)
        
        running_avg = df.cumsum() / (pd.Series(range(1, num_runs + 1)).values[:, None])
        
        
        plt.plot(running_avg['time_to_failure'], label=f'{j} extra {mod_type}s')
        
        
    print(f"Analysing data...")  
    print(f"\n-----------------RESULTS-------------------")
    print(f"Estimated TTF (system):")
    for i in range(extra_mod+1):
        val = DF[i]["time_to_failure"].mean()
        print(f"{i} extra {mod_type}s:.........{val:.2f} hours")
        
    plt.title("Convergence of Time to Failure")
    plt.xlabel("Simulation Run")
    plt.ylabel("Running Average Time to Failure")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
        
    num_BUs = num_hybrid_locs + extra_BU
    num_DGUs = num_hybrid_locs + extra_DGU
    return DF
















