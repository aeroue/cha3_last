# # # strong_communication/utility_calculator.py
# # import numpy as np
# # import sys
# # import os
# # from typing import Dict, Any, List, TYPE_CHECKING
# # import matplotlib.pyplot as plt 
# # import matplotlib 

# # # --- 统一的导入路径管理 ---
# # _utility_calc_dir = os.path.dirname(os.path.abspath(__file__))
# # _project_root_dir_util = os.path.dirname(_utility_calc_dir) 
# # if _project_root_dir_util not in sys.path:
# #     sys.path.insert(0, _project_root_dir_util)
# # # --- 路径管理结束 ---

# # from common.delta_v_estimator import DeltaVEstimatorMPC
# # from common.dynamics import CWEquation 

# # if TYPE_CHECKING: 
# #     from strong_communication.belief_manager import BeliefManager 

# # class UtilityCalculator:
# #     def __init__(self,
# #                  delta_v_estimator: DeltaVEstimatorMPC,
# #                  task_definitions: Dict[str, Dict[str, Any]], 
# #                  spacecraft_params: Dict[str, Dict[str, Any]] 
# #                 ):
# #         self.delta_v_estimator = delta_v_estimator
# #         self.task_definitions = task_definitions # This should be kept updated by the environment/KSC
# #         self.spacecraft_params = spacecraft_params
# #         self.g0 = 9.80665

# #     def calculate_fuel_mass_cost(self, spacecraft_id: str, delta_v_mps: float) -> float:
# #         if delta_v_mps == np.inf:
# #             return np.inf
        
# #         sc_params = self.spacecraft_params.get(spacecraft_id)
# #         if not sc_params:
# #             # print(f"    [FuelCalc WARNING] 未找到航天器 {spacecraft_id} 的参数。")
# #             return np.inf # Or raise error

# #         m0 = sc_params.get('initial_mass_kg')
# #         isp = sc_params.get('isp_s')

# #         if m0 is None or isp is None:
# #             # print(f"    [FuelCalc WARNING] 航天器 {spacecraft_id} 缺少 initial_mass_kg 或 isp_s 参数。")
# #             return np.inf
# #         if isp <= 1e-6: # 比冲必须为正
# #             # print(f"    [FuelCalc WARNING] 航天器 {spacecraft_id} Isp过小或为零 ({isp}).")
# #             return np.inf 

# #         fuel_mass = m0 * (1 - np.exp(-delta_v_mps / (isp * self.g0)))
# #         return fuel_mass

# #     def get_expected_task_utility_for_agent(self,
# #                                            agent_id: str,
# #                                            current_agent_state: np.ndarray, 
# #                                            task_id: str,
# #                                            agent_task_beliefs: Optional[np.ndarray], 
# #                                            task_type_names: List[str] 
# #                                            ) -> float:
# #         task_def = self.task_definitions.get(task_id) # Use the up-to-date task_definitions
# #         sc_params = self.spacecraft_params.get(agent_id)

# #         if not task_def or not sc_params:
# #             # print(f"    [UtilCalc DEBUG get_expected] Task {task_id} or SC {agent_id} def not found. TaskDef keys: {list(self.task_definitions.keys()) if self.task_definitions else 'None'}")
# #             return -np.inf

# #         if task_def.get("status") == "completed" or task_def.get("current_completion_percentage", 0.0) >= 100.0:
# #             # print(f"    [UtilCalc DEBUG get_expected] Task {task_id} is COMPLETED. Utility for assignment = -inf.")
# #             return -np.inf 

# #         expected_reward = 0.0
# #         task_rewards_by_type = task_def.get('type_rewards', {})
# #         if agent_task_beliefs is not None and len(task_type_names) == len(agent_task_beliefs) and self.task_type_names:
# #             for i, type_name in enumerate(self.task_type_names): # Use self.task_type_names for consistency
# #                 reward_value = task_rewards_by_type.get(type_name, 0.0) 
# #                 expected_reward += agent_task_beliefs[i] * reward_value
# #         elif agent_task_beliefs is None:
# #              # print(f"    [UtilCalc WARNING get_expected] Agent {agent_id} has NO beliefs for task {task_id}. Expected reward = 0.")
# #              pass # expected_reward remains 0
# #         else: # Mismatch in belief/type_names length
# #             # print(f"    [UtilCalc ERROR get_expected] Task type names len ({len(task_type_names)}) vs beliefs len ({len(agent_task_beliefs)}) mismatch for task {task_id}.")
# #             return -np.inf
# #         # print(f"      [UtilCalc DEBUG get_expected] Task {task_id}, Agent {agent_id}: Expected Reward = {expected_reward:.2f}")

# #         expected_risk_cost = 0.0
# #         task_risks_by_type = task_def.get('type_risks', {})
# #         risk_aversion = sc_params.get('risk_aversion_factor', 1.0)
# #         if agent_task_beliefs is not None and len(self.task_type_names) == len(agent_task_beliefs) and self.task_type_names:
# #             for i, type_name in enumerate(self.task_type_names):
# #                 risk_value = task_risks_by_type.get(type_name, 0.0)
# #                 expected_risk_cost += agent_task_beliefs[i] * risk_value * risk_aversion
# #         # print(f"      [UtilCalc DEBUG get_expected] Task {task_id}, Agent {agent_id}: Expected Risk Cost = {expected_risk_cost:.2f}")
        
# #         task_position = task_def.get('position')
# #         if task_position is None:
# #             # print(f"    [UtilCalc ERROR get_expected] Task {task_id} has no position defined. Returning -inf.")
# #             return -np.inf
        
# #         target_velocity_nominal = np.array([0.0] * self.delta_v_estimator.control_size) # Default to zero velocity if not specified
# #         target_velocity = task_def.get('target_velocity', target_velocity_nominal)
        
# #         # Ensure target_velocity has the correct dimension (control_size for estimator)
# #         if not isinstance(target_velocity, np.ndarray) or len(target_velocity) != self.delta_v_estimator.control_size:
# #             target_velocity = target_velocity_nominal
        
# #         # Construct target state [pos, vel] for DeltaV estimator
# #         # Ensure position part matches the state_size//2 expected by estimator (usually 2 for [x,y])
# #         pos_part_len = self.delta_v_estimator.state_size // 2
# #         vel_part_len = self.delta_v_estimator.state_size - pos_part_len

# #         try:
# #             pos_for_target = np.array(task_position)[:pos_part_len]
# #             vel_for_target = np.array(target_velocity)[:vel_part_len] # Should match control_size
# #             if len(pos_for_target) != pos_part_len or len(vel_for_target) != vel_part_len:
# #                  raise ValueError("Position or velocity part length mismatch for target state.")
# #             target_task_state = np.concatenate((pos_for_target, vel_for_target))
# #         except Exception as e_state_build:
# #             # print(f"    [UtilCalc ERROR get_expected] Failed to build target_task_state for {task_id}: {e_state_build}. Returning -inf.")
# #             return -np.inf

# #         if len(target_task_state) != self.delta_v_estimator.state_size:
# #             # print(f"    [UtilCalc ERROR get_expected] Constructed target_task_state for {task_id} has wrong dimension {len(target_task_state)}. Expected {self.delta_v_estimator.state_size}. Returning -inf.")
# #             return -np.inf

# #         delta_v, _, success = self.delta_v_estimator.estimate_delta_v(current_agent_state, target_task_state)
# #         # print(f"      [UtilCalc DEBUG get_expected] Task {task_id}, Agent {agent_id}: Delta-V Est - Success: {success}, Value: {'Inf' if delta_v == np.inf else delta_v:.3f} m/s")
        
# #         if not success or delta_v == np.inf:
# #             return -np.inf

# #         fuel_cost_mass = self.calculate_fuel_mass_cost(agent_id, delta_v)
# #         if fuel_cost_mass == np.inf:
# #             return -np.inf

# #         fuel_cost_coeff = sc_params.get('fuel_cost_coeff', 1.0) # Get fuel cost coefficient
# #         fuel_utility_cost = fuel_cost_mass * fuel_cost_coeff
# #         # print(f"      [UtilCalc DEBUG get_expected] Task {task_id}, Agent {agent_id}: Fuel Mass = {fuel_cost_mass:.3f} kg, Coeff: {fuel_cost_coeff}, Fuel Util Cost = {fuel_utility_cost:.2f}")

# #         total_utility = expected_reward - expected_risk_cost - fuel_utility_cost
# #         # print(f"    [UtilCalc INFO get_expected] Task {task_id}, Agent {agent_id}: Calc Total Util = {total_utility:.2f} (R:{expected_reward:.1f}-RiskC:{expected_risk_cost:.1f}-FuelC:{fuel_utility_cost:.1f})")
        
# #         return total_utility

# #     def calculate_coalition_task_utility(self, # (保持与您上传版本一致)
# #                                          coalition_agent_ids: List[str],
# #                                          current_coalition_states: Dict[str, np.ndarray], 
# #                                          task_id: str,
# #                                          belief_manager: 'BeliefManager', 
# #                                          task_type_names: List[str]
# #                                          ) -> float:
# #         if not coalition_agent_ids:
# #             return -np.inf
# #         max_individual_utility_in_coalition = -np.inf
# #         for agent_id in coalition_agent_ids:
# #             current_state = current_coalition_states.get(agent_id)
# #             if current_state is None: continue
# #             agent_beliefs_for_task = belief_manager.get_agent_belief_for_task(agent_id, task_id) # KSC Belief manager
# #             if agent_beliefs_for_task is None: continue

# #             utility = self.get_expected_task_utility_for_agent(
# #                 agent_id, current_state, task_id, agent_beliefs_for_task, task_type_names
# #             )
# #             if utility > max_individual_utility_in_coalition:
# #                 max_individual_utility_in_coalition = utility
# #         return max_individual_utility_in_coalition

# # # --- 模块自测试与可视化 (保持与您上传版本一致) ---
# # # (if __name__ == '__main__': test_utility_calculator() ...)
# # strong_communication/utility_calculator.py
# import numpy as np
# import sys
# import os
# from typing import Dict, Any, List, TYPE_CHECKING, Optional
# import matplotlib.pyplot as plt 
# import matplotlib 

# # --- 统一的导入路径管理 ---
# _utility_calc_dir = os.path.dirname(os.path.abspath(__file__))
# _project_root_dir_util = os.path.dirname(_utility_calc_dir) 
# if _project_root_dir_util not in sys.path:
#     sys.path.insert(0, _project_root_dir_util)
# # --- 路径管理结束 ---

# from common.delta_v_estimator import DeltaVEstimatorMPC
# from common.dynamics import CWEquation 

# if TYPE_CHECKING: 
#     from strong_communication.belief_manager import BeliefManager 

# class UtilityCalculator:
#     def __init__(self,
#                  delta_v_estimator: DeltaVEstimatorMPC,
#                  task_definitions: Dict[str, Dict[str, Any]], 
#                  spacecraft_params: Dict[str, Dict[str, Any]] 
#                 ):
#         self.delta_v_estimator = delta_v_estimator
#         self.task_definitions = task_definitions # This should be kept updated by the environment/KSC
#         self.spacecraft_params = spacecraft_params
#         self.g0 = 9.80665

#     def calculate_fuel_mass_cost(self, spacecraft_id: str, delta_v_mps: float) -> float:
#         if delta_v_mps == np.inf:
#             return np.inf
        
#         sc_params = self.spacecraft_params.get(spacecraft_id)
#         if not sc_params:
#             # print(f"    [FuelCalc WARNING] 未找到航天器 {spacecraft_id} 的参数。")
#             return np.inf # Or raise error

#         m0 = sc_params.get('initial_mass_kg')
#         isp = sc_params.get('isp_s')

#         if m0 is None or isp is None:
#             # print(f"    [FuelCalc WARNING] 航天器 {spacecraft_id} 缺少 initial_mass_kg 或 isp_s 参数。")
#             return np.inf
#         if isp <= 1e-6: # 比冲必须为正
#             # print(f"    [FuelCalc WARNING] 航天器 {spacecraft_id} Isp过小或为零 ({isp}).")
#             return np.inf 

#         fuel_mass = m0 * (1 - np.exp(-delta_v_mps / (isp * self.g0)))
#         return fuel_mass

#     def get_expected_task_utility_for_agent(self,
#                                            agent_id: str,
#                                            current_agent_state: np.ndarray, 
#                                            task_id: str,
#                                            agent_task_beliefs: Optional[np.ndarray], 
#                                            task_type_names: List[str] 
#                                            ) -> float:
#         task_def = self.task_definitions.get(task_id) # Use the up-to-date task_definitions
#         sc_params = self.spacecraft_params.get(agent_id)

#         if not task_def or not sc_params:
#             # print(f"    [UtilCalc DEBUG get_expected] Task {task_id} or SC {agent_id} def not found. TaskDef keys: {list(self.task_definitions.keys()) if self.task_definitions else 'None'}")
#             return -np.inf

#         if task_def.get("status") == "completed" or task_def.get("current_completion_percentage", 0.0) >= 100.0:
#             # print(f"    [UtilCalc DEBUG get_expected] Task {task_id} is COMPLETED. Utility for assignment = -inf.")
#             return -np.inf 

#         expected_reward = 0.0
#         task_rewards_by_type = task_def.get('type_rewards', {})
#         if agent_task_beliefs is not None and len(task_type_names) == len(agent_task_beliefs) and self.task_type_names:
#             for i, type_name in enumerate(self.task_type_names): # Use self.task_type_names for consistency
#                 reward_value = task_rewards_by_type.get(type_name, 0.0) 
#                 expected_reward += agent_task_beliefs[i] * reward_value
#         elif agent_task_beliefs is None:
#              # print(f"    [UtilCalc WARNING get_expected] Agent {agent_id} has NO beliefs for task {task_id}. Expected reward = 0.")
#              pass # expected_reward remains 0
#         else: # Mismatch in belief/type_names length
#             # print(f"    [UtilCalc ERROR get_expected] Task type names len ({len(task_type_names)}) vs beliefs len ({len(agent_task_beliefs)}) mismatch for task {task_id}.")
#             return -np.inf
#         # print(f"      [UtilCalc DEBUG get_expected] Task {task_id}, Agent {agent_id}: Expected Reward = {expected_reward:.2f}")

#         expected_risk_cost = 0.0
#         task_risks_by_type = task_def.get('type_risks', {})
#         risk_aversion = sc_params.get('risk_aversion_factor', 1.0)
#         if agent_task_beliefs is not None and len(self.task_type_names) == len(agent_task_beliefs) and self.task_type_names:
#             for i, type_name in enumerate(self.task_type_names):
#                 risk_value = task_risks_by_type.get(type_name, 0.0)
#                 expected_risk_cost += agent_task_beliefs[i] * risk_value * risk_aversion
#         # print(f"      [UtilCalc DEBUG get_expected] Task {task_id}, Agent {agent_id}: Expected Risk Cost = {expected_risk_cost:.2f}")
        
#         task_position = task_def.get('position')
#         if task_position is None:
#             # print(f"    [UtilCalc ERROR get_expected] Task {task_id} has no position defined. Returning -inf.")
#             return -np.inf
        
#         target_velocity_nominal = np.array([0.0] * self.delta_v_estimator.control_size) # Default to zero velocity if not specified
#         target_velocity = task_def.get('target_velocity', target_velocity_nominal)
        
#         # Ensure target_velocity has the correct dimension (control_size for estimator)
#         if not isinstance(target_velocity, np.ndarray) or len(target_velocity) != self.delta_v_estimator.control_size:
#             target_velocity = target_velocity_nominal
        
#         # Construct target state [pos, vel] for DeltaV estimator
#         # Ensure position part matches the state_size//2 expected by estimator (usually 2 for [x,y])
#         pos_part_len = self.delta_v_estimator.state_size // 2
#         vel_part_len = self.delta_v_estimator.state_size - pos_part_len

#         try:
#             pos_for_target = np.array(task_position)[:pos_part_len]
#             vel_for_target = np.array(target_velocity)[:vel_part_len] # Should match control_size
#             if len(pos_for_target) != pos_part_len or len(vel_for_target) != vel_part_len:
#                  raise ValueError("Position or velocity part length mismatch for target state.")
#             target_task_state = np.concatenate((pos_for_target, vel_for_target))
#         except Exception as e_state_build:
#             # print(f"    [UtilCalc ERROR get_expected] Failed to build target_task_state for {task_id}: {e_state_build}. Returning -inf.")
#             return -np.inf

#         if len(target_task_state) != self.delta_v_estimator.state_size:
#             # print(f"    [UtilCalc ERROR get_expected] Constructed target_task_state for {task_id} has wrong dimension {len(target_task_state)}. Expected {self.delta_v_estimator.state_size}. Returning -inf.")
#             return -np.inf

#         delta_v, _, success = self.delta_v_estimator.estimate_delta_v(current_agent_state, target_task_state)
#         # print(f"      [UtilCalc DEBUG get_expected] Task {task_id}, Agent {agent_id}: Delta-V Est - Success: {success}, Value: {'Inf' if delta_v == np.inf else delta_v:.3f} m/s")
        
#         if not success or delta_v == np.inf:
#             return -np.inf

#         fuel_cost_mass = self.calculate_fuel_mass_cost(agent_id, delta_v)
#         if fuel_cost_mass == np.inf:
#             return -np.inf

#         fuel_cost_coeff = sc_params.get('fuel_cost_coeff', 1.0) # Get fuel cost coefficient
#         fuel_utility_cost = fuel_cost_mass * fuel_cost_coeff
#         # print(f"      [UtilCalc DEBUG get_expected] Task {task_id}, Agent {agent_id}: Fuel Mass = {fuel_cost_mass:.3f} kg, Coeff: {fuel_cost_coeff}, Fuel Util Cost = {fuel_utility_cost:.2f}")

#         total_utility = expected_reward - expected_risk_cost - fuel_utility_cost
#         # print(f"    [UtilCalc INFO get_expected] Task {task_id}, Agent {agent_id}: Calc Total Util = {total_utility:.2f} (R:{expected_reward:.1f}-RiskC:{expected_risk_cost:.1f}-FuelC:{fuel_utility_cost:.1f})")
        
#         return total_utility

#     def calculate_coalition_task_utility(self, # (保持与您上传版本一致)
#                                          coalition_agent_ids: List[str],
#                                          current_coalition_states: Dict[str, np.ndarray], 
#                                          task_id: str,
#                                          belief_manager: 'BeliefManager', 
#                                          task_type_names: List[str]
#                                          ) -> float:
#         if not coalition_agent_ids:
#             return -np.inf
#         max_individual_utility_in_coalition = -np.inf
#         for agent_id in coalition_agent_ids:
#             current_state = current_coalition_states.get(agent_id)
#             if current_state is None: continue
#             agent_beliefs_for_task = belief_manager.get_agent_belief_for_task(agent_id, task_id) # KSC Belief manager
#             if agent_beliefs_for_task is None: continue

#             utility = self.get_expected_task_utility_for_agent(
#                 agent_id, current_state, task_id, agent_beliefs_for_task, task_type_names
#             )
#             if utility > max_individual_utility_in_coalition:
#                 max_individual_utility_in_coalition = utility
#         return max_individual_utility_in_coalition

# # --- 模块自测试与可视化 (保持与您上传版本一致) ---
# # (if __name__ == '__main__': test_utility_calculator() ...)
# strong_communication/utility_calculator.py
import numpy as np
import sys
import os
from typing import Dict, Any, List, TYPE_CHECKING, Optional
import matplotlib.pyplot as plt 
import matplotlib 

# --- 统一的导入路径管理 ---
_utility_calc_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_dir_util = os.path.dirname(_utility_calc_dir) 
if _project_root_dir_util not in sys.path:
    sys.path.insert(0, _project_root_dir_util)
# --- 路径管理结束 ---

from common.delta_v_estimator import DeltaVEstimatorMPC
from common.dynamics import CWEquation 

if TYPE_CHECKING: 
    from strong_communication.belief_manager import BeliefManager 

class UtilityCalculator:
    def __init__(self,
                 delta_v_estimator: DeltaVEstimatorMPC,
                 task_definitions: Dict[str, Dict[str, Any]], 
                 spacecraft_params: Dict[str, Dict[str, Any]]
                ):
        self.delta_v_estimator = delta_v_estimator
        self.task_definitions = task_definitions # This will be updated by the environment
        self.spacecraft_params = spacecraft_params
        self.g0 = 9.80665  

    def calculate_fuel_mass_cost(self, spacecraft_id: str, delta_v_mps: float) -> float:
        if delta_v_mps == np.inf:
            return np.inf
        
        sc_params = self.spacecraft_params.get(spacecraft_id)
        if not sc_params:
            # print(f"    [FuelCalc DEBUG] 未找到航天器 {spacecraft_id} 的参数。")
            return np.inf # Or raise error

        m0 = sc_params.get('initial_mass_kg')
        isp = sc_params.get('isp_s')

        if m0 is None or isp is None:
            # print(f"    [FuelCalc DEBUG] 航天器 {spacecraft_id} 缺少 initial_mass_kg 或 isp_s 参数。")
            return np.inf
        if isp <= 0: 
            return np.inf 

        fuel_mass = m0 * (1 - np.exp(-delta_v_mps / (isp * self.g0 + 1e-9))) # Add epsilon to avoid div by zero if g0 or isp is zero
        return fuel_mass

    def get_expected_task_utility_for_agent(self,
                                           agent_id: str,
                                           current_agent_state: np.ndarray, 
                                           task_id: str,
                                           agent_task_beliefs: Optional[np.ndarray], 
                                           task_type_names: List[str]
                                           ) -> float:
        # print(f"  [UtilCalc DEBUG get_expected_task_utility] Agent: {agent_id}, Task: {task_id}")
        task_def = self.task_definitions.get(task_id) # Use the live task_definitions
        sc_params = self.spacecraft_params.get(agent_id)

        if not task_def or not sc_params:
            # print(f"    [UtilCalc WARNING] Task {task_id} (Def: {'Found' if task_def else 'Not Found'}) or SC {agent_id} (Params: {'Found' if sc_params else 'Not Found'}) def not found. Returning -inf.")
            return -np.inf
        
        if task_def.get("status") == "completed" or task_def.get("current_completion_percentage", 0.0) >= 100.0:
            # print(f"    [UtilCalc INFO] Task {task_id} is already completed. Utility for assignment is -inf.")
            return -np.inf

        expected_reward = 0.0
        task_rewards_by_type = task_def.get('type_rewards', {})
        if agent_task_beliefs is None:
            # print(f"    [UtilCalc WARNING] Agent {agent_id} has NO beliefs for task {task_id}. Assuming zero reward & risk.")
            pass # Reward and risk will be 0
        elif len(task_type_names) != len(agent_task_beliefs):
            # print(f"    [UtilCalc ERROR] Task type names len ({len(task_type_names)}) != beliefs len ({len(agent_task_beliefs)}) for task {task_id}. Returning -inf.")
            return -np.inf
        else:
            for i, type_name in enumerate(task_type_names):
                reward_value = task_rewards_by_type.get(type_name, 0.0) 
                expected_reward += agent_task_beliefs[i] * reward_value
        # print(f"      [UtilCalc DEBUG] ER for {agent_id} to {task_id}: {expected_reward:.2f} (Beliefs: {agent_task_beliefs}, Rewards: {task_rewards_by_type})")

        expected_risk_cost = 0.0
        task_risks_by_type = task_def.get('type_risks', {})
        risk_aversion = sc_params.get('risk_aversion_factor', 1.0)
        if agent_task_beliefs is not None and len(task_type_names) == len(agent_task_beliefs):
            for i, type_name in enumerate(task_type_names):
                risk_value = task_risks_by_type.get(type_name, 0.0)
                expected_risk_cost += agent_task_beliefs[i] * risk_value * risk_aversion
        # print(f"      [UtilCalc DEBUG] ERiskC for {agent_id} to {task_id}: {expected_risk_cost:.2f} (Risks: {task_risks_by_type})")

        task_position = task_def.get('position')
        if task_position is None:
            # print(f"    [UtilCalc ERROR] Task {task_id} has no position. Returning -inf.")
            return -np.inf
        
        target_velocity_raw = task_def.get('target_velocity', np.array([0.0, 0.0]))
        # Ensure target_velocity has correct dimension for concatenation
        if len(target_velocity_raw) != self.delta_v_estimator.control_size :
             target_velocity = np.array([0.0] * self.delta_v_estimator.control_size)
        else:
             target_velocity = np.array(target_velocity_raw)

        # Ensure task_position is also correctly dimensioned for concatenation
        pos_dims_needed = self.delta_v_estimator.state_size - len(target_velocity)
        if len(task_position) != pos_dims_needed:
            # print(f"    [UtilCalc ERROR] Task {task_id} position has {len(task_position)} dims, need {pos_dims_needed}. Returning -inf.")
            return -np.inf
        
        target_task_state = np.concatenate((np.array(task_position)[:pos_dims_needed], target_velocity))
        if len(target_task_state) != self.delta_v_estimator.state_size:
            # print(f"    [UtilCalc ERROR] Constructed target_task_state for {task_id} has wrong dimension {len(target_task_state)} vs {self.delta_v_estimator.state_size}. Returning -inf.")
            return -np.inf
        
        # print(f"      [UtilCalc DEBUG] Estimating Delta-V for {agent_id} from {np.round(current_agent_state[:2],1)} to {np.round(target_task_state[:2],1)} (Task {task_id})")
        delta_v, _, success = self.delta_v_estimator.estimate_delta_v(current_agent_state, target_task_state)
        
        if not success or delta_v == np.inf:
            # print(f"    [UtilCalc WARNING] Delta-V estimation failed or inf for agent {agent_id} to task {task_id}. Success: {success}, DV: {delta_v}. Returning -inf.")
            return -np.inf

        fuel_cost_mass = self.calculate_fuel_mass_cost(agent_id, delta_v)
        if fuel_cost_mass == np.inf:
            # print(f"    [UtilCalc WARNING] Infinite fuel mass for agent {agent_id} to task {task_id} (DV: {delta_v}). Returning -inf.")
            return -np.inf

        fuel_utility_cost = fuel_cost_mass * sc_params.get('fuel_cost_coeff', 1.0) # Ensure fuel_cost_coeff is in sc_params
        # print(f"      [UtilCalc DEBUG] Fuel cost for {agent_id} to {task_id}: DV={delta_v:.2f}, Mass={fuel_cost_mass:.3f}, UtilCost={fuel_utility_cost:.2f} (Coeff: {sc_params.get('fuel_cost_coeff', 1.0)})")

        total_utility = expected_reward - expected_risk_cost - fuel_utility_cost
        # print(f"    [UtilCalc INFO] Final Utility for {agent_id} to {task_id}: {total_utility:.2f} = ER({expected_reward:.1f}) - ERiskC({expected_risk_cost:.1f}) - FCost({fuel_utility_cost:.1f})")
        
        # If total_utility is extremely small and positive, KSC might still prefer it over 0 (idle).
        # This check might be too aggressive, but can help if all tasks yield near-zero or negative utility.
        # if 0 < total_utility < 1e-3 and expected_reward > 1 : # If reward was high but utility is tiny due to cost
        #     print(f"    [UtilCalc DEBUG NOTE] Tiny positive utility {total_utility:.3f} for {agent_id} to {task_id}. ER was {expected_reward:.1f}.")
        
        return total_utility

    def calculate_coalition_task_utility(self, # (保持不变)
                                         coalition_agent_ids: List[str],
                                         current_coalition_states: Dict[str, np.ndarray], 
                                         task_id: str,
                                         belief_manager: 'BeliefManager', 
                                         task_type_names: List[str]
                                         ) -> float:
        if not coalition_agent_ids: return -np.inf
        max_individual_utility_in_coalition = -np.inf
        for agent_id in coalition_agent_ids:
            current_state = current_coalition_states.get(agent_id)
            if current_state is None: continue
            agent_beliefs_for_task = belief_manager.get_agent_belief_for_task(agent_id, task_id)
            # if agent_beliefs_for_task is None: continue # Handled by get_expected_task_utility_for_agent

            utility = self.get_expected_task_utility_for_agent(
                agent_id, current_state, task_id, agent_beliefs_for_task, task_type_names
            )
            if utility > max_individual_utility_in_coalition:
                max_individual_utility_in_coalition = utility
        return max_individual_utility_in_coalition

# ... (模块自测试与可视化 test_utility_calculator 保持不变) ...