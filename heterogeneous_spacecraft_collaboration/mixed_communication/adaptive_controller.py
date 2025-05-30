# # mixed_communication/adaptive_controller.py
# import numpy as np
# from typing import List, Dict, Tuple, Any, Optional
# import sys
# import os
# # --- 路径管理 (保持不变) ---
# _current_file_directory = os.path.dirname(os.path.abspath(__file__))
# _project_root_directory = os.path.dirname(_current_file_directory)
# if _project_root_directory not in sys.path:
#     sys.path.insert(0, _project_root_directory)

# from common.dynamics import CWEquation
# from common.spectral_analysis import SpectralAnalyzer # AIF会用
# from common.delta_v_estimator import DeltaVEstimatorMPC # 不再用于强模式下的主要目标生成
# from strong_communication.ksc_coalition_formation import KSCCoalitionFormation, IDLE_TASK_ID as KSC_IDLE_TASK_ID
# from weak_communication.active_inference import ActiveInferenceAgent
# import weak_communication.aif_functions_isobeliefs_convergent as aif_core


# class AdaptiveController:
#     def __init__(self,
#                  agent_id_str: str,
#                  all_agent_ids_ordered: List[str],
#                  ksc_algorithm_instance: Optional[KSCCoalitionFormation],
#                  delta_v_estimator_for_strong_nav: Optional[DeltaVEstimatorMPC], # 或可移除
#                  task_definitions_for_ksc: Optional[Dict[str, Dict[str, Any]]],
#                  agent_physical_params: Dict,
#                  all_agents_physical_params_list: List[Dict],
#                  aif_goal_positions_list: List[np.ndarray],
#                  global_aif_hyperparams: Dict,
#                  dynamics_model_for_agent: CWEquation,
#                  spectral_analysis_horizon_H: int,
#                  csi_threshold_low: float = 0.3,
#                  csi_threshold_high: float = 0.7,
#                  csi_ema_alpha: float = 0.2,
#                  spectral_analysis_alpha_scale: float = 1.0,
#                  spectral_control_norm_matrix: Optional[np.ndarray] = None,
#                  num_spectral_modes_to_consider: Optional[int] = None,
#                  allow_multiple_to_same_goal_for_aif_g: bool = False
#                  ):
#         self.agent_id_str = agent_id_str
#         self.all_agent_ids_ordered = list(all_agent_ids_ordered)
#         try:
#             self.agent_idx_internal = self.all_agent_ids_ordered.index(agent_id_str)
#         except ValueError:
#             raise ValueError(f"AdaptiveController: Agent ID {agent_id_str} not found in all_agent_ids_ordered list.")
#         self.num_agents = len(all_agent_ids_ordered)
#         self.state_size = dynamics_model_for_agent.state_size

#         self.ksc_algorithm = ksc_algorithm_instance
#         # self.delta_v_estimator_strong_nav = delta_v_estimator_for_strong_nav # 如果完全不用可以注释掉
#         self.task_definitions_ksc = task_definitions_for_ksc # 这应该是动态更新的

#         self.active_inference_agent = ActiveInferenceAgent( # AIF部分保持不变
#             agent_id_str=agent_id_str,
#             all_agent_ids_ordered=all_agent_ids_ordered,
#             agent_physical_params=agent_physical_params,
#             all_agents_physical_params_list=all_agents_physical_params_list,
#             goal_positions_list=aif_goal_positions_list,
#             global_aif_hyperparams=global_aif_hyperparams,
#             dynamics_model_for_agent=dynamics_model_for_agent,
#             spectral_analysis_horizon_H=spectral_analysis_horizon_H,
#             spectral_analysis_alpha_scale=spectral_analysis_alpha_scale,
#             spectral_control_norm_matrix=spectral_control_norm_matrix,
#             num_spectral_modes_to_consider=num_spectral_modes_to_consider,
#             allow_multiple_to_same_goal_for_g=allow_multiple_to_same_goal_for_aif_g
#         )
#         # spectral_analyzer_for_strong_nav 不再需要用于目标生成，除非有其他用途
#         # self.spectral_analyzer_for_strong_nav = SpectralAnalyzer(...)

#         self.csi_low = csi_threshold_low
#         self.csi_high = csi_threshold_high
#         self.csi_ema_alpha = csi_ema_alpha
#         self.current_csi_raw: float = 1.0
#         self.current_csi_smooth: float = 1.0
#         self.current_decision_mode: str = "strong"
#         print(f"[AdaptiveController '{self.agent_id_str}'] Initialized. CSI thresholds: L={self.csi_low}, H={self.csi_high}")

#     def _calculate_csi(self, environment_comm_data: Dict[str, Any]) -> float:
#         # (保持不变)
#         connected_neighbors = environment_comm_data.get("connected_neighbors", [])
#         num_neighbors = len(connected_neighbors)
#         max_possible_neighbors = self.num_agents - 1 if self.num_agents > 1 else 1
#         if max_possible_neighbors == 0: return 1.0
#         connectivity_factor = num_neighbors / max_possible_neighbors
#         avg_link_quality = 1.0
#         if connected_neighbors:
#             qualities = [environment_comm_data.get("link_qualities", {}).get(n_id, 0.5) for n_id in connected_neighbors]
#             if qualities: avg_link_quality = np.mean(qualities)
#         raw_csi = connectivity_factor * avg_link_quality
#         return np.clip(raw_csi, 0.0, 1.0)

#     def _update_decision_mode(self):
#         # (保持不变)
#         if self.current_csi_smooth >= self.csi_high:
#             self.current_decision_mode = "strong"
#         elif self.current_csi_smooth < self.csi_low:
#             self.current_decision_mode = "weak"
#             # 根据CSI微调AIF推理级别 (可选)
#             if self.current_csi_smooth < self.csi_low * 0.33: self.active_inference_agent.set_reasoning_mode(0, False)
#             elif self.current_csi_smooth < self.csi_low * 0.66: self.active_inference_agent.set_reasoning_mode(1, False)
#             else: self.active_inference_agent.set_reasoning_mode(2, True) # 假设默认高阶
#         else: self.current_decision_mode = "transition"

#     def _get_target_state_for_strong_mode(self, my_current_physical_state: np.ndarray) -> Optional[np.ndarray]:
#         """
#         在强通信模式下，获取KSC分配的任务的绝对目标状态。
#         如果未分配任务或任务无效，则返回None（表示保持当前状态或无特定目标）。
#         """
#         if self.ksc_algorithm is None or self.task_definitions_ksc is None:
#             return None # 或者 my_current_physical_state.copy() 如果希望保持

#         current_ksc_assignment = self.ksc_algorithm._sim_best_assignment
#         assigned_task_id = current_ksc_assignment.get(self.agent_id_str)

#         if assigned_task_id is None or assigned_task_id == KSC_IDLE_TASK_ID:
#             return None # 或者 my_current_physical_state.copy()

#         task_def = self.task_definitions_ksc.get(assigned_task_id)
#         if not task_def:
#             return None # 或者 my_current_physical_state.copy()

#         # 获取任务的当前绝对状态 (对于动态任务，这是其 current_state)
#         # 对于静态任务，其 current_state 就是其 initial_state 或定义的固定状态
#         task_target_state_abs_list = task_def.get("current_state", task_def.get("initial_state"))
#         if task_target_state_abs_list is None: return None

#         task_target_state_abs = np.array(task_target_state_abs_list)

#         # 确保状态向量是完整的 (例如, 4D for [x,y,vx,vy])
#         if task_target_state_abs.shape[0] < self.state_size:
#             # 如果只定义了位置，速度通常设为0，或者任务定义的target_velocity
#             pos = np.array(task_def.get("position", task_target_state_abs[:2] if len(task_target_state_abs)>=2 else [0,0] ))
#             vel = np.array(task_def.get("target_velocity", [0,0]*(self.state_size//2) if len(pos)==self.state_size//2 else [0,0]))
#             if len(pos) + len(vel) == self.state_size:
#                  task_target_state_abs = np.concatenate((pos, vel))
#             else: # 无法构建完整状态
#                 return None # 或者 my_current_physical_state.copy()

#         return task_target_state_abs


#     def decide_action(self,
#                       all_agent_true_states_global: List[np.ndarray],
#                       all_agent_aif_params_list: List[Dict],
#                       environment_comm_data: Dict[str, Any]
#                      ) -> Tuple[Optional[np.ndarray], str, Dict[str, Any]]:
#         """
#         根据当前模式决定动作。
#         强模式: 返回KSC任务的绝对目标状态。
#         弱模式: 返回由AIF和谱分析决定的目标状态偏差 delta_s。
#         过渡模式: 融合强模式的绝对目标和弱模式的 delta_s。
#                   为了让MPC能统一处理，过渡模式最终也应输出一个MPC能理解的目标。
#                   这里我们将弱模式的 delta_s 转换为绝对目标 (current + delta_s)，然后加权。
#         """
#         self.current_csi_raw = self._calculate_csi(environment_comm_data)
#         self.current_csi_smooth = (1 - self.csi_ema_alpha) * self.current_csi_smooth + \
#                                   self.csi_ema_alpha * self.current_csi_raw
#         self.current_csi_smooth = np.clip(self.current_csi_smooth, 0.0, 1.0)
#         self._update_decision_mode()

#         final_mpc_target_state = None # 这是MPC在N步后要达到的绝对状态
#         debug_info = {"csi_raw": self.current_csi_raw, "csi_smooth": self.current_csi_smooth,
#                       "mode": self.current_decision_mode, "agent_id": self.agent_id_str}
#         my_current_physical_state = all_agent_true_states_global[self.agent_idx_internal]

#         if self.current_decision_mode == "strong":
#             # 强模式直接获取任务的绝对目标状态
#             strong_mode_target_abs = self._get_target_state_for_strong_mode(my_current_physical_state)
#             if strong_mode_target_abs is not None:
#                 final_mpc_target_state = strong_mode_target_abs
#                 debug_info["strong_mode_target_abs_pos"] = final_mpc_target_state[:2].tolist()
#             else: # 未分配任务或任务无效，则目标是保持当前状态
#                 final_mpc_target_state = my_current_physical_state.copy()
#                 debug_info["strong_mode_target_abs_pos"] = "hold_current"

#         elif self.current_decision_mode == "weak":
#             # 弱模式获取 delta_s，然后转换为MPC的绝对目标
#             selected_ds_weak, efe_val, _, q_g = self.active_inference_agent.decide_action(
#                 all_agent_true_states_global, all_agent_aif_params_list)
#             if selected_ds_weak is not None:
#                 final_mpc_target_state = my_current_physical_state + selected_ds_weak
#                 debug_info["weak_mode_delta_s_pos"] = selected_ds_weak[:2].tolist()
#             else: # AIF未选择动作，保持当前
#                 final_mpc_target_state = my_current_physical_state.copy()
#                 debug_info["weak_mode_delta_s_pos"] = "zeros"
#             debug_info["weak_mode_efe"] = efe_val

#         elif self.current_decision_mode == "transition":
#             # 过渡模式：融合强模式的绝对目标和弱模式计算出的绝对目标
#             strong_target_abs = self._get_target_state_for_strong_mode(my_current_physical_state)
#             if strong_target_abs is None: strong_target_abs = my_current_physical_state.copy() # 未分配则保持

#             delta_s_weak_cand, _, _, _ = self.active_inference_agent.decide_action(
#                 all_agent_true_states_global, all_agent_aif_params_list)
#             weak_target_abs = my_current_physical_state + (delta_s_weak_cand if delta_s_weak_cand is not None else np.zeros(self.state_size))

#             if self.csi_high == self.csi_low : # 避免除零
#                 weight_strong = 1.0 if self.current_csi_smooth >= self.csi_high else 0.0
#             else:
#                 weight_strong = (self.current_csi_smooth - self.csi_low) / (self.csi_high - self.csi_low)
#             weight_strong = np.clip(weight_strong, 0.0, 1.0)
#             weight_weak = 1.0 - weight_strong

#             final_mpc_target_state = weight_strong * strong_target_abs + weight_weak * weak_target_abs
#             debug_info["trans_strong_target_abs_pos"] = strong_target_abs[:2].tolist()
#             debug_info["trans_weak_target_abs_pos"] = weak_target_abs[:2].tolist()
#             debug_info["trans_weight_strong"] = weight_strong

#         if final_mpc_target_state is None: # 确保总有返回值
#             final_mpc_target_state = my_current_physical_state.copy() # 默认保持当前状态

#         # 为了与 SpacecraftAgentSim.get_decision_delta_s 的原始期望（返回delta_s）兼容，
#         # 我们可以计算这个最终绝对目标相对于当前状态的偏差并返回。
#         # SpacecraftAgentSim 之后会 current_state + delta_s 来得到MPC目标，结果不变。
#         final_delta_s_output = final_mpc_target_state

#         # 记录KSC的目标任务ID，如果是在强或过渡模式且KSC有分配
#         if self.ksc_algorithm and (self.current_decision_mode == "strong" or self.current_decision_mode == "transition"):
#             assigned_task_id = self.ksc_algorithm._sim_best_assignment.get(self.agent_id_str)
#             if assigned_task_id and assigned_task_id != KSC_IDLE_TASK_ID:
#                  debug_info["ksc_target_task_id"] = assigned_task_id


#         return final_delta_s_output, self.current_decision_mode, debug_info

# # --- 模块自测试 (保持不变，但其内部对 _get_delta_s_for_strong_mode 的理解需要更新) ---
# # test_adaptive_controller 函数需要更新，因为它可能期望 _get_delta_s_for_strong_mode 返回 delta_s
# # 而现在我们让 decide_action 返回 delta_s，但内部的 _get_target_state_for_strong_mode 返回绝对状态。
# # 为了简洁，这里的自测试将不再详细绘制 delta_s_x 和 delta_s_y，只关注模式切换。
# def test_adaptive_controller():
#     # ... (测试代码的设置部分可以保持，但对结果的期望和绘制需要调整) ...
#     print("--- 测试 AdaptiveController (新强模式目标逻辑) ---")
#     # ... (如之前的测试设置 num_agents, agent_ids, dynamics_instances, ksc_mock, mock_task_defs_ksc, dv_estimator)
#     num_agents = 3; num_tasks_strong = 2; num_goals_aif = 2
#     agent_ids = [f"AdptSC{i}" for i in range(num_agents)]
#     n_orb = 2 * np.pi / (3600*1.5); dt_ctrl = 20.0
#     dynamics_instances = [CWEquation(n=n_orb, dt=dt_ctrl) for _ in range(num_agents)]
#     mock_ksc_assignment = {agent_ids[0]: "T1_ksc", agent_ids[1]: "T2_ksc", agent_ids[2]: KSC_IDLE_TASK_ID}
#     class MockKSC:
#         def __init__(self, assignment): self._sim_best_assignment = assignment; self.task_ids_actual = ["T1_ksc", "T2_ksc"]
#     ksc_mock_instance = MockKSC(mock_ksc_assignment)
#     mock_task_defs_ksc = {
#         "T1_ksc": {"id": "T1_ksc", "position": np.array([200, 100]), "target_velocity": np.array([0,0]), "current_state": [200,100,0,0]},
#         "T2_ksc": {"id": "T2_ksc", "position": np.array([-200, -100]), "target_velocity": np.array([0,0]), "current_state": [-200,-100,0,0]}
#     }
#     agent_phys_params_list_test = [ {"type": 's'}] * num_agents
#     aif_target_goal_positions = [np.array([150.0, 80.0]), np.array([-150.0, -80.0])]
#     global_aif_hyper_test = {"reasoning_level": 1}
#     _possible_G_adpt_test = aif_core.generate_possible_goal_configs(num_agents, num_goals_aif, False)
#     if _possible_G_adpt_test.size > 0: global_aif_hyper_test["C_target_dist_G"] = np.ones(_possible_G_adpt_test.shape[0])/_possible_G_adpt_test.shape[0]
#     all_agents_aif_params_for_test = [ aif_core.initialize_agent_aif_params(
#             agent_ids[i], i, agent_ids, num_goals_aif, agent_phys_params_list_test[i],
#             aif_target_goal_positions, global_aif_hyper_test, allow_multiple_to_same_goal=False
#         ) for i in range(num_agents)]
#     controller = AdaptiveController(
#         agent_ids[0], agent_ids, ksc_mock_instance, None, mock_task_defs_ksc,
#         agent_phys_params_list_test[0], agent_phys_params_list_test, aif_target_goal_positions,
#         global_aif_hyper_test, dynamics_instances[0], 8, spectral_analysis_alpha_scale=30.0)

#     current_true_states_ctrl_test = [np.array([10,5,0.1,0]), np.array([-10,-5,-0.1,0]), np.array([0,15,0,0.1])]
#     print(f"\n--- 测试 {controller.agent_id_str} 的自适应决策 (CSI变化) ---")
#     for csi_val_test in np.linspace(1.0, 0.0, 5):
#         controller.current_csi_smooth = csi_val_test # 手动设置CSI
#         delta_s_out, mode_out, debug_out = controller.decide_action(current_true_states_ctrl_test, all_agents_aif_params_for_test, {})
#         print(f"  CSI: {csi_val_test:.2f} -> Mode: {mode_out}, Output Delta_S (pos): {delta_s_out[:2] if delta_s_out is not None else 'None'}")
#         if mode_out == "strong":
#             abs_target = controller._get_target_state_for_strong_mode(current_true_states_ctrl_test[0])
#             print(f"    Strong mode raw absolute target: {abs_target[:2] if abs_target is not None else 'None'}")
#         elif mode_out == "transition":
#             strong_target_abs_trans = debug_out.get("trans_strong_target_abs_pos", "N/A")
#             weak_target_abs_trans = debug_out.get("trans_weak_target_abs_pos", "N/A")
#             weight_s_trans = debug_out.get("trans_weight_strong", "N/A")
#             print(f"    Transition: StrongTarget={strong_target_abs_trans}, WeakTarget={weak_target_abs_trans}, WeightStrong={weight_s_trans:.2f}")

# if __name__ == '__main__':
#     test_adaptive_controller()



# mixed_communication/adaptive_controller.py
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import sys
import os

# --- 路径管理 (保持不变) ---
_current_file_directory = os.path.dirname(os.path.abspath(__file__))
_project_root_directory = os.path.dirname(os.path.dirname(_current_file_directory)) # adaptive_controller.py 在 mixed_communication 下
if _project_root_directory not in sys.path:
    sys.path.insert(0, _project_root_directory)

from common.dynamics import CWEquation
from common.spectral_analysis import SpectralAnalyzer
# from common.delta_v_estimator import DeltaVEstimatorMPC # 强模式不再用它生成主要目标
from strong_communication.ksc_coalition_formation import KSCCoalitionFormation, IDLE_TASK_ID as KSC_IDLE_TASK_ID
from weak_communication.active_inference import ActiveInferenceAgent
import weak_communication.aif_functions_isobeliefs_convergent as aif_core


class AdaptiveController:
    def __init__(self,
                 agent_id_str: str,
                 all_agent_ids_ordered: List[str],
                 ksc_algorithm_instance: Optional[KSCCoalitionFormation],
                 delta_v_estimator_for_strong_nav: Optional[Any], # 不再是DeltaVEstimatorMPC类型
                 task_definitions_for_ksc: Optional[Dict[str, Dict[str, Any]]], # 会被环境的self.tasks覆盖
                 agent_physical_params: Dict,
                 all_agents_physical_params_list: List[Dict],
                 aif_goal_positions_list: List[np.ndarray],
                 global_aif_hyperparams: Dict,
                 dynamics_model_for_agent: CWEquation,
                 spectral_analysis_horizon_H: int,
                 csi_threshold_low: float = 0.3,
                 csi_threshold_high: float = 0.7,
                 csi_ema_alpha: float = 0.2,
                 spectral_analysis_alpha_scale: float = 1.0, # AIF用
                 spectral_control_norm_matrix: Optional[np.ndarray] = None, # AIF用
                 num_spectral_modes_to_consider: Optional[int] = None, # AIF用
                 allow_multiple_to_same_goal_for_aif_g: bool = False # AIF用
                 ):
        self.agent_id_str = agent_id_str
        self.all_agent_ids_ordered = list(all_agent_ids_ordered)
        try:
            self.agent_idx_internal = self.all_agent_ids_ordered.index(agent_id_str)
        except ValueError:
            raise ValueError(f"AdaptiveController: Agent ID {agent_id_str} not found in all_agent_ids_ordered list.")
        self.num_agents = len(all_agent_ids_ordered)
        self.state_size = dynamics_model_for_agent.state_size

        self.ksc_algorithm = ksc_algorithm_instance
        self.task_definitions_for_ksc = task_definitions_for_ksc # 重要: 这个会被环境用最新的task dict覆盖

        self.active_inference_agent = ActiveInferenceAgent(
            agent_id_str=agent_id_str,
            all_agent_ids_ordered=all_agent_ids_ordered,
            agent_physical_params=agent_physical_params,
            all_agents_physical_params_list=all_agents_physical_params_list,
            goal_positions_list=aif_goal_positions_list,
            global_aif_hyperparams=global_aif_hyperparams,
            dynamics_model_for_agent=dynamics_model_for_agent,
            spectral_analysis_horizon_H=spectral_analysis_horizon_H,
            spectral_analysis_alpha_scale=spectral_analysis_alpha_scale,
            spectral_control_norm_matrix=spectral_control_norm_matrix,
            num_spectral_modes_to_consider=num_spectral_modes_to_consider,
            allow_multiple_to_same_goal_for_g=allow_multiple_to_same_goal_for_aif_g
        )

        # 在 AdaptiveController.__init__ 方法中添加以下代码
        # 在 self.active_inference_agent = ... 之后

        # 检查AIF目标配置，如果为空则使用任务位置作为备选
        if aif_goal_positions_list is None or len(aif_goal_positions_list) == 0:
            print(f"[AC DEBUG __init__ {agent_id_str}] 未找到AIF目标位置，尝试使用任务位置作为备选。")
            # 如果有任务定义，则使用任务位置
            if task_definitions_for_ksc and len(task_definitions_for_ksc) > 0:
                task_positions = [np.array(task_def["position"]) for task_id, task_def in task_definitions_for_ksc.items() 
                                if "position" in task_def]
                if task_positions:
                    # 使用任务位置作为AIF目标位置
                    aif_goal_positions_list = task_positions
                    print(f"[AC DEBUG __init__ {agent_id_str}] 使用 {len(aif_goal_positions_list)} 个任务位置作为AIF目标。")
                    
                    # 重新生成可能目标配置
                    if self.active_inference_agent is not None:
                        print(f"[AC DEBUG __init__ {agent_id_str}] 更新AIF智能体的目标位置配置。")
                        self.active_inference_agent.goal_positions = aif_goal_positions_list
                        self.active_inference_agent.possible_goal_configs = aif_core.generate_possible_goal_configs(
                            len(all_agent_ids_ordered), 
                            len(aif_goal_positions_list),
                            allow_multiple_to_same_goal_for_aif_g
                        )
                        if len(self.active_inference_agent.possible_goal_configs) == 0:
                            print(f"警告: 即使使用任务位置作为备选，智能体 '{agent_id_str}' 仍没有有效的目标配置。")
                            

        self.csi_low = csi_threshold_low
        self.csi_high = csi_threshold_high
        self.csi_ema_alpha = csi_ema_alpha
        self.current_csi_raw: float = 1.0
        self.current_csi_smooth: float = 1.0
        self.current_decision_mode: str = "strong"
        print(f"[AC DEBUG __init__ {self.agent_id_str}] Initialized. CSI thresholds: L={self.csi_low}, H={self.csi_high}")

    def _calculate_csi(self, environment_comm_data: Dict[str, Any]) -> float:
        # (与您上传版本一致)
        connected_neighbors = environment_comm_data.get("connected_neighbors", [])
        num_neighbors = len(connected_neighbors)
        max_possible_neighbors = self.num_agents - 1 if self.num_agents > 1 else 1 # handle single agent case
        if max_possible_neighbors == 0: return 1.0
        connectivity_factor = num_neighbors / max_possible_neighbors
        avg_link_quality = 1.0
        if connected_neighbors: # Only average if there are neighbors
            qualities = [environment_comm_data.get("link_qualities", {}).get(n_id, 0.5) for n_id in connected_neighbors] # default 0.5 if not found
            if qualities: avg_link_quality = np.mean(qualities)
        raw_csi = connectivity_factor * avg_link_quality
        return np.clip(raw_csi, 0.0, 1.0)


    def _update_decision_mode(self):
        # (与您上传版本一致)
        if self.current_csi_smooth >= self.csi_high:
            self.current_decision_mode = "strong"
        elif self.current_csi_smooth < self.csi_low:
            self.current_decision_mode = "weak"
            if self.active_inference_agent: # Ensure AIF agent exists
                # 您的代码中是直接设置，这里保持
                self.active_inference_agent.set_reasoning_mode(
                    reasoning_level=self.active_inference_agent.agent_core_params.get("default_reasoning_level", 1), # 使用AIF内部的默认值
                    use_epistemic_planning=self.active_inference_agent.agent_core_params.get("default_use_ep", True)
                )
        else:
            self.current_decision_mode = "transition"


    def _get_target_state_for_strong_mode(self, my_current_physical_state: np.ndarray) -> Optional[np.ndarray]:
        """
        强通信模式: 获取KSC分配任务的绝对目标状态。
        如果任务无效或已完成，返回None，表示应保持当前状态。
        """
        # print(f"  [AC DEBUG _get_target_strong {self.agent_id_str}] My current state (pos): {np.round(my_current_physical_state[:2],1)}")
        if self.ksc_algorithm is None or self.task_definitions_for_ksc is None:
            # print(f"    [AC DEBUG _get_target_strong {self.agent_id_str}] KSC instance or task_definitions_for_ksc is None. Returning None.")
            return None

        current_ksc_assignment = self.ksc_algorithm._sim_best_assignment
        assigned_task_id = current_ksc_assignment.get(self.agent_id_str)
        # print(f"    [AC DEBUG _get_target_strong {self.agent_id_str}] KSC assigned task ID: '{assigned_task_id}'")


        if assigned_task_id is None or assigned_task_id == KSC_IDLE_TASK_ID:
            # print(f"    [AC DEBUG _get_target_strong {self.agent_id_str}] Assigned IDLE or None. Returning None.")
            return None # 指示上层应保持当前状态

        task_def = self.task_definitions_for_ksc.get(assigned_task_id)
        if not task_def:
            print(f"    [AC CRITICAL _get_target_strong {self.agent_id_str}] Task def for '{assigned_task_id}' NOT FOUND in self.task_definitions_for_ksc (keys: {list(self.task_definitions_for_ksc.keys())}). Returning None.")
            return None
        
        if task_def.get("current_completion_percentage", 0.0) >= 100.0:
            # print(f"    [AC DEBUG _get_target_strong {self.agent_id_str}] Task '{assigned_task_id}' is already completed (completion: {task_def.get('current_completion_percentage')}%). Returning None.")
            return None # 任务已完成，不应再作为目标

        # 任务的绝对目标状态 (可能是动态的)
        task_target_state_abs_list = task_def.get("current_state", task_def.get("initial_state"))
        if task_target_state_abs_list is None:
            print(f"    [AC CRITICAL _get_target_strong {self.agent_id_str}] Task '{assigned_task_id}' current_state and initial_state are None. Returning None.")
            return None
        
        task_target_state_abs = np.array(task_target_state_abs_list)
        # print(f"    [AC DEBUG _get_target_strong {self.agent_id_str}] Task '{assigned_task_id}' raw target_state_abs_list (pos): {np.round(task_target_state_abs_list[:2],1)}")


        if task_target_state_abs.shape[0] != self.state_size: # 检查维度是否匹配
            # 尝试从 position 和 target_velocity 构建
            pos_list = task_def.get("position") # 这是任务的标称/初始位置
            vel_list = task_def.get("target_velocity") # 任务的期望速度
            
            # print(f"      Task '{assigned_task_id}' state size mismatch. Raw state shape: {task_target_state_abs.shape}. Expected: {self.state_size}")
            # print(f"      Trying to use 'position': {pos_list} and 'target_velocity': {vel_list}")

            if pos_list is not None and vel_list is not None:
                pos = np.array(pos_list)
                vel = np.array(vel_list)
                if pos.shape[0] == self.state_size // 2 and vel.shape[0] == self.state_size // 2:
                     task_target_state_abs = np.concatenate((pos, vel))
                     # print(f"      Reconstructed task_target_state_abs (pos): {np.round(task_target_state_abs[:2],1)}")
                else:
                    print(f"    [AC CRITICAL _get_target_strong {self.agent_id_str}] Failed to reconstruct full state for task '{assigned_task_id}' from pos/vel. Pos shape: {pos.shape}, Vel shape: {vel.shape}. Returning None.")
                    return None
            else:
                print(f"    [AC CRITICAL _get_target_strong {self.agent_id_str}] Task '{assigned_task_id}' pos or target_vel missing for full state reconstruction. Returning None.")
                return None
        
        # print(f"    [AC DEBUG _get_target_strong {self.agent_id_str}] Returning ABSOLUTE target state for task '{assigned_task_id}': Pos={np.round(task_target_state_abs[:2],1)}, Vel={np.round(task_target_state_abs[2:],2)}")
        return task_target_state_abs


    def decide_action(self,
                      all_agent_true_states_global: List[np.ndarray],
                      all_agent_aif_params_list: List[Dict],
                      environment_comm_data: Dict[str, Any]
                     ) -> Tuple[Optional[np.ndarray], str, Dict[str, Any]]:
        self.current_csi_raw = self._calculate_csi(environment_comm_data)
        self.current_csi_smooth = (1 - self.csi_ema_alpha) * self.current_csi_smooth + \
                                  self.csi_ema_alpha * self.current_csi_raw
        self.current_csi_smooth = np.clip(self.current_csi_smooth, 0.0, 1.0)
        self._update_decision_mode() # 更新 self.current_decision_mode

        final_mpc_target_state_abs: Optional[np.ndarray] = None
        debug_info = {"csi_raw": self.current_csi_raw, "csi_smooth": self.current_csi_smooth,
                      "mode": self.current_decision_mode, "agent_id": self.agent_id_str}
        my_current_physical_state = all_agent_true_states_global[self.agent_idx_internal]

        # print(f"  [AC DEBUG decide_action {self.agent_id_str}] Mode: {self.current_decision_mode}, MyPos: {np.round(my_current_physical_state[:2],1)}")

        if self.current_decision_mode == "strong":
            strong_mode_target_abs = self._get_target_state_for_strong_mode(my_current_physical_state)
            if strong_mode_target_abs is not None:
                final_mpc_target_state_abs = strong_mode_target_abs
                debug_info["strong_mode_target_abs_pos"] = final_mpc_target_state_abs[:2].tolist()
                # print(f"    [AC DEBUG decide_action {self.agent_id_str}] Strong mode target (ABS pos): {np.round(final_mpc_target_state_abs[:2],1)}")
            else:
                final_mpc_target_state_abs = my_current_physical_state.copy()
                debug_info["strong_mode_target_abs_pos"] = "hold_current"
                # print(f"    [AC DEBUG decide_action {self.agent_id_str}] Strong mode: No valid target, holding current state.")

        elif self.current_decision_mode == "weak":
            # print(f"    [AC DEBUG decide_action {self.agent_id_str}] Weak mode.")
            selected_ds_weak, efe_val, _, q_g = self.active_inference_agent.decide_action(
                all_agent_true_states_global, all_agent_aif_params_list)
            if selected_ds_weak is not None:
                final_mpc_target_state_abs = my_current_physical_state + selected_ds_weak
                debug_info["weak_mode_delta_s_pos"] = selected_ds_weak[:2].tolist()
                # print(f"      AIF selected delta_s (pos): {np.round(selected_ds_weak[:2],1)}. MPC target (ABS pos): {np.round(final_mpc_target_state_abs[:2],1)}")
            else:
                final_mpc_target_state_abs = my_current_physical_state.copy()
                debug_info["weak_mode_delta_s_pos"] = "zeros"
                # print(f"      AIF did not select delta_s. MPC target: holding current.")
            debug_info["weak_mode_efe"] = efe_val

        elif self.current_decision_mode == "transition":
            strong_target_abs_trans = self._get_target_state_for_strong_mode(my_current_physical_state)
            if strong_target_abs_trans is None: strong_target_abs_trans = my_current_physical_state.copy()

            delta_s_weak_cand, _, _, _ = self.active_inference_agent.decide_action(
                all_agent_true_states_global, all_agent_aif_params_list)
            weak_target_abs_trans = my_current_physical_state + (delta_s_weak_cand if delta_s_weak_cand is not None else np.zeros(self.state_size))

            if self.csi_high == self.csi_low :
                weight_strong = 1.0 if self.current_csi_smooth >= self.csi_high else 0.0
            else:
                weight_strong = (self.current_csi_smooth - self.csi_low) / (self.csi_high - self.csi_low)
            weight_strong = np.clip(weight_strong, 0.0, 1.0)
            weight_weak = 1.0 - weight_strong

            final_mpc_target_state_abs = weight_strong * strong_target_abs_trans + weight_weak * weak_target_abs_trans
            debug_info["trans_strong_target_abs_pos"] = strong_target_abs_trans[:2].tolist()
            debug_info["trans_weak_target_abs_pos"] = weak_target_abs_trans[:2].tolist()
            debug_info["trans_weight_strong"] = weight_strong
            # print(f"    [AC DEBUG decide_action {self.agent_id_str}] Transition mode. Blended MPC target (ABS pos): {np.round(final_mpc_target_state_abs[:2],1)}")

        if final_mpc_target_state_abs is None: # 理论上不应为None，因为有默认保持当前状态的逻辑
            # print(f"    [AC CRITICAL decide_action {self.agent_id_str}] final_mpc_target_state_abs is NONE. Defaulting to current state.")
            final_mpc_target_state_abs = my_current_physical_state.copy()

        # 计算并返回相对于当前状态的偏差 (名义上的delta_s)
        final_delta_s_output = final_mpc_target_state_abs - my_current_physical_state
        
        # print(f"    [AC DEBUG decide_action {self.agent_id_str}] Final output delta_s (pos): {np.round(final_delta_s_output[:2],1)}")

        # 记录KSC的目标任务ID
        if self.ksc_algorithm and (self.current_decision_mode == "strong" or self.current_decision_mode == "transition"):
            assigned_task_id_debug = self.ksc_algorithm._sim_best_assignment.get(self.agent_id_str)
            if assigned_task_id_debug and assigned_task_id_debug != KSC_IDLE_TASK_ID:
                 debug_info["ksc_target_task_id"] = assigned_task_id_debug
        
        return final_delta_s_output, self.current_decision_mode, debug_info

# --- 自测试 (暂时注释或确保它能处理新的返回逻辑) ---
# if __name__ == '__main__':
    # ... test_adaptive_controller() ...