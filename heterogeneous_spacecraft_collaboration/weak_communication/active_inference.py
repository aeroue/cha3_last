# weak_communication/active_inference.py
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import sys
import os
import matplotlib
import matplotlib.pyplot as plt


# --- 临时路径处理 ---
# 获取当前文件 (active_inference.py) 所在的目录 (weak_communication/)
_current_file_directory = os.path.dirname(os.path.abspath(__file__))
# 获取 weak_communication/ 目录的父目录 (heterogeneous_spacecraft_collaboration/)
_project_root_directory = os.path.dirname(_current_file_directory)

# 将项目根目录 (_project_root_directory) 添加到 sys.path 的开头
# 这样，像 import common.xyz 或 import weak_communication.xyz 这样的导入就能被正确解析
if _project_root_directory not in sys.path:
    sys.path.insert(0, _project_root_directory)
# --- 结束临时路径处理 ---

from common.spectral_analysis import SpectralAnalyzer
from common.dynamics import CWEquation # SpectralAnalyzer 和测试需要

# 现在可以安全地导入 aif_core 了
import weak_communication.aif_functions_isobeliefs_convergent as aif_core


class ActiveInferenceAgent:
    """
    代表一个在弱通信条件下使用主动推理进行决策的航天器。
    它内部封装了AIF核心逻辑的调用 (通过直接调用aif_core函数)
    以及用于动作生成的谱分析器。
    """
    def __init__(self,
                 agent_id_str: str, 
                 all_agent_ids_ordered: List[str],
                 agent_physical_params: Dict, 
                 all_agents_physical_params_list: List[Dict], 
                 goal_positions_list: List[np.ndarray],
                 global_aif_hyperparams: Dict,
                 dynamics_model_for_agent: CWEquation, 
                 spectral_analysis_horizon_H: int, 
                 spectral_analysis_alpha_scale: float = 1.0,
                 spectral_control_norm_matrix: Optional[np.ndarray] = None,
                 num_spectral_modes_to_consider: Optional[int] = None,
                 allow_multiple_to_same_goal_for_g: bool = False
                 ):
        self.agent_id_str = agent_id_str
        try:
            self.agent_idx_internal = all_agent_ids_ordered.index(agent_id_str)
        except ValueError:
            raise ValueError(f"Agent ID {agent_id_str} not found in all_agent_ids_ordered list.")

        self.num_agents = len(all_agent_ids_ordered)
        
        self.agent_core_params = aif_core.initialize_agent_aif_params(
            agent_id_str=self.agent_id_str,
            agent_idx_internal=self.agent_idx_internal,
            all_agent_ids_ordered=all_agent_ids_ordered,
            num_goals=len(goal_positions_list),
            agent_physical_params=agent_physical_params, 
            goal_positions_list=goal_positions_list,
            global_aif_config=global_aif_hyperparams,
            allow_multiple_to_same_goal=allow_multiple_to_same_goal_for_g
        )
        
        self.spectral_analyzer = SpectralAnalyzer(
            dynamics_model=dynamics_model_for_agent, 
            prediction_horizon_H=spectral_analysis_horizon_H,
            control_normalization_matrix=spectral_control_norm_matrix,
            alpha_scale=spectral_analysis_alpha_scale
        )
        self.num_spectral_modes_to_consider = num_spectral_modes_to_consider \
            if num_spectral_modes_to_consider is not None \
            else dynamics_model_for_agent.state_size 

        print(f"[ActiveInferenceAgent '{self.agent_id_str}'] Initialized. Internal idx: {self.agent_idx_internal}")
        print(f"  Reasoning: Lvl={self.agent_core_params['reasoning_level']}, EP={self.agent_core_params['use_epistemic_planning']}")
        if self.agent_core_params["possible_goal_configs_G"].size == 0 :
             print(f"  警告: Agent '{self.agent_id_str}' 没有有效的可能目标配置 G。")


    def set_reasoning_mode(self, reasoning_level: int, use_epistemic_planning: bool):
        self.agent_core_params["reasoning_level"] = reasoning_level
        self.agent_core_params["use_epistemic_planning"] = use_epistemic_planning
        num_configs = self.agent_core_params["possible_goal_configs_G"].shape[0] if self.agent_core_params["possible_goal_configs_G"].size >0 else 0
        if num_configs > 0 :
            # 重置信念为均匀先验
            self.agent_core_params["q_G_posterior"] = np.ones(num_configs) / num_configs
            # 如果 beliefs_about_others_q_G 也依赖于推理模式，可能也需要重置
            # for i in range(self.num_agents):
            # self.agent_core_params["beliefs_about_others_q_G"][i] = np.ones(num_configs) / num_configs
        else:
            self.agent_core_params["q_G_posterior"] = np.array([])


    def decide_action(self,
                      all_agent_true_states_global: List[np.ndarray], 
                      all_agent_params_list_for_aif: List[Dict] 
                     ) -> Tuple[Optional[np.ndarray], float, Optional[np.ndarray], Optional[np.ndarray]]:
        
        my_observations_of_world: Dict[int, np.ndarray] = {} 
        for target_j_idx in range(self.num_agents):
            my_observations_of_world[target_j_idx] = aif_core.simulate_observation_for_agent(
                agent_idx_observer=self.agent_idx_internal,
                agent_idx_target=target_j_idx,
                all_agent_true_states=all_agent_true_states_global,
                agent_params_observer=self.agent_core_params 
            )
        
        true_states_for_ho_sim = all_agent_true_states_global \
            if self.agent_core_params["use_epistemic_planning"] and self.agent_core_params["reasoning_level"] >=2 \
            else None

        log_likelihood_G = aif_core.get_likelihood_for_agent(
            current_agent_idx=self.agent_idx_internal,
            current_agent_observations=my_observations_of_world,
            agent_params_list_all=all_agent_params_list_for_aif, 
            true_states_for_simulation=true_states_for_ho_sim
        )

        updated_q_G_posterior = None # 初始化
        if log_likelihood_G.size == 0: 
            if self.agent_core_params["q_G_posterior"] is not None and self.agent_core_params["q_G_posterior"].size > 0:
                 updated_q_G_posterior = self.agent_core_params["q_G_posterior"] 
            elif self.agent_core_params["q_G_prior"] is not None and self.agent_core_params["q_G_prior"].size > 0 :
                 updated_q_G_posterior = self.agent_core_params["q_G_prior"]
            else:
                 updated_q_G_posterior = np.array([])
        else:
            aif_core.update_belief(
                agent_idx=self.agent_idx_internal,
                log_likelihood_G=log_likelihood_G,
                agent_params_list_all=all_agent_params_list_for_aif 
            )
            updated_q_G_posterior = all_agent_params_list_for_aif[self.agent_idx_internal]["q_G_posterior"]

        candidate_delta_s_list = self.spectral_analyzer.calculate_target_deviation_options(
            num_modes_to_consider=self.num_spectral_modes_to_consider
        )
        
        if not candidate_delta_s_list:
            zero_delta_s = np.zeros(self.spectral_analyzer.state_size)
            return zero_delta_s, np.inf, np.array([np.inf]), updated_q_G_posterior

        my_current_physical_state = all_agent_true_states_global[self.agent_idx_internal]
        selected_action_delta_s, min_efe, all_efes_values = aif_core.choice_heuristic(
            agent_idx=self.agent_idx_internal,
            candidate_actions_delta_s=candidate_delta_s_list,
            current_agent_physical_state=my_current_physical_state,
            all_agent_current_states_for_sim=all_agent_true_states_global, 
            agent_params_list_all=all_agent_params_list_for_aif 
        )
        
        if selected_action_delta_s is None : 
            selected_action_delta_s = np.zeros(self.spectral_analyzer.state_size)
            min_efe = np.inf 

        return selected_action_delta_s, min_efe, all_efes_values, updated_q_G_posterior

# --- 模块自测试 ---
def test_active_inference_agent():
    print("--- 测试 ActiveInferenceAgent ---")
    num_agents_test = 2
    num_goals_test = 2
    agent_ids_str_test = [f"AIAgent{i}" for i in range(num_agents_test)]
    
    n_orbit_test = 2 * np.pi / (3600 * 1.5) 
    dt_decision_step = 20.0 
    dynamics_models_test = [CWEquation(n=n_orbit_test, dt=dt_decision_step) for _ in range(num_agents_test)]

    agent_phys_params_list_test = [
        {"type": 's', "agent_radius": 0.5, "evidence_sensitivity_eta": 30.0, 
         "vel_mag_thresh": 0.01, "dist_thresh_evidence": 200.0, "evidence_norm_factor": 1.0, "EPS":1e-9}, 
        {"type": 'A', "agent_radius": 0.3, "evidence_sensitivity_eta": 20.0, "angle_evidence_weight_factor": 1.0,
         "vel_mag_thresh": 0.01, "dist_thresh_evidence": 200.0, "evidence_norm_factor": 1.0, "EPS":1e-9}
    ]
    goal_positions_test = [np.array([100.0, 0.0]), np.array([-100.0, 0.0])]
    
    global_hyperparams_test = {
        "reasoning_level": 2, "use_epistemic_planning": True,
        "alpha_epistemic": 1.0, "beta_pragmatic": 1.0, "gamma_control_effort": 0.005,
        "policy_temperature": 0.01,
        "belief_update_temperature": 1.0,
        "perspective_fusion_lambda": 0.5,
        "obs_variance_pos": 0.05, "obs_variance_vel": 0.005 
    }
    
    # 在这里调用 aif_core.generate_possible_goal_configs
    _possible_G_aitest = aif_core.generate_possible_goal_configs(num_agents_test, num_goals_test, False)
    if _possible_G_aitest.size > 0:
        _num_configs_aitest = _possible_G_aitest.shape[0]
        if _num_configs_aitest == 2: 
             global_hyperparams_test["C_target_dist_G"] = np.array([0.95, 0.05]) 
        elif _num_configs_aitest > 0 : # 确保 _num_configs_aitest > 0
             global_hyperparams_test["C_target_dist_G"] = np.ones(_num_configs_aitest)/_num_configs_aitest
        else: # 如果仍然为0（例如 num_agents > num_goals 且不允许重复）
            global_hyperparams_test["C_target_dist_G"] = np.array([])
    else: 
        global_hyperparams_test["C_target_dist_G"] = np.array([])


    all_agents_core_params_list = [
        aif_core.initialize_agent_aif_params(
            agent_id_str=agent_ids_str_test[i],
            agent_idx_internal=i,
            all_agent_ids_ordered=agent_ids_str_test,
            num_goals=num_goals_test,
            agent_physical_params=agent_phys_params_list_test[i],
            goal_positions_list=goal_positions_test,
            global_aif_config=global_hyperparams_test,
            allow_multiple_to_same_goal=False
        ) for i in range(num_agents_test)
    ]
    
    agent_to_test_idx = 0
    agent_to_test_id = agent_ids_str_test[agent_to_test_idx]
    
    spectral_H = 10
    spectral_alpha = 15.0 
    u_max_for_spectral = 0.1 
    s_diag_elements_spectral = [1.0 / u_max_for_spectral] * dynamics_models_test[agent_to_test_idx].control_size
    S_norm_spectral = np.diag(s_diag_elements_spectral)
    num_modes_spec = 4 

    try:
        ai_agent_instance = ActiveInferenceAgent( 
            agent_id_str=agent_to_test_id,
            all_agent_ids_ordered=agent_ids_str_test,
            agent_physical_params=agent_phys_params_list_test[agent_to_test_idx], 
            all_agents_physical_params_list=agent_phys_params_list_test, 
            goal_positions_list=goal_positions_test,
            global_aif_hyperparams=global_hyperparams_test,
            dynamics_model_for_agent=dynamics_models_test[agent_to_test_idx], 
            spectral_analysis_horizon_H=spectral_H,
            spectral_analysis_alpha_scale=spectral_alpha,
            spectral_control_norm_matrix=S_norm_spectral,
            num_spectral_modes_to_consider=num_modes_spec,
            allow_multiple_to_same_goal_for_g=False
        )
        print(f"\nActiveInferenceAgent {agent_to_test_id} 初始化成功。")
    except Exception as e:
        print(f"ActiveInferenceAgent 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
        
    current_true_states_test = [
        np.array([90.0, 5.0, 1.0, 0.0]),    
        np.array([-90.0, -5.0, -1.0, 0.0])  
    ]

    print(f"\n--- {agent_to_test_id} 执行决策 ---")
    selected_ds, efe_value, all_efes, final_belief = ai_agent_instance.decide_action(
        current_true_states_test,
        all_agents_core_params_list 
    )

    if selected_ds is not None:
        if final_belief is not None and final_belief.size > 0 : 
            print(f"  {agent_to_test_id} 的最终信念 Q(G): {final_belief}")
        else:
            print(f"  {agent_to_test_id} 的最终信念 Q(G): 无有效信念或未更新")

        if all_efes is not None and all_efes.size > 0: 
            print(f"  {agent_to_test_id} 候选动作EFE列表长度: {len(all_efes)}") 
            print(f"  {agent_to_test_id} 候选动作EFE列表 (前5): {all_efes[:5]}") 
        else:
            print(f"  {agent_to_test_id} 候选动作EFE列表: 无有效EFE值")
            
        print(f"  {agent_to_test_id} 选择的动作 (delta_s): {selected_ds[:2] if selected_ds is not None else 'None'}, EFE: {efe_value:.3f}")

        current_physical_state_of_agent = current_true_states_test[ai_agent_instance.agent_idx_internal]
        mpc_target_state_at_N = current_physical_state_of_agent + selected_ds
        print(f"  MPC的期望终端目标状态 (s_current + delta_s): {mpc_target_state_at_N[:2]}")
    else:
        print(f"  {agent_to_test_id} 未能选择动作。")


if __name__ == '__main__':
    # 设置matplotlib支持中文显示 (如果需要在此处测试绘图)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    test_active_inference_agent()