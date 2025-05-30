# simulation/environment.py
import numpy as np
import copy
from typing import Dict, List, Any, Optional, Tuple
import sys
import os
import time
import random
import math #确保导入math模块，因为场景回调中可能用到

# --- 统一的导入路径管理 (保持与您之前版本一致) ---
_simulation_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_dir_env = os.path.dirname(_simulation_dir)
if _project_root_dir_env not in sys.path:
    sys.path.insert(0, _project_root_dir_env)
# --- 路径管理结束 ---

try:
    from common.dynamics import CWEquation
    from common.mpc_controller import MPCController
    from common.delta_v_estimator import DeltaVEstimatorMPC
    from strong_communication.belief_manager import BeliefManager
    from strong_communication.utility_calculator import UtilityCalculator
    from strong_communication.ksc_coalition_formation import KSCCoalitionFormation, IDLE_TASK_ID
    from weak_communication.active_inference import ActiveInferenceAgent
    import weak_communication.aif_functions_isobeliefs_convergent as aif_core
    from mixed_communication.adaptive_controller import AdaptiveController
    from simulation.scenarios import load_scenario, DEFAULT_MPC_PARAMS, DEFAULT_AIF_GLOBAL_HYPERPARAMS, DEFAULT_SC_PHYSICAL_PARAMS, SCENARIOS
except ImportError as e:
    print(f"ERROR [simulation/environment.py]: Failed to import modules: {e}")
    raise

# SpacecraftAgentSim 类定义 (与您上一版本提供的代码一致，这里不再重复)
class SpacecraftAgentSim:
    def __init__(self,
                 agent_config: Dict[str, Any], scenario_config: Dict[str, Any],
                 all_agent_ids_ordered: List[str], ksc_instance: Optional[KSCCoalitionFormation],
                 env_dynamics_dt: float, all_agents_physical_params_list: List[Dict[str, Any]]):
        self.id = agent_config['id']
        self.physical_params = copy.deepcopy(agent_config.get("physical_params", {}))
        self.ksc_params = copy.deepcopy(agent_config.get("ksc_params", {}))
        self.current_state: np.ndarray = np.array(agent_config['initial_state'])
        sim_params_overall = scenario_config.get("simulation_params", {})
        mpc_params_scen = scenario_config.get("mpc_params", copy.deepcopy(DEFAULT_MPC_PARAMS))
        adaptive_params_scen = scenario_config.get("adaptive_controller_params", {})
        spectral_params_scen = scenario_config.get("spectral_params", {})
        aif_global_hyperparams_scen = scenario_config.get("aif_global_hyperparams", copy.deepcopy(DEFAULT_AIF_GLOBAL_HYPERPARAMS))
        self.dynamics_model = CWEquation(n=sim_params_overall.get("orbit_mean_motion_n", 0.0011), dt=env_dynamics_dt)
        self.mpc_controller = MPCController(
            dynamics_model=self.dynamics_model, prediction_horizon=mpc_params_scen["prediction_horizon"],
            Q_terminal=np.diag(mpc_params_scen["Q_terminal_diag"]), R_control=np.diag(mpc_params_scen["R_control_diag"]),
            control_constraints={'min': [-mpc_params_scen["u_max_abs"]] * self.dynamics_model.control_size,
                                 'max': [mpc_params_scen["u_max_abs"]] * self.dynamics_model.control_size},
            solver_print_level=mpc_params_scen.get("solver_print_level", 0)
        )
        tasks_definitions_for_ksc_ctrl = scenario_config.get("tasks", {})
        if not isinstance(tasks_definitions_for_ksc_ctrl, dict): 
            tasks_definitions_for_ksc_ctrl = {}

        self.adaptive_controller = AdaptiveController(
            agent_id_str=self.id, all_agent_ids_ordered=all_agent_ids_ordered,
            ksc_algorithm_instance=ksc_instance,
            delta_v_estimator_for_strong_nav=None, 
            task_definitions_for_ksc=tasks_definitions_for_ksc_ctrl, 
            agent_physical_params=self.physical_params,
            all_agents_physical_params_list=all_agents_physical_params_list,
            aif_goal_positions_list=scenario_config.get("aif_goal_positions", []),
            global_aif_hyperparams=aif_global_hyperparams_scen,
            dynamics_model_for_agent=self.dynamics_model,
            spectral_analysis_horizon_H=spectral_params_scen["horizon_H"],
            spectral_analysis_alpha_scale=spectral_params_scen["alpha_scale"],
            spectral_control_norm_matrix=np.diag([1.0/spectral_params_scen.get("u_max_for_norm", 0.1)]*self.dynamics_model.control_size),
            num_spectral_modes_to_consider=spectral_params_scen["num_modes_to_consider"],
            allow_multiple_to_same_goal_for_aif_g=scenario_config.get("aif_allow_multiple_to_same_goal", False),
            csi_threshold_low=adaptive_params_scen["csi_threshold_low"],
            csi_threshold_high=adaptive_params_scen["csi_threshold_high"],
            csi_ema_alpha=adaptive_params_scen["csi_ema_alpha"]
        )
        self.current_target_task_id: Optional[str] = None
        self.current_decision_mode: str = "strong" 
        self.history_states: List[np.ndarray] = [self.current_state.copy()]
        self.history_controls: List[np.ndarray] = []
        self.history_delta_s_planned: List[Optional[np.ndarray]] = [] 
        self.history_decision_mode: List[str] = [self.current_decision_mode]
        self.history_csi_smooth: List[float] = [1.0] 

    def update_physical_state(self, control_input: np.ndarray) -> None:
        self.current_state = self.dynamics_model.step(self.current_state, control_input)
        self.history_states.append(self.current_state.copy())
        if control_input is not None:
            self.history_controls.append(control_input.copy())
        elif self.history_controls: 
            self.history_controls.append(np.zeros(self.dynamics_model.control_size))

    def get_decision_delta_s(self,
                             all_agent_true_states_list: List[np.ndarray],
                             all_agents_aif_params_list_env: List[Dict[str, Any]],
                             communication_data_for_agent: Dict[str, Any]
                             ) -> np.ndarray:
        if hasattr(self, 'env_tasks_ref'): 
            self.adaptive_controller.task_definitions_for_ksc = self.env_tasks_ref
        nominal_delta_s, mode, debug_info = self.adaptive_controller.decide_action(
            all_agent_true_states_list, all_agents_aif_params_list_env, communication_data_for_agent)
        self.current_decision_mode = mode
        current_delta_s_to_log = nominal_delta_s if nominal_delta_s is not None else np.zeros(self.dynamics_model.state_size)
        self.history_delta_s_planned.append(current_delta_s_to_log.copy())
        self.history_decision_mode.append(mode)
        self.history_csi_smooth.append(debug_info.get("csi_smooth", self.adaptive_controller.current_csi_smooth))
        self.current_target_task_id = debug_info.get("ksc_target_task_id")
        return np.zeros(self.dynamics_model.state_size) if nominal_delta_s is None else nominal_delta_s

    def get_control_input(self, target_delta_s: np.ndarray) -> np.ndarray:
        mpc_target_terminal_state = self.current_state + target_delta_s
        state_diff_norm = np.linalg.norm(mpc_target_terminal_state - self.current_state)
        if state_diff_norm < 0.5: 
            return np.zeros(self.dynamics_model.control_size)
        optimal_u0, _, success = self.mpc_controller.solve(
            current_state=self.current_state,
            target_terminal_state=mpc_target_terminal_state
        )
        if success: return optimal_u0
        else: return np.zeros(self.dynamics_model.control_size)

class SimulationEnvironment:
    # __init__ 和其他辅助方法 (_initialize_history_logs, _setup_strong_communication_components, 
    # _record_beliefs_for_plot, _collect_observations_and_update_ksc_beliefs, 
    # _simulate_physical_state_random_perturbations, _update_dynamic_task_states,
    # _update_task_completion_status)
    # 保持与上一回复中提供的版本一致。
    # 关键在于 step 方法中的回调处理 和 _get_communication_data_for_adaptive_controller 的修正。
    def __init__(self, scenario_config_name: str): # (与上一回复一致)
        self.scenario_config = load_scenario(scenario_config_name)
        if not self.scenario_config: raise ValueError(f"无法加载场景: {scenario_config_name}")
        # print(f"\n--- 初始化仿真环境 (场景: {scenario_config_name}) ---")
        self.sim_params = self.scenario_config.get("simulation_params", {})
        self.dt = self.sim_params["dt_simulation"]
        self.time_step = 0; self.current_time = 0.0; self.ksc_decision_round_count = 0
        self.agent_ids: List[str] = [sc_def["id"] for sc_def in self.scenario_config["spacecrafts"]]
        self.num_agents: int = len(self.agent_ids)
        self.tasks: Dict[str, Dict[str, Any]] = copy.deepcopy(self.scenario_config.get("tasks", {}))
        self.task_type_names: List[str] = self.scenario_config.get("task_type_names", [])
        self.task_dynamics_models: Dict[str, CWEquation] = {}
        for task_id, task_def in self.tasks.items():
            task_def.setdefault("initial_state", [0.0, 0.0, 0.0, 0.0])
            task_def["current_state"] = np.array(task_def["initial_state"])
            task_def["position"] = task_def["current_state"][:2].tolist()
            self.task_dynamics_models[task_id] = CWEquation(n=self.sim_params.get("orbit_mean_motion_n",0.0011), dt=self.dt)
            task_def.setdefault("total_workload", 100.0); task_def.setdefault("current_completion_percentage", 0.0)
            task_def.setdefault("work_rate_per_agent_per_dt", 1.0); task_def.setdefault("status", "active")
        self.active_task_ids: List[str] = [tid for tid, tval in self.tasks.items() if tval.get("current_completion_percentage", 0.0) < 100.0]
        self._initialize_history_logs()
        self.ksc_algorithm: Optional[KSCCoalitionFormation]=None; self.belief_manager_ksc:Optional[BeliefManager]=None; self.utility_calculator_ksc:Optional[UtilityCalculator]=None
        if self.sim_params.get("strong_mode_components_enabled", True):
            self._setup_strong_communication_components()
            if self.ksc_algorithm:
                 self.history_global_utility_ksc.append(self.ksc_algorithm._sim_best_utility)
                 self.history_assignments_ksc.append(copy.deepcopy(self.ksc_algorithm._sim_best_assignment))
                 if self.ksc_algorithm.stats_active_links_per_iteration: self.ksc_stats_active_links.append(self.ksc_algorithm.stats_active_links_per_iteration[0])
                 else: self.ksc_algorithm._update_neighbor_graph(); self.ksc_stats_active_links.append(self.ksc_algorithm.stats_active_links_per_iteration[0] if self.ksc_algorithm.stats_active_links_per_iteration else 0)
            else: self._initialize_empty_ksc_history_iter_data()
            self._record_beliefs_for_plot()
        else: self._initialize_empty_ksc_history_iter_data()
        self.all_agents_physical_params_list: List[Dict[str, Any]] = [copy.deepcopy(sc.get("physical_params", DEFAULT_SC_PHYSICAL_PARAMS)) for sc in self.scenario_config["spacecrafts"]]
        self.all_agents_aif_core_params_shared: List[Dict[str, Any]] = []
        if self.num_agents > 0:
            aif_g_h_s = self.scenario_config.get("aif_global_hyperparams", copy.deepcopy(DEFAULT_AIF_GLOBAL_HYPERPARAMS)); aif_g_p_s = self.scenario_config.get("aif_goal_positions", []); aif_a_m_s = self.scenario_config.get("aif_allow_multiple_to_same_goal", False)
            for i, agent_id_for_aif in enumerate(self.agent_ids):
                agent_cfg_aif = next((sc for sc in self.scenario_config["spacecrafts"] if sc["id"] == agent_id_for_aif), None)
                if agent_cfg_aif:
                    phys_p_aif = agent_cfg_aif.get("physical_params", copy.deepcopy(DEFAULT_SC_PHYSICAL_PARAMS))
                    core_p = aif_core.initialize_agent_aif_params(agent_id_for_aif, i, self.agent_ids, len(aif_g_p_s), phys_p_aif, aif_g_p_s, aif_g_h_s, aif_a_m_s)
                    self.all_agents_aif_core_params_shared.append(core_p)
                else: self.all_agents_aif_core_params_shared.append(copy.deepcopy(DEFAULT_AIF_GLOBAL_HYPERPARAMS))
        self.agents: Dict[str, SpacecraftAgentSim] = {}
        for sc_def_agent in self.scenario_config["spacecrafts"]:
            agent_instance = SpacecraftAgentSim(sc_def_agent, self.scenario_config, self.agent_ids, self.ksc_algorithm, self.dt, self.all_agents_physical_params_list)
            agent_instance.env_tasks_ref = self.tasks; self.agents[agent_instance.id] = agent_instance
        # print(f"环境初始化完毕。共 {self.num_agents} 航天器, {len(self.tasks)} 初始任务。")
        # if self.ksc_stats_active_links: print(f"  初始活动通信链接数 (KSC图): {self.ksc_stats_active_links[0]}")

    def _initialize_history_logs(self): # (与上一回复一致)
        self.history_global_utility_ksc = [] ; self.history_assignments_ksc = []
        self.history_task_status = {tid: [self.tasks[tid].get("status", "active")] for tid in self.tasks.keys() if self.tasks}
        self.history_task_completion_percentage = {tid: [self.tasks[tid].get("current_completion_percentage", 0.0)] for tid in self.tasks.keys() if self.tasks}
        self.history_task_dynamic_states = {tid: [np.array(self.tasks[tid]["initial_state"]).copy()] for tid in self.tasks.keys() if self.tasks and "initial_state" in self.tasks[tid]}
        self.history_num_active_tasks = [len(self.active_task_ids) if hasattr(self, 'active_task_ids') and self.active_task_ids is not None else 0]
        self.ksc_stats_messages_broadcasted = []; self.ksc_stats_messages_processed = []; self.ksc_stats_active_links = []
        self.env_level_active_comm_links = []; self.history_beliefs_at_ksc_step = {}

    def _initialize_empty_ksc_history_iter_data(self): # (与上一回复一致)
        self.history_global_utility_ksc.append(self.history_global_utility_ksc[-1] if self.history_global_utility_ksc else 0.0)
        self.history_assignments_ksc.append(self.history_assignments_ksc[-1] if self.history_assignments_ksc else {aid: IDLE_TASK_ID for aid in self.agent_ids})
        self.ksc_stats_messages_broadcasted.append(0); self.ksc_stats_messages_processed.append(0)
        self.ksc_stats_active_links.append(self.ksc_stats_active_links[-1] if self.ksc_stats_active_links else 0)
        if self.ksc_decision_round_count not in self.history_beliefs_at_ksc_step: self.history_beliefs_at_ksc_step[self.ksc_decision_round_count] = {}

    def _setup_strong_communication_components(self): # (与上一回复一致)
        initial_alpha_ksc = self.scenario_config.get("aif_global_hyperparams", {}).get("initial_belief_alpha_ksc", 1.0)
        self.belief_manager_ksc = BeliefManager(self.agent_ids, list(self.tasks.keys()), len(self.task_type_names), initial_alpha_ksc)
        orig_scen_def = SCENARIOS.get(self.scenario_config.get("scenario_name_loaded_from", ""), {})
        belief_overrides = orig_scen_def.get("initial_belief_overrides")
        if belief_overrides and self.belief_manager_ksc:
            for aid, task_bels in belief_overrides.items():
                for tid, alphas in task_bels.items(): self.belief_manager_ksc.override_agent_belief_alphas_for_task(aid,tid,np.array(alphas,dtype=float))
        mpc_p = self.scenario_config.get("mpc_params", copy.deepcopy(DEFAULT_MPC_PARAMS)); dv_model = CWEquation(n=self.sim_params.get("orbit_mean_motion_n",0.0011),dt=self.dt)
        dv_R_diag = mpc_p.get("R_control_diag_dv_est", [0.01]*dv_model.control_size); dv_Qf_raw = mpc_p.get("Q_terminal_diag_dv_est", [100.0,100.0,10.0,10.0])
        qf_len = dv_model.state_size; dv_Qf_diag = dv_Qf_raw if len(dv_Qf_raw)==qf_len else ([100.0]*(qf_len//2)+[10.0]*(qf_len-qf_len//2))[:qf_len]
        ksc_dv_est = DeltaVEstimatorMPC(dv_model, mpc_p.get("prediction_horizon_dv_est",10),np.diag(dv_R_diag),np.diag(dv_Qf_diag),{'min':[-mpc_p["u_max_abs"]*1.5]*dv_model.control_size, 'max':[mpc_p["u_max_abs"]*1.5]*dv_model.control_size})
        sc_phys_p = {sc["id"]:sc.get("physical_params",copy.deepcopy(DEFAULT_SC_PHYSICAL_PARAMS)) for sc in self.scenario_config["spacecrafts"]}
        self.utility_calculator_ksc=UtilityCalculator(ksc_dv_est,self.tasks,sc_phys_p)
        ksc_assign_init = {aid:IDLE_TASK_ID for aid in self.agent_ids}
        k_vals_ksc = {sc["id"]:sc.get("ksc_params",{}).get("k_value",2) for sc in self.scenario_config["spacecrafts"]}
        ksc_init_states = {sc["id"]:np.array(sc["initial_state"]) for sc in self.scenario_config["spacecrafts"]}
        ksc_cfg = self.scenario_config.get("ksc_algorithm_config",{})
        self.ksc_algorithm = KSCCoalitionFormation(self.agent_ids,list(self.tasks.keys()),k_vals_ksc,self.utility_calculator_ksc,self.belief_manager_ksc,self.task_type_names,ksc_assign_init,
                                                 ksc_cfg.get("ksc_max_coalition_size_per_task"), ksc_cfg.get("ksc_min_agents_per_task"), ksc_init_states,
                                                 self.sim_params.get("communication_range"), ksc_cfg.get("max_dfs_branching_factor",2))
        if self.ksc_algorithm and not self.ksc_algorithm.stats_active_links_per_iteration: self.ksc_algorithm._update_neighbor_graph()

    def _record_beliefs_for_plot(self) -> None: # (与上一回复一致)
        current_ksc_round = self.ksc_decision_round_count
        if current_ksc_round not in self.history_beliefs_at_ksc_step: self.history_beliefs_at_ksc_step[current_ksc_round] = {}
        if not self.belief_manager_ksc: return
        current_beliefs_snapshot: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        for agent_id_rec_bel in self.agent_ids:
            current_beliefs_snapshot[agent_id_rec_bel] = {}
            for task_id_rec_bel in self.tasks.keys(): 
                belief_dist = self.belief_manager_ksc.get_expected_belief_dist_for_task(agent_id_rec_bel, task_id_rec_bel)
                if belief_dist is not None: current_beliefs_snapshot[agent_id_rec_bel][task_id_rec_bel] = belief_dist.copy()
                else:
                    last_belief_data = None
                    if current_ksc_round > 0:
                        prev_beliefs = self.history_beliefs_at_ksc_step.get(current_ksc_round - 1, {})
                        agent_prev_bels = prev_beliefs.get(agent_id_rec_bel, {})
                        last_belief_data = agent_prev_bels.get(task_id_rec_bel)
                    current_beliefs_snapshot[agent_id_rec_bel][task_id_rec_bel] = last_belief_data.copy() if last_belief_data is not None else None
        self.history_beliefs_at_ksc_step[current_ksc_round] = current_beliefs_snapshot

    def _collect_observations_and_update_ksc_beliefs(self) -> None: # (与上一回复一致)
        if not self.active_task_ids or not self.belief_manager_ksc or not self.ksc_algorithm: return
        shared_obs_agg: Dict[str, np.ndarray] = {}; num_types = self.belief_manager_ksc.num_task_types
        if num_types == 0: return
        current_ksc_assign = self.ksc_algorithm._sim_best_assignment
        for task_id_obs in self.active_task_ids:
            if task_id_obs not in self.tasks: continue
            task_def_obs = self.tasks[task_id_obs]; true_type_idx = task_def_obs.get("true_type_idx",-1)
            if true_type_idx == -1: continue
            obs_counts_task = np.zeros(num_types)
            for agent_id_obs, agent_obj_obs in self.agents.items():
                if current_ksc_assign.get(agent_id_obs) == task_id_obs:
                    acc = agent_obj_obs.physical_params.get("observation_accuracy_p_corr",0.85)
                    if random.random() < acc: obs_counts_task[true_type_idx] += 1
                    elif num_types > 1:
                        wrong_types = [i for i in range(num_types) if i != true_type_idx]
                        if wrong_types: obs_counts_task[random.choice(wrong_types)] +=1
            if np.sum(obs_counts_task) > 0: shared_obs_agg[task_id_obs] = obs_counts_task
        if shared_obs_agg: self.belief_manager_ksc.aggregate_and_update_beliefs_from_shared_observations(shared_obs_agg)
        self._record_beliefs_for_plot()

    def _simulate_physical_state_random_perturbations(self) -> None: # (与上一回复一致)
        new_phys_states:Dict[str,np.ndarray]={}; pos_drift_max_dt=self.sim_params.get("perturbation_pos_drift_max_per_dt",0.1)*self.dt; vel_drift_max_dt=self.sim_params.get("perturbation_vel_drift_max_per_dt",0.005)*self.dt
        for aid,aobj in self.agents.items():
            s_curr=aobj.current_state; s_dim=aobj.dynamics_model.state_size; p_dims=s_dim//2; v_dims=s_dim-p_dims
            p_drift=(np.random.rand(p_dims)*2-1)*pos_drift_max_dt; v_drift=(np.random.rand(v_dims)*2-1)*vel_drift_max_dt
            s_pert=s_curr.copy(); s_pert[:p_dims]+=p_drift; s_pert[p_dims:p_dims+v_dims]+=v_drift; new_phys_states[aid]=s_pert
        for aid,aobj in self.agents.items(): aobj.current_state=new_phys_states[aid].copy()
        if self.ksc_algorithm: self.ksc_algorithm.update_agent_physical_states(new_phys_states)

    def _get_communication_data_for_adaptive_controller(self, current_agent_id: str) -> Dict[str, Any]:
        # (使用上一回复中已修正的版本，确保对 comm_range_param 的类型检查和转换)
        comm_data: Dict[str, Any] = {"connected_neighbors": [], "link_qualities": {}}
        agent_ids_list = self.agent_ids
        
        comm_range_param = self.sim_params.get("communication_range")
        communication_radius_sq = -1.0 

        if comm_range_param is not None:
            if isinstance(comm_range_param, (int, float)): 
                if comm_range_param >= 0:
                    communication_radius_sq = comm_range_param**2
            elif isinstance(comm_range_param, str):
                try:
                    numeric_comm_range = float(comm_range_param)
                    if numeric_comm_range >= 0:
                        communication_radius_sq = numeric_comm_range**2
                except ValueError:
                    if comm_range_param.strip().lower() != "none":
                        # This warning was in your logs, so the string is indeed the expression
                        # print(f"[ENV WARNING] _get_communication_data: 'communication_range' in sim_params ('{comm_range_param}') is an invalid non-numeric string. Treating as infinite range for this step.")
                        pass # communication_radius_sq remains -1.0
            # else: other type, communication_radius_sq remains -1.0

        current_agent_obj = self.agents.get(current_agent_id)
        if not current_agent_obj: return comm_data 
        current_agent_pos = current_agent_obj.current_state[:2]

        for other_agent_id in agent_ids_list:
            if other_agent_id == current_agent_id: continue
            other_agent_obj = self.agents.get(other_agent_id)
            if not other_agent_obj: continue
            other_agent_pos = other_agent_obj.current_state[:2]
            
            if communication_radius_sq < 0: 
                comm_data["connected_neighbors"].append(other_agent_id)
                comm_data["link_qualities"][other_agent_id] = 1.0
            else:
                dist_sq = np.sum((current_agent_pos - other_agent_pos)**2)
                if dist_sq <= communication_radius_sq:
                    comm_data["connected_neighbors"].append(other_agent_id)
                    quality = 1.0 if communication_radius_sq == 0 else max(0.0, 1.0 - (dist_sq / communication_radius_sq) )
                    comm_data["link_qualities"][other_agent_id] = quality
        return comm_data
        
    def _update_dynamic_task_states(self): pass # (与上一回复一致)

    def _update_task_completion_status(self) -> None: # (与上一回复一致)
        if not self.active_task_ids: return
        tasks_newly_completed_this_step: List[str] = []
        current_ksc_assignment_local = self.ksc_algorithm._sim_best_assignment if self.ksc_algorithm and hasattr(self.ksc_algorithm, '_sim_best_assignment') else {}
        for task_id_prog in list(self.active_task_ids):
            if task_id_prog not in self.tasks: continue
            task_def_prog = self.tasks[task_id_prog]
            if task_def_prog.get("current_completion_percentage", 0.0) >= 100.0:
                if task_id_prog in self.active_task_ids: self.active_task_ids.remove(task_id_prog)
                continue
            agents_assigned = [aid for aid,tid in current_ksc_assignment_local.items() if tid==task_id_prog]
            min_needed = task_def_prog.get("min_agents_needed",1)
            if len(agents_assigned) < min_needed:
                self.history_task_completion_percentage.setdefault(task_id_prog,[]).append(task_def_prog.get("current_completion_percentage",0.0)); continue
            task_pos = task_def_prog.get("position")
            if task_pos is None:
                self.history_task_completion_percentage.setdefault(task_id_prog,[]).append(task_def_prog.get("current_completion_percentage",0.0)); continue
            num_working = 0
            for agent_id_work in agents_assigned:
                if agent_id_work not in self.agents: continue
                dist = np.linalg.norm(self.agents[agent_id_work].current_state[:2] - np.array(task_pos))
                if dist <= self.sim_params.get("task_completion_distance_threshold",25.0): num_working+=1
            if num_working > 0:
                work_rate = task_def_prog.get("work_rate_per_agent_per_dt",1.0) * num_working
                curr_comp = task_def_prog.get("current_completion_percentage",0.0)
                total_load = task_def_prog.get("total_workload",100.0)
                if total_load <= 1e-6: total_load=1.0
                inc_perc = (work_rate / total_load) * 100.0
                new_comp = min(100.0, curr_comp + inc_perc)
                task_def_prog["current_completion_percentage"] = new_comp
                if new_comp >= 100.0: tasks_newly_completed_this_step.append(task_id_prog)
            self.history_task_completion_percentage.setdefault(task_id_prog,[]).append(task_def_prog.get("current_completion_percentage",0.0))
        for task_id_done in tasks_newly_completed_this_step:
            if self.tasks[task_id_done].get("status") == "active":
                self.tasks[task_id_done]["status"]="completed"; self.tasks[task_id_done]["completion_time"]=self.current_time
                ttn_done=self.tasks[task_id_done].get("true_type_name",""); self.tasks[task_id_done]["value_realized"]=self.tasks[task_id_done].get("type_rewards",{}).get(ttn_done,0.0)
                if task_id_done in self.active_task_ids: self.active_task_ids.remove(task_id_done)
                agents_on_comp=[aid for aid,tid in current_ksc_assignment_local.items() if tid==task_id_done]
                if self.ksc_algorithm: self.ksc_algorithm.notify_task_completion(task_id_done, agents_on_comp)

    def step(self) -> bool:
        if self.sim_params.get("stop_if_all_tasks_done", False) and \
           not self.active_task_ids and len(self.tasks) > 0 and self.time_step > 0:
            return False
        if self.time_step >= self.sim_params.get("max_time_steps", 100):
            return False

        self.current_time += self.dt
        self.time_step += 1
        self._update_dynamic_task_states()

        # --- Callback processing with robust eval handling ---
        callbacks = self.scenario_config.get("simulation_callbacks", {}).get("on_step_begin", [])
        for callback in callbacks:
            try:
                condition_met = eval(callback["condition"], {"env": self, "np": np, "random": random, "math": math, "time": time})
                if condition_met:
                    action_type = callback.get("action")
                    if action_type == "update_env_param":
                        param_path_str = callback.get("param_path")
                        new_value_from_callback = callback.get("value")
                        
                        parts = param_path_str.split('.')
                        target_obj = self
                        for part_idx, part in enumerate(parts[:-1]):
                            if hasattr(target_obj, part): target_obj = getattr(target_obj, part)
                            elif isinstance(target_obj, dict): target_obj = target_obj.get(part)
                            else: target_obj = None; break
                        
                        param_name_to_update = parts[-1]
                        evaluated_value = new_value_from_callback # Default to original if all parsing fails

                        if isinstance(new_value_from_callback, str):
                            try:
                                # Attempt to evaluate if it's an expression
                                evaluated_value = eval(new_value_from_callback, {"env": self, "np": np, "random": random, "math": math, "time": time})
                            except Exception: # Eval failed
                                if new_value_from_callback.strip().lower() == "none":
                                    evaluated_value = None
                                else:
                                    try: # Try float conversion for simple number strings
                                        evaluated_value = float(new_value_from_callback)
                                    except ValueError:
                                        # If it's 'sim_params.communication_range' and still a problematic string, set to None
                                        if param_path_str == "sim_params.communication_range":
                                            print(f"  CALLBACK WARNING: Invalid string value '{new_value_from_callback}' for critical param '{param_path_str}'. Setting to None.")
                                            evaluated_value = None
                                        # For other params, might keep original string or raise error depending on policy
                                        # else: print(f"  CALLBACK WARNING: Could not process value '{new_value_from_callback}' for '{param_path_str}'. Kept as string.")
                        
                        # Apply the (potentially converted) value
                        if target_obj is not None:
                            if hasattr(target_obj, param_name_to_update):
                                setattr(target_obj, param_name_to_update, evaluated_value)
                            elif isinstance(target_obj, dict) and param_name_to_update in target_obj:
                                target_obj[param_name_to_update] = evaluated_value
                            # print(f"  CALLBACK @t={self.current_time:.1f}s: Updated '{param_path_str}' to '{evaluated_value}' (type: {type(evaluated_value)})")
                    # ... (other callback actions like update_task_reward)
            except Exception as e_cb:
                print(f"  CALLBACK ERROR @t={self.current_time:.1f}s: Executing callback failed - {e_cb}")


        phys_perturb_interval = self.sim_params.get("physical_state_update_interval_sim_steps", -1)
        if self.time_step > 0 and phys_perturb_interval > 0 and self.time_step % phys_perturb_interval == 0:
            self._simulate_physical_state_random_perturbations()

        dt_ksc = self.sim_params.get("dt_decision_ksc", self.dt)
        ksc_decision_interval_steps = int(max(1, dt_ksc / self.dt))
        
        if self.time_step % ksc_decision_interval_steps == 0:
            self.ksc_decision_round_count += 1
            if self.ksc_algorithm and self.sim_params.get("strong_mode_components_enabled", True):
                current_physical_states_for_ksc = {aid: ag_obj.current_state for aid, ag_obj in self.agents.items()}
                self.ksc_algorithm.update_agent_physical_states(current_physical_states_for_ksc)
                if self.utility_calculator_ksc: self.utility_calculator_ksc.task_definitions = self.tasks # Update KSC's view of tasks
                
                ksc_assignment, ksc_utility, ksc_stats = self.ksc_algorithm.run_iteration(self.ksc_decision_round_count)
                self.history_global_utility_ksc.append(ksc_utility)
                self.history_assignments_ksc.append(copy.deepcopy(ksc_assignment))
                self.ksc_stats_messages_broadcasted.append(ksc_stats.get('messages_broadcasted', 0))
                self.ksc_stats_messages_processed.append(ksc_stats.get('messages_processed_by_agents', 0))
                if self.ksc_algorithm.stats_active_links_per_iteration : # Ensure list is not empty
                    self.ksc_stats_active_links.append(ksc_stats.get('active_communication_links', self.ksc_algorithm.stats_active_links_per_iteration[-1]))
                else :
                    self.ksc_stats_active_links.append(ksc_stats.get('active_communication_links',0))


            else: self._initialize_empty_ksc_history_iter_data()
            belief_update_freq_ksc = self.sim_params.get("belief_update_interval_ksc_steps", 1)
            if self.ksc_decision_round_count % belief_update_freq_ksc == 0:
                if self.belief_manager_ksc and self.sim_params.get("strong_mode_components_enabled", True) and self.active_task_ids:
                    self._collect_observations_and_update_ksc_beliefs() 
                else: self._record_beliefs_for_plot()
            elif self.ksc_decision_round_count not in self.history_beliefs_at_ksc_step:
                last_recorded_round = -1
                if self.history_beliefs_at_ksc_step:
                    valid_rounds = [r for r in self.history_beliefs_at_ksc_step.keys() if r < self.ksc_decision_round_count]
                    if valid_rounds: last_recorded_round = max(valid_rounds)
                if last_recorded_round != -1: self.history_beliefs_at_ksc_step[self.ksc_decision_round_count] = copy.deepcopy(self.history_beliefs_at_ksc_step[last_recorded_round])
                else: self._record_beliefs_for_plot()
        
        dt_aif = self.sim_params.get("dt_decision_aif", self.dt)
        aif_decision_interval_steps = int(max(1, dt_aif / self.dt))
        target_delta_s_for_all_agents: Dict[str, np.ndarray] = {}

        if self.time_step % aif_decision_interval_steps == 0:
            all_current_true_states_list: List[np.ndarray] = [self.agents[aid].current_state for aid in self.agent_ids]
            for agent_id_decision in self.agent_ids:
                agent_object = self.agents[agent_id_decision]
                agent_object.adaptive_controller.task_definitions_for_ksc = self.tasks
                comm_data = self._get_communication_data_for_adaptive_controller(agent_id_decision)
                current_delta_s = agent_object.get_decision_delta_s(all_current_true_states_list, self.all_agents_aif_core_params_shared, comm_data)
                target_delta_s_for_all_agents[agent_id_decision] = current_delta_s
        else: 
            for agent_id_decision in self.agent_ids:
                if self.agents[agent_id_decision].history_delta_s_planned: target_delta_s_for_all_agents[agent_id_decision] = self.agents[agent_id_decision].history_delta_s_planned[-1]
                else: target_delta_s_for_all_agents[agent_id_decision] = np.zeros(self.agents[agent_id_decision].dynamics_model.state_size)

        for agent_id_control in self.agent_ids:
            agent_object = self.agents[agent_id_control]
            delta_s_to_achieve = target_delta_s_for_all_agents.get(agent_id_control)
            if delta_s_to_achieve is None: delta_s_to_achieve = np.zeros(agent_object.dynamics_model.state_size)
            control_input_actual = agent_object.get_control_input(delta_s_to_achieve)
            agent_object.update_physical_state(control_input_actual)

        self._update_task_completion_status()

        self.history_num_active_tasks.append(len(self.active_task_ids))
        
        # --- Corrected history logging for env_level_active_comm_links ---
        current_env_links = 0
        comm_range_val_for_hist = self.sim_params.get("communication_range")
        valid_comm_radius_sq_for_hist = -1.0
        if comm_range_val_for_hist is not None:
            if isinstance(comm_range_val_for_hist, (int, float)):
                if comm_range_val_for_hist >=0: valid_comm_radius_sq_for_hist = comm_range_val_for_hist**2
            # String conversion already handled by robust _get_communication_data or callback logic
            # No need to repeat string to float conversion here if callback is fixed
            # However, if a string somehow persists, this part would error without a try-except
            # For now, assume comm_range_val_for_hist is number or None due to callback fix.

        if valid_comm_radius_sq_for_hist >= 0: # Finite, non-negative range
            for i_link in range(self.num_agents):
                for j_link in range(i_link + 1, self.num_agents):
                    agent1 = self.agents.get(self.agent_ids[i_link])
                    agent2 = self.agents.get(self.agent_ids[j_link])
                    if agent1 and agent2 and agent1.current_state is not None and agent2.current_state is not None:
                        pos1 = agent1.current_state[:2]
                        pos2 = agent2.current_state[:2]
                        if np.sum((pos1 - pos2)**2) <= valid_comm_radius_sq_for_hist:
                            current_env_links += 1
        elif self.num_agents > 1: 
            current_env_links = self.num_agents * (self.num_agents - 1) // 2
        self.env_level_active_comm_links.append(current_env_links)
        # --- End of corrected history logging ---

        for task_id_hist, task_data_hist in self.tasks.items():
            self.history_task_status.setdefault(task_id_hist, []).append(task_data_hist.get("status", "unknown"))
            if task_id_hist in self.history_task_dynamic_states and "current_state" in task_data_hist:
                self.history_task_dynamic_states[task_id_hist].append(np.array(task_data_hist["current_state"]).copy())
        
        return True

    # ... (模块自测试 __main__ 保持不变) ...