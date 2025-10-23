# import numpy as np
# from mpe.core import World, Agent, Landmark
# from mpe.scenario import BaseScenario


# class Scenario(BaseScenario):
#     def make_world(self):
#         world = World()
#         # set any world properties first
#         world.dim_c = 2
#         # --- 変更点: エージェント数とランドマーク数を設定 ---
#         # 2エージェントで実験する場合は、以下の数値を2に変更してください。
#         num_agents = 3
#         num_landmarks = 3
#         world.collaborative = False # Falseに設定し、各エージェントが個別の報酬を受け取るように変更
#         # add agents
#         world.agents = [Agent() for i in range(num_agents)]
#         for i, agent in enumerate(world.agents):
#             agent.name = 'agent %d' % i
#             agent.collide = True
#             agent.silent = True
#             agent.size = 0.15
#         # add landmarks
#         world.landmarks = [Landmark() for i in range(num_landmarks)]
#         for i, landmark in enumerate(world.landmarks):
#             landmark.name = 'landmark %d' % i
#             landmark.collide = False
#             landmark.movable = False
#         # make initial conditions
#         self.reset_world(world)
#         return world

#     def reset_world(self, world):
#         # random properties for agents
#         for i, agent in enumerate(world.agents):
#             agent.color = np.array([0.35, 0.35, 0.85])
#         # random properties for landmarks
#         for i, landmark in enumerate(world.landmarks):
#             landmark.color = np.array([0.25, 0.25, 0.25])
#         # set random initial states
#         for agent in world.agents:
#             agent.state.p_pos = world.np_random.uniform(-1, +1, world.dim_p)
#             agent.state.p_vel = np.zeros(world.dim_p)
#             agent.state.c = np.zeros(world.dim_c)
#         for i, landmark in enumerate(world.landmarks):
#             landmark.state.p_pos = world.np_random.uniform(-1, +1, world.dim_p)
#             landmark.state.p_vel = np.zeros(world.dim_p)
        
#         # --- 追加: 各エージェントにゴールとなるランドマークを割り当てる ---
#         for i, agent in enumerate(world.agents):
#             agent.goal_l = world.landmarks[i]


#     def is_collision(self, agent1, agent2):
#         delta_pos = agent1.state.p_pos - agent2.state.p_pos
#         dist = np.sqrt(np.sum(np.square(delta_pos)))
#         dist_min = agent1.size + agent2.size
#         return True if dist < dist_min else False

#     # --- 変更点: 報酬関数の修正 ---
#     def reward(self, agent, world):
#         # 報酬 = (エージェントとゴールまでの距離の負の値) + (衝突ペナルティ)
        
#         # 自分のゴールとなるランドマークとの距離に基づく報酬
#         dist = np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_l.state.p_pos)))
#         rew = -dist

#         # 他のエージェントとの衝突に対するペナルティ
#         if agent.collide:
#             for a in world.agents:
#                 if a is agent:
#                     continue
#                 if self.is_collision(a, agent):
#                     rew -= 1
#         return rew

#     # --- 変更点: 観測（状態）の次元数を8に変更 ---
#     def observation(self, agent, world):
#         # 観測の内訳 (合計8次元):
#         # 1. 自身の速度 (2次元)
#         # 2. 他のエージェントへの相対位置 (2次元 * (エージェント数 - 1))
#         # 3. 自身のゴールランドマークへの相対位置 (2次元)
        
#         # 自身のゴールランドマークへの相対位置
#         goal_pos = [agent.goal_l.state.p_pos - agent.state.p_pos]
        
#         # 他のエージェントへの相対位置
#         other_pos = []
#         for other in world.agents:
#             if other is agent:
#                 continue
#             other_pos.append(other.state.p_pos - agent.state.p_pos)
            
#         return np.concatenate([agent.state.p_vel] + other_pos + goal_pos)
