from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import algorithms.evaluation as evaluation
from world.game import Agent, Directions

if TYPE_CHECKING:
    from world.game_state import GameState


class MultiAgentSearchAgent(Agent, ABC):
    """
    Base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """

    def __init__(self, depth: str = "2", _index: int = 0, prob: str = "0.0") -> None:
        self.index = 0  # Drone is always agent 0
        self.depth = int(depth)
        self.prob = float(
            prob
        )  # Probability that each hunter acts randomly (0=greedy, 1=random)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone from the current GameState.
        """
        pass


class RandomAgent(MultiAgentSearchAgent):
    """
    Agent that chooses a legal action uniformly at random.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Get a random legal action for the drone.
        """
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using minimax.

        Tips:
        - The game tree alternates: drone (MAX) -> hunter1 (MIN) -> hunter2 (MIN) -> ... -> drone (MAX) -> ...
        - Use self.depth to control the search depth. depth=1 means the drone moves once and each hunter moves once.
        - Use state.get_legal_actions(agent_index) to get legal actions for a specific agent.
        - Use state.generate_successor(agent_index, action) to get the successor state after an action.
        - Use state.is_win() and state.is_lose() to check terminal states.
        - Use state.get_num_agents() to get the total number of agents.
        - Use self.evaluation_function(state) to evaluate leaf/terminal states.
        - The next agent is (agent_index + 1) % num_agents. Depth decreases after all agents have moved (full ply).
        - Return the ACTION (not the value) that maximizes the minimax value for the drone.
        """

        # VERSIÓN INICIAL:
        # m_agents = state.get_num_agents()
        #
        # def minimax(state: GameState, agent: int, depth: int) -> float:
        #     if state.is_win() or state.is_lose() or depth == 0:
        #         return self.evaluation_function(state)
        #
        # acciones = state.get_legal_actions(agent)
        # if not acciones:
        #     return self.evaluation_function(state)
        #
        # siguiente_agente = (agent + 1) % m_agents
        #
        # if siguiente_agente == 0:
        #     siguiente_depth = depth - 1
        # else:
        #     siguiente_depth = depth
        #
        # valores = [
        #     minimax(state.generate_successor(agent, accion), siguiente_agente, siguiente_depth)
        #     for accion in acciones
        # ]
        #
        # if agent == 0:
        #     return max(valores)
        # else:
        #     return min(valores)

        # PROMPT:
        # "Ayudame a revisar que el código esté correcto y siga las indicaciones
        # de minimax, y también ayudame a entender cómo devolver la mejor acción."

        # CORRECCIÓN:
        # Se organizó el código para que minimax quede completamente dentro
        # de get_action y toda la lógica recursiva quede bien implementada.
        # Además, al final se recorren las acciones legales del drone y se
        # escoge la que produce el mayor valor minimax.

        num_agents = state.get_num_agents()

        def minimax(state: GameState, agent: int, depth: int) -> float:
            if state.is_win() or state.is_lose() or depth == 0:
                return self.evaluation_function(state)

            acciones = state.get_legal_actions(agent)
            if not acciones:
                return self.evaluation_function(state)

            siguiente_agente = (agent + 1) % num_agents

            if siguiente_agente == 0:
                siguiente_depth = depth - 1
            else:
                siguiente_depth = depth

            valores = [
                minimax(state.generate_successor(agent, accion), siguiente_agente, siguiente_depth)
                for accion in acciones
            ]

            if agent == 0:
                return max(valores)
            else:
                return min(valores)

        acciones = state.get_legal_actions(0)
        if not acciones:
            return None

        best_action = max(
            acciones,
            key=lambda a: minimax(
                state.generate_successor(0, a),
                (0 + 1) % num_agents,
                self.depth if num_agents > 1 else self.depth - 1
            )
        )

        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent with a mixed hunter model.

    Each hunter acts randomly with probability self.prob and greedily
    (worst-case / MIN) with probability 1 - self.prob.

    * When prob = 0:  behaves like Minimax (hunters always play optimally).
    * When prob = 1:  pure expectimax (hunters always play uniformly at random).
    * When 0 < prob < 1: weighted combination that correctly models the
      actual MixedHunterAgent used at game-play time.

    Chance node formula:
        value = (1 - p) * min(child_values) + p * mean(child_values)
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using expectimax with mixed hunter model.

        Tips:
        - Drone nodes are MAX (same as Minimax).
        - Hunter nodes are CHANCE with mixed model: the hunter acts greedily with
          probability (1 - self.prob) and uniformly at random with probability self.prob.
        - Mixed expected value = (1-p) * min(child_values) + p * mean(child_values).
        - When p=0 this reduces to Minimax; when p=1 it is pure uniform expectimax.
        - Do NOT prune in expectimax (unlike alpha-beta).
        - self.prob is set via the constructor argument prob.
        """
        # TODO: Implement your code here
        return None
