import random
from collections import defaultdict
from typing import Dict, List, Tuple


class QLearningAgent:
    def __init__(
        self,
        actions: List[str],
        alpha: float = 0.12,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.992,
    ):
        self.actions = list(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[Tuple[int, ...], Dict[str, float]] = defaultdict(
            lambda: {a: 0.0 for a in self.actions}
        )

    def q_values(self, state: Tuple[int, ...]) -> Dict[str, float]:
        return self.q_table[state]

    def choose_action(
        self,
        state: Tuple[int, ...],
        explore: bool = True,
        candidates: List[str] | None = None,
    ) -> str:
        actions = candidates if candidates is not None else self.actions

        # epsilon exploration
        if explore and random.random() < self.epsilon:
            return random.choice(actions)

        qvals = self.q_values(state)
        best_val = max(qvals[a] for a in actions)
        best_actions = [a for a in actions if qvals[a] == best_val]
        # Avoid idling on ties when movement options exist.
        if "STAY" in best_actions and len(best_actions) > 1:
            non_stay = [a for a in best_actions if a != "STAY"]
            if non_stay:
                best_actions = non_stay
        return random.choice(best_actions)

    def learn(
        self,
        state: Tuple[int, ...],
        action: str,
        reward: float,
        next_state: Tuple[int, ...],
        done: bool,
        next_candidates: List[str] | None = None,
    ) -> None:
        qvals = self.q_values(state)
        current_q = qvals[action]
        if done:
            next_best = 0.0
        else:
            next_q = self.q_values(next_state)
            if next_candidates:
                next_best = max(next_q[a] for a in next_candidates)
            else:
                next_best = max(next_q.values())
        target = reward + self.gamma * next_best
        qvals[action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
