import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pygame
from q_learning_agent import QLearningAgent

Pos = Tuple[int, int]


@dataclass
class RewardConfig:
    exit_with_key: float = 100.0
    pick_key: float = 20.0
    explore_new: float = 0.2
    step_cost: float = -0.2
    hit_wall: float = -5.0
    trap_penalty: float = -10.0
    death_penalty: float = -100.0
    approach_target: float = 0.25   # 距离变近的奖励
    away_target: float = -0.05      # 距离变远的小惩罚（不想惩罚可设 0.0）


@dataclass
class GameConfig:
    cols: int = 9
    rows: int = 9
    tile_size: int = 44
    max_hp: int = 100
    trap_damage: int = 20
    medkit_heal: int = 15
    trap_count: int = 4
    medkit_count: int = 3
    fps: int = 60
    sidebar_width: int = 320


class MazeKeyGame:
    WALL = 0
    FLOOR = 1

    ACTIONS: Dict[str, Tuple[int, int]] = {
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0),
        "STAY": (0, 0),
    }

    def __init__(self, config: Optional[GameConfig] = None):
        self.cfg = config or GameConfig()
        if self.cfg.cols % 2 == 0:
            self.cfg.cols += 1
        if self.cfg.rows % 2 == 0:
            self.cfg.rows += 1

        self.rewards = RewardConfig()
        pygame.init()
        pygame.display.set_caption("Bunny & Carrot Maze - Pastel Garden")

        self.map_w = self.cfg.cols * self.cfg.tile_size
        self.map_h = self.cfg.rows * self.cfg.tile_size
        # Keep a balanced layout: sidebar remains readable and map panel gets vertical padding.
        self.window_h = max(self.map_h + 24, 540)
        self.map_offset_y = (self.window_h - self.map_h) // 2
        self.screen = pygame.display.set_mode((self.map_w + self.cfg.sidebar_width, self.window_h))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("segoeui", 20)
        self.small_font = pygame.font.SysFont("segoeui", 16)
        self.big_font = pygame.font.SysFont("comicsansms", 30, bold=True)
        self.title_font = pygame.font.SysFont("comicsansms", 24, bold=True)

        self.agent = QLearningAgent(list(self.ACTIONS.keys()))
        self.agent_autoplay = False
        self.agent_thinking_cooldown = 0
        self.agent_action_delay_frames = 4
        self.agent_episodes_trained = 0
        self.agent_last_state_key: Tuple[int, ...] = tuple()
        self.last_action_source = "System"
        self.agent_action_log: deque[str] = deque(maxlen=8)
        self.episode_result_log: deque[str] = deque(maxlen=8)
        self.reward_positive_log: deque[str] = deque(maxlen=8)
        self.reward_negative_log: deque[str] = deque(maxlen=8)
        self.hp_positive_log: deque[str] = deque(maxlen=8)
        self.hp_negative_log: deque[str] = deque(maxlen=8)
        self.ui_buttons: Dict[str, pygame.Rect] = {}
        self.recent_positions: deque[Pos] = deque(maxlen=8)
        self.last_move_from_pos: Optional[Pos] = None
        self.current_world_snapshot: Optional[Dict[str, object]] = None
        self.last_trained_world_signature: Optional[Tuple[object, ...]] = None
        self.agent_auto_restart_delay_frames = 18
        self.last_reward_positive: List[str] = []
        self.last_reward_negative: List[str] = []
        self.last_hp_positive: List[str] = []
        self.last_hp_negative: List[str] = []
        self.last_reward_breakdown: List[Tuple[str, float]] = []
        self.difficulty_level = "easy"
        self.difficulty_profiles: Dict[str, Dict[str, float]] = {
            # Easier => fewer extra openings => fewer loops/branches.
            "easy": {
                "loop_ratio": 0.055,
                "junction_prob": 0.45,
                "corner_prob": 0.15,
                "spur_prob": 0.05,
                "second_pass_prob": 0.10,
            },
        }
        self.training_summary = "Press T to train Q-learning agent"
        self.print_state_to_console = True
        self.print_training_steps = False

        self.reset()

    def reset(self) -> None:
        self.grid = self._generate_maze(self.cfg.cols, self.cfg.rows)
        self.start = (1, 1)
        self.player = self.start
        self.visited: Set[Pos] = {self.start}

        self.hp = self.cfg.max_hp
        self.max_hp = self.cfg.max_hp
        self.has_key = False
        self.steps = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.last_action = "-"
        self.last_event = "Bunny starts the garden run"
        self.last_action_source = "System"
        self.done = False
        self.win = False
        self.tick_count = 0
        self.agent_thinking_cooldown = 0
        self.message_log: deque[str] = deque(maxlen=8)
        self.agent_action_log.clear()
        self.episode_result_log.clear()
        self.reward_positive_log.clear()
        self.reward_negative_log.clear()
        self.hp_positive_log.clear()
        self.hp_negative_log.clear()
        self.last_reward_positive = []
        self.last_reward_negative = []
        self.last_hp_positive = []
        self.last_hp_negative = []
        self.last_reward_breakdown = []
        self.recent_positions = deque([self.player], maxlen=8)
        self.last_move_from_pos = None

        self.key_pos, self.exit_pos = self._place_key_and_exit()
        self.traps = self._place_items(self.cfg.trap_count, exclude={self.start, self.key_pos, self.exit_pos})
        self.medkits = self._place_items(
            self.cfg.medkit_count, exclude={self.start, self.key_pos, self.exit_pos, *self.traps}
        )

        self.current_world_snapshot = self._snapshot_world()
        self.agent_last_state_key = self.get_agent_state()
        self._log("A new pastel garden maze appears")

    def _snapshot_world(self) -> Dict[str, object]:
        return {
            "grid": [row[:] for row in self.grid],
            "start": self.start,
            "key_pos": self.key_pos,
            "exit_pos": self.exit_pos,
            "traps": set(self.traps),
            "medkits": set(self.medkits),
        }

    def _world_signature(self, snapshot: Dict[str, object]) -> Tuple[object, ...]:
        grid_sig = tuple(tuple(row) for row in snapshot["grid"])  # type: ignore[index]
        traps_sig = tuple(sorted(snapshot["traps"]))  # type: ignore[arg-type]
        meds_sig = tuple(sorted(snapshot["medkits"]))  # type: ignore[arg-type]
        return (
            grid_sig,
            snapshot["start"],
            snapshot["key_pos"],
            snapshot["exit_pos"],
            traps_sig,
            meds_sig,
        )

    def _reset_episode_from_snapshot(self, snapshot: Dict[str, object], keep_logs: bool = False) -> None:
        self.grid = [row[:] for row in snapshot["grid"]]  # type: ignore[index]
        self.start = snapshot["start"]  # type: ignore[assignment]
        self.player = self.start
        self.visited = {self.start}

        self.hp = self.cfg.max_hp
        self.max_hp = self.cfg.max_hp
        self.has_key = False
        self.steps = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.last_action = "-"
        self.last_action_source = "System"
        self.last_event = "Episode reset"
        self.done = False
        self.win = False
        self.agent_thinking_cooldown = 0
        self.recent_positions = deque([self.player], maxlen=8)
        self.last_move_from_pos = None

        self.key_pos = snapshot["key_pos"]  # type: ignore[assignment]
        self.exit_pos = snapshot["exit_pos"]  # type: ignore[assignment]
        self.traps = set(snapshot["traps"])  # type: ignore[arg-type]
        self.medkits = set(snapshot["medkits"])  # type: ignore[arg-type]

        if not keep_logs:
            self.agent_action_log.clear()
            self.reward_positive_log.clear()
            self.reward_negative_log.clear()
            self.hp_positive_log.clear()
            self.hp_negative_log.clear()
            self.message_log = deque(maxlen=8)
        self.last_reward_positive = []
        self.last_reward_negative = []
        self.last_hp_positive = []
        self.last_hp_negative = []
        self.last_reward_breakdown = []
        self.agent_last_state_key = self.get_agent_state()

    def _log(self, msg: str) -> None:
        self.message_log.appendleft(msg)

    def _floor_cells(self) -> List[Pos]:
        cells: List[Pos] = []
        for y in range(self.cfg.rows):
            for x in range(self.cfg.cols):
                if self.grid[y][x] == self.FLOOR:
                    cells.append((x, y))
        return cells

    def _generate_maze(self, cols: int, rows: int) -> List[List[int]]:
        grid = [[self.WALL for _ in range(cols)] for _ in range(rows)]
        for y in range(1, rows, 2):
            for x in range(1, cols, 2):
                grid[y][x] = self.FLOOR

        stack = [(1, 1)]
        visited = {(1, 1)}
        dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]

        while stack:
            x, y = stack[-1]
            options = []
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 1 <= nx < cols - 1 and 1 <= ny < rows - 1 and (nx, ny) not in visited:
                    options.append((nx, ny, dx, dy))
            if not options:
                stack.pop()
                continue
            nx, ny, dx, dy = random.choice(options)
            grid[y + dy // 2][x + dx // 2] = self.FLOOR
            visited.add((nx, ny))
            stack.append((nx, ny))

        profile = self.difficulty_profiles.get(self.difficulty_level, self.difficulty_profiles["easy"])
        junction_prob = profile["junction_prob"]
        corner_prob = profile["corner_prob"]
        spur_prob = profile["spur_prob"]
        second_pass_prob = profile["second_pass_prob"]

        # Add many extra openings to create more loops / branches (less single-path feel).
        # Easy mode uses conservative extra openings for simpler navigation.
        # We run several passes so the maze keeps its structure but gains route options.
        def try_open_wall(x: int, y: int) -> bool:
            if grid[y][x] != self.WALL:
                return False

            up = grid[y - 1][x] == self.FLOOR
            down = grid[y + 1][x] == self.FLOOR
            left = grid[y][x - 1] == self.FLOOR
            right = grid[y][x + 1] == self.FLOOR
            count = up + down + left + right

            straight = (left and right) or (up and down)
            corner = count == 2 and not straight
            junction = count >= 3
            spur = count == 1

            # Prefer connectors/junctions to add loops first; allow a few spurs for side branches.
            if straight:
                grid[y][x] = self.FLOOR
                return True
            if junction and random.random() < junction_prob:
                grid[y][x] = self.FLOOR
                return True
            if corner and random.random() < corner_prob:
                grid[y][x] = self.FLOOR
                return True
            if spur and random.random() < spur_prob:
                grid[y][x] = self.FLOOR
                return True
            return False

        # Candidate walls between carved cells are most effective for route diversity.
        connector_candidates: List[Pos] = []
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if grid[y][x] != self.WALL:
                    continue
                # Focus on in-between cells (one odd, one even) to preserve the grid rhythm.
                if (x % 2) == (y % 2):
                    continue
                connector_candidates.append((x, y))

        random.shuffle(connector_candidates)
        target_opens = max(6, int(cols * rows * profile["loop_ratio"]))
        opened = 0

        # Main pass: aggressively add loops.
        for x, y in connector_candidates:
            if opened >= target_opens:
                break
            if try_open_wall(x, y):
                opened += 1

        # Second pass: top up openings if the first pass was conservative due to probabilities.
        if opened < target_opens:
            random.shuffle(connector_candidates)
            for x, y in connector_candidates:
                if opened >= target_opens:
                    break
                if grid[y][x] == self.WALL and random.random() < second_pass_prob:
                    if try_open_wall(x, y):
                        opened += 1

        return grid

    def _neighbors(self, pos: Pos) -> List[Pos]:
        x, y = pos
        out: List[Pos] = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.cfg.cols and 0 <= ny < self.cfg.rows and self.grid[ny][nx] == self.FLOOR:
                out.append((nx, ny))
        return out

    def _bfs_distances(self, start: Pos) -> Dict[Pos, int]:
        q = deque([start])
        dist = {start: 0}
        while q:
            cur = q.popleft()
            for nxt in self._neighbors(cur):
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        return dist

    def _place_key_and_exit(self) -> Tuple[Pos, Pos]:
        dist_start = self._bfs_distances(self.start)
        cells = [p for p in dist_start if p != self.start]
        cells.sort(key=lambda p: dist_start[p], reverse=True)
        key_pos = cells[0]

        dist_key = self._bfs_distances(key_pos)
        exits = [p for p in dist_key if p not in {self.start, key_pos}]
        exits.sort(key=lambda p: dist_key[p], reverse=True)
        exit_pos = exits[0]
        return key_pos, exit_pos

    def _place_items(self, count: int, exclude: Set[Pos]) -> Set[Pos]:
        pool = [p for p in self._floor_cells() if p not in exclude]
        random.shuffle(pool)
        return set(pool[: min(count, len(pool))])

    def _sign(self, v: int) -> int:
        return 0 if v == 0 else (1 if v > 0 else -1)

    def _blocked_flags(self, pos: Pos) -> Tuple[int, int, int, int]:
        x, y = pos
        flags: List[int] = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            blocked = int(
                not (0 <= nx < self.cfg.cols and 0 <= ny < self.cfg.rows)
                or self.grid[ny][nx] == self.WALL
            )
            flags.append(blocked)
        return tuple(flags)  # type: ignore[return-value]

    def _adjacent_item_flags(self, pos: Pos, items: Set[Pos]) -> Tuple[int, int, int, int]:
        x, y = pos
        flags = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            flags.append(int((x + dx, y + dy) in items))
        return tuple(flags)  # type: ignore[return-value]

    def _local_density(self, pos: Pos, items: Set[Pos], radius: int = 2) -> int:
        x0, y0 = pos
        cnt = 0
        for y in range(max(0, y0 - radius), min(self.cfg.rows, y0 + radius + 1)):
            for x in range(max(0, x0 - radius), min(self.cfg.cols, x0 + radius + 1)):
                if (x, y) in items and abs(x - x0) + abs(y - y0) <= radius:
                    cnt += 1
        return cnt

    def get_agent_state(self) -> Tuple[int, ...]:
        target = self.exit_pos if self.has_key else self.key_pos
        px, py = self.player
        tx, ty = target
        dx = tx - px
        dy = ty - py
        manhattan = abs(dx) + abs(dy)
        hp_bucket = min(3, self.hp // 25)
        dist_bucket = min(7, manhattan // 2)
        traps_near = min(3, self._local_density(self.player, self.traps, radius=2))
        meds_near = min(3, self._local_density(self.player, self.medkits, radius=2))
        # `self.player in self.visited` is always true, so use a meaningful local-exploration signal.
        unvisited_neighbors = min(4, sum(1 for p in self._neighbors(self.player) if p not in self.visited))
        state = (
            px,
            py,
            int(self.has_key),
            hp_bucket,
            self._sign(dx),
            self._sign(dy),
            dist_bucket,
            *self._blocked_flags(self.player),
            *self._adjacent_item_flags(self.player, self.traps),
            *self._adjacent_item_flags(self.player, self.medkits),
            traps_near,
            meds_near,
            unvisited_neighbors,
        )
        return state

    def _state_lines_for_ui(self) -> List[str]:
        target = self.exit_pos if self.has_key else self.key_pos
        state_key = self.get_agent_state()
        self.agent_last_state_key = state_key
        return [
            f"Pos: {self.player}  Target: {target}",
            f"HasCarrot: {int(self.has_key)}  HP: {self.hp}",
            # f"Done: {int(self.done)}  Win: {int(self.win)}",
            f"StateKey: {state_key}",
        ]

    def _record_action_ui(self, action_name: str, source: str) -> None:
        label = "AGENT" if source.startswith("Agent") else source.upper()
        self.agent_action_log.appendleft(f"S{self.steps:03d} {label}: {action_name}")

    def _record_episode_result(self, source: str) -> None:
        outcome = "WIN" if self.win else "LOSE"
        line = (
            f"EpisodeEnd {outcome} | done={int(self.done)} win={int(self.win)} | "
            f"src={source} | step={self.steps} | score={self.total_reward:+.1f}"
        )
        self.episode_result_log.appendleft(line)

    def _record_reward_logs(self, components: List[Tuple[str, float]]) -> None:
        self.last_reward_breakdown = components
        self.last_reward_positive = []
        self.last_reward_negative = []
        for name, value in components:
            line = f"S{self.steps:03d} {name}: {value:+.1f}"
            if value >= 0:
                self.last_reward_positive.append(line)
                self.reward_positive_log.appendleft(line)
            else:
                self.last_reward_negative.append(line)
                self.reward_negative_log.appendleft(line)

    def _record_hp_change(self, label: str, delta_hp: int) -> None:
        if delta_hp == 0:
            return
        line = f"S{self.steps:03d} {label}: {delta_hp:+d} HP"
        if delta_hp > 0:
            self.last_hp_positive.append(line)
            self.hp_positive_log.appendleft(line)
        else:
            self.last_hp_negative.append(line)
            self.hp_negative_log.appendleft(line)

    def _next_pos_for_action(self, pos: Pos, action_name: str) -> Pos:
        dx, dy = self.ACTIONS[action_name]
        return (pos[0] + dx, pos[1] + dy)

    def _is_walkable(self, pos: Pos) -> bool:
        x, y = pos
        return 0 <= x < self.cfg.cols and 0 <= y < self.cfg.rows and self.grid[y][x] == self.FLOOR

    def _choose_autoplay_action(self, state: Tuple[int, ...]) -> str:
        # Start from Q-values, but break ties / near-ties in a way that avoids 2-cell bouncing.
        qvals = self.agent.q_values(state)
        ranked = sorted(self.ACTIONS.keys(), key=lambda a: qvals[a], reverse=True)
        best_q = qvals[ranked[0]]
        candidate_actions = [a for a in ranked if qvals[a] >= best_q - 0.05]
        walkable_non_stay = [
            a for a in candidate_actions if a != "STAY" and self._is_walkable(self._next_pos_for_action(self.player, a))
        ]
        if walkable_non_stay:
            candidate_actions = walkable_non_stay

        target = self.exit_pos if self.has_key else self.key_pos

        def score(action_name: str) -> Tuple[float, float, float, float]:
            nxt = self._next_pos_for_action(self.player, action_name)
            walkable = action_name == "STAY" or self._is_walkable(nxt)
            if not walkable:
                return (-9999.0, -9999.0, -9999.0, -9999.0)
            q = qvals[action_name]
            backtrack_penalty = -2.0 if self.last_move_from_pos is not None and nxt == self.last_move_from_pos else 0.0
            recent_penalty = -0.35 if nxt in self.recent_positions else 0.0
            dist_bonus = 0.0
            if action_name != "STAY":
                cur_d = abs(self.player[0] - target[0]) + abs(self.player[1] - target[1])
                nxt_d = abs(nxt[0] - target[0]) + abs(nxt[1] - target[1])
                dist_bonus = 0.10 * (cur_d - nxt_d)  # prefer moves that reduce distance
            stay_penalty = -0.4 if action_name == "STAY" else 0.0
            total = q + backtrack_penalty + recent_penalty + dist_bonus + stay_penalty
            return (total, q, dist_bonus, recent_penalty)

        best_action = max(candidate_actions, key=score)
        return best_action

    def _print_state_transition(
        self,
        source: str,
        state_before: Tuple[int, ...],
        action: str,
        reward: float,
        state_after: Tuple[int, ...],
    ) -> None:
        if not self.print_state_to_console:
            return
        if source == "Agent(train)" and not self.print_training_steps:
            return
        print(
            (
                f"[STATE] source={source} step={self.steps} "
                f"s={state_before} a={action} s'={state_after} "
                f"hp={self.hp} carrot={int(self.has_key)} done={int(self.done)} win={int(self.win)} "
                f"event={self.last_event}"
            ),
            flush=True,
        )

    def agent_step_once(self, learn: bool = False, explore: bool = False) -> None:
        if self.done:
            return
        state = self.get_agent_state()

        if explore:
            # training/exploration: epsilon-greedy
            valid = self.agent_actions(self.player)
            action = self.agent.choose_action(state, explore=True, candidates=valid)
        else:
            # autoplay: PURE greedy from Q-table (learned policy)
            action = self.greedy_q_action(state)
       
        reward = self.step(action, source="Agent(train)" if learn else "Agent")
        next_state = self.get_agent_state()
        if learn:
            next_valid = self.agent_actions(self.player)
            self.agent.learn(
                state,
                action,
                reward,
                next_state,
                self.done,
                next_candidates=next_valid,
            )

    def _evaluate_greedy_policy(
        self,
        snapshot: Dict[str, object],
        episodes: int = 20,
        max_steps_per_episode: Optional[int] = None,
    ) -> Tuple[int, float]:
        max_steps = max_steps_per_episode or (self.cfg.cols * self.cfg.rows * 3)
        wins = 0
        reward_sum = 0.0
        old_print_flag = self.print_state_to_console
        self.print_state_to_console = False
        try:
            for _ in range(episodes):
                self._reset_episode_from_snapshot(snapshot)
                for _step in range(max_steps):
                    if self.done:
                        break
                    state = self.get_agent_state()
                    action = self.greedy_q_action(state)
                    self.step(action, source="Agent(eval)")
                wins += int(self.win)
                reward_sum += self.total_reward
        finally:
            self.print_state_to_console = old_print_flag
        return wins, reward_sum / max(1, episodes)

    def train_agent(self, episodes: int = 200, max_steps_per_episode: Optional[int] = None) -> None:
        was_autoplay = self.agent_autoplay
        self.agent_autoplay = False
        world_snapshot = self._snapshot_world()
        world_sig = self._world_signature(world_snapshot)
        # Train for the current maze layout so the agent can visibly solve THIS map.
        # Only clear Q-table when the map layout/items changed; allow repeated Train x200 to accumulate learning.
        if self.last_trained_world_signature != world_sig:
            self.agent.q_table.clear()
            self.last_trained_world_signature = world_sig
            self.agent.epsilon = 1.0
        max_steps = max_steps_per_episode or (self.cfg.cols * self.cfg.rows * 3)
        wins = 0
        total_reward_sum = 0.0
        best_reward = -10_000.0

        for _ in range(episodes):
            self._reset_episode_from_snapshot(world_snapshot)
            ep_reward = 0.0
            for _step in range(max_steps):
                state = self.get_agent_state()
                valid = self.agent_actions(self.player)
                action = self.agent.choose_action(state, explore=True, candidates=valid)

                reward = self.step(action, source="Agent(train)")
                next_state = self.get_agent_state()
                next_valid = self.agent_actions(self.player)

                self.agent.learn(
                    state,
                    action,
                    reward,
                    next_state,
                    self.done,
                    next_candidates=next_valid,
                )
                ep_reward += reward
                if self.done:
                    break                
        
            self.agent.decay_epsilon()
            total_reward_sum += ep_reward
            best_reward = max(best_reward, ep_reward)
            wins += int(self.win)

        eval_episodes = 20
        eval_wins, eval_avg_reward = self._evaluate_greedy_policy(
            world_snapshot,
            episodes=eval_episodes,
            max_steps_per_episode=max_steps,
        )
        self.agent_episodes_trained += episodes
        avg_reward = total_reward_sum / max(1, episodes)
        self.training_summary = (
            f"Q-learning trained {episodes} eps | total {self.agent_episodes_trained} | "
            f"train_win {wins}/{episodes} | train_avg {avg_reward:+.1f} | "
            f"eval_win {eval_wins}/{eval_episodes} | eval_avg {eval_avg_reward:+.1f} | "
            f"eps={self.agent.epsilon:.3f}"
        )
        self._reset_episode_from_snapshot(world_snapshot, keep_logs=True)
        self.current_world_snapshot = world_snapshot
        self.agent_autoplay = was_autoplay
        self._log(self.training_summary)
        self.agent_action_log.appendleft("TRAIN: " + self.training_summary[:52])

    def step(self, action_name: str, source: str = "Human") -> float:
        if self.done:
            return 0.0
        if action_name not in self.ACTIONS:
            raise ValueError(f"Unknown action: {action_name}")

        state_before = self.get_agent_state()
        target_before = self.exit_pos if self.has_key else self.key_pos
        old_dist = abs(self.player[0] - target_before[0]) + abs(self.player[1] - target_before[1])
        reward_components: List[Tuple[str, float]] = [("step", self.rewards.step_cost)]
        reward = self.rewards.step_cost
        events: List[str] = [f"Step {self.steps + 1}: {action_name}"]

        dx, dy = self.ACTIONS[action_name]
        x, y = self.player
        nx, ny = x + dx, y + dy
        self.steps += 1
        self.last_hp_positive = []
        self.last_hp_negative = []

        moved = True
        prev_pos = self.player
        if action_name == "STAY":
            events.append("Stay")
        elif not (0 <= nx < self.cfg.cols and 0 <= ny < self.cfg.rows) or self.grid[ny][nx] == self.WALL:
            reward += self.rewards.hit_wall
            reward_components.append(("hit_wall", self.rewards.hit_wall))
            events.append("Hit wall")
            moved = False
        else:
            self.player = (nx, ny)

            # distance shaping (only when move succeeds)
            new_dist = abs(self.player[0] - target_before[0]) + abs(self.player[1] - target_before[1])
            if new_dist < old_dist:
                reward += self.rewards.approach_target
                reward_components.append(("approach_target", self.rewards.approach_target))
                events.append("Closer to target")
            elif new_dist > old_dist and self.rewards.away_target != 0.0:
                reward += self.rewards.away_target
                reward_components.append(("away_target", self.rewards.away_target))
                events.append("Farther from target")
        if moved and self.player not in self.visited:
            self.visited.add(self.player)
            reward += self.rewards.explore_new
            reward_components.append(("explore", self.rewards.explore_new))
            events.append("Explore new tile")

        if moved and self.player == self.key_pos and not self.has_key:
            self.has_key = True
            reward += self.rewards.pick_key
            reward_components.append(("pick_carrot", self.rewards.pick_key))
            events.append("Picked carrot")

        if moved and self.player in self.medkits:
            old_hp = self.hp
            self.hp = min(self.max_hp, self.hp + self.cfg.medkit_heal)
            self.medkits.remove(self.player)
            hp_gain = self.hp - old_hp
            self._record_hp_change("potion", hp_gain)
            events.append(f"Potion +{hp_gain} HP")

        if moved and self.player in self.traps:
            self.hp -= self.cfg.trap_damage
            self.traps.remove(self.player)
            reward += self.rewards.trap_penalty
            reward_components.append(("poison", self.rewards.trap_penalty))
            self._record_hp_change("poison", -self.cfg.trap_damage)
            events.append(f"Poison trap -{self.cfg.trap_damage} HP")

        if self.hp <= 0:
            self.hp = 0
            reward += self.rewards.death_penalty
            reward_components.append(("death", self.rewards.death_penalty))
            self.done = True
            self.win = False
            events.append("Dead")

        if not self.done and moved and self.player == self.exit_pos:
            if self.has_key:
                reward += self.rewards.exit_with_key
                reward_components.append(("exit", self.rewards.exit_with_key))
                self.done = True
                self.win = True
                events.append("Cottage gate reached")
            else:
                events.append("Gate locked (need carrot)")

        self.last_reward = reward
        self.total_reward += reward
        self.last_action = action_name
        self.last_action_source = source
        self.last_event = " | ".join(events)
        if moved and self.player != prev_pos:
            self.last_move_from_pos = prev_pos
            self.recent_positions.append(self.player)
        self._record_action_ui(action_name, source)
        if self.done and source not in {"Agent(train)", "Agent(eval)"}:
            self._record_episode_result(source)
        self._record_reward_logs(reward_components)
        self.agent_last_state_key = self.get_agent_state()
        self._log(self.last_event)
        self._print_state_transition(source, state_before, action_name, reward, self.agent_last_state_key)
        return reward

    def _cell_rect(self, pos: Pos) -> pygame.Rect:
        x, y = pos
        ts = self.cfg.tile_size
        return pygame.Rect(x * ts, y * ts + self.map_offset_y, ts, ts)

    def _draw_brick_tile(self, rect: pygame.Rect, wall: bool, pulse: float, visited: bool = False) -> None:
        if wall:
            outer = (252, 193, 166)
            inner = (242, 174, 145)
            seam = (216, 142, 121)
            shade = (232, 160, 136)
            pygame.draw.rect(self.screen, outer, rect, border_radius=7)
            inset = rect.inflate(-2, -2)
            pygame.draw.rect(self.screen, inner, inset, border_radius=7)
            pygame.draw.rect(self.screen, seam, inset, 1, border_radius=7)

            row_h = max(6, inset.h // 2)
            y_mid = inset.y + row_h
            pygame.draw.line(self.screen, seam, (inset.x + 2, y_mid), (inset.right - 3, y_mid), 1)

            top_xs = [inset.x + inset.w // 2]
            bottom_xs = [inset.x + inset.w // 4, inset.x + (inset.w * 3) // 4]
            for sx in top_xs:
                pygame.draw.line(self.screen, seam, (sx, inset.y + 2), (sx, y_mid - 1), 1)
            for sx in bottom_xs:
                pygame.draw.line(self.screen, seam, (sx, y_mid + 1), (sx, inset.bottom - 3), 1)

            shine_y = inset.y + 3 + int(pulse * 1)
            pygame.draw.line(self.screen, (255, 216, 194), (inset.x + 4, shine_y), (inset.right - 5, shine_y), 1)
            pygame.draw.line(self.screen, shade, (inset.x + 3, inset.bottom - 4), (inset.right - 4, inset.bottom - 4), 1)
        else:
            base = (243, 242, 222)
            if visited:
                base = (255, 247, 190)
            pygame.draw.rect(self.screen, base, rect)
            pygame.draw.rect(self.screen, (219, 212, 187), rect, 1)
            pygame.draw.rect(self.screen, (177, 230, 159), (rect.x, rect.y, rect.w, 4))
            # Subtle path tile texture lines
            pygame.draw.line(self.screen, (232, 226, 201), (rect.x + 5, rect.y + 8), (rect.right - 5, rect.y + 8), 1)
            pygame.draw.line(self.screen, (235, 230, 208), (rect.x + 4, rect.bottom - 8), (rect.right - 4, rect.bottom - 8), 1)
            pygame.draw.line(self.screen, (226, 220, 196), (rect.x + 8, rect.y + 5), (rect.x + 8, rect.bottom - 5), 1)

    def _draw_flower(self, x: int, y: int, size: int = 4, phase: float = 0.0) -> None:
        sway = int(round(math.sin(phase) * 2))
        bob = int(round(math.sin(phase * 1.7) * 1))
        stem_h = size * 2 + 3
        stem_top = (x + sway, y + bob + 1)
        stem_bottom = (x, y + stem_h + 2)
        pygame.draw.line(self.screen, (112, 196, 118), stem_top, stem_bottom, 2)
        pygame.draw.line(self.screen, (146, 220, 142), (stem_top[0] + 1, stem_top[1]), (stem_bottom[0] + 1, stem_bottom[1]), 1)

        cx = x + sway
        cy = y + bob
        petal_r = max(2, size // 2)
        petals = [(255, 182, 209), (255, 212, 138), (186, 230, 255), (210, 200, 255)]
        petal_offsets = [(0, -size), (size, 0), (0, size), (-size, 0)]
        for i, (dx, dy) in enumerate(petal_offsets):
            pygame.draw.circle(self.screen, petals[i % len(petals)], (cx + dx, cy + dy), petal_r)
        pygame.draw.circle(self.screen, (255, 242, 150), (cx, cy), petal_r)
        # leaf
        pygame.draw.ellipse(self.screen, (146, 222, 147), (x - 4, y + size + 1, 6, 3))

    def _draw_mushroom(self, x: int, y: int, phase: float = 0.0, scale: float = 1.0) -> None:
        bob = int(round(math.sin(phase) * 1))
        sway = int(round(math.sin(phase * 0.6) * 1))
        stem_w = max(3, int(5 * scale))
        stem_h = max(5, int(7 * scale))
        cap_w = max(8, int(14 * scale))
        cap_h = max(6, int(9 * scale))

        stem = pygame.Rect(x - stem_w // 2 + sway, y - stem_h // 2 + 3 + bob, stem_w, stem_h)
        pygame.draw.rect(self.screen, (255, 244, 228), stem, border_radius=3)
        pygame.draw.rect(self.screen, (230, 214, 199), stem, 1, border_radius=3)

        cap = pygame.Rect(x - cap_w // 2 + sway, y - cap_h // 2 - 2 + bob, cap_w, cap_h)
        pygame.draw.ellipse(self.screen, (248, 132, 142), cap)
        pygame.draw.ellipse(self.screen, (227, 106, 118), cap, 1)
        pygame.draw.arc(self.screen, (255, 180, 188), cap.inflate(-2, -2), math.pi, 2 * math.pi, 1)
        for dx, dy, rr in [(-3, -1, 2), (2, 0, 2), (0, 2, 1)]:
            pygame.draw.circle(self.screen, (255, 245, 247), (x + dx + sway, y + dy - 3 + bob), rr)

        grass_y = stem.bottom + 1
        pygame.draw.line(self.screen, (127, 203, 123), (x - 6, grass_y), (x - 2, grass_y - 2), 1)
        pygame.draw.line(self.screen, (127, 203, 123), (x + 6, grass_y), (x + 2, grass_y - 2), 1)

    def _draw_bunny_icon(self, rect: pygame.Rect, pulse: float) -> None:
        cx, cy = rect.center
        shadow = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (80, 60, 60, 70), (rect.w // 2 - 10, rect.h // 2 + 6, 20, 10))
        self.screen.blit(shadow, rect.topleft)

        ear_w = 7
        ear_h = 14 + int(2 * pulse)
        left_ear = pygame.Rect(cx - 10, cy - 18 - int(2 * pulse), ear_w, ear_h)
        right_ear = pygame.Rect(cx + 3, cy - 18 - int(2 * pulse), ear_w, ear_h)
        for ear in (left_ear, right_ear):
            pygame.draw.ellipse(self.screen, (249, 246, 252), ear)
            inner = ear.inflate(-4, -4)
            inner.y += 2
            pygame.draw.ellipse(self.screen, (255, 205, 225), inner)

        body = pygame.Rect(cx - 9, cy - 3, 18, 18)
        head = pygame.Rect(cx - 10, cy - 12, 20, 18)
        pygame.draw.ellipse(self.screen, (252, 249, 253), body)
        pygame.draw.ellipse(self.screen, (252, 249, 253), head)
        pygame.draw.circle(self.screen, (255, 255, 255), (cx + 8, cy + 6), 4)  # tail
        pygame.draw.circle(self.screen, (75, 75, 85), (cx - 4, cy - 6), 1)
        pygame.draw.circle(self.screen, (75, 75, 85), (cx + 4, cy - 6), 1)
        pygame.draw.circle(self.screen, (255, 170, 190), (cx, cy - 3), 2)
        pygame.draw.circle(self.screen, (255, 205, 220), (cx - 7, cy - 2), 2)
        pygame.draw.circle(self.screen, (255, 205, 220), (cx + 7, cy - 2), 2)
        pygame.draw.ellipse(self.screen, (210, 190, 205), head, 1)
        pygame.draw.ellipse(self.screen, (210, 190, 205), body, 1)

    def _draw_carrot_icon(self, rect: pygame.Rect, pulse: float) -> None:
        cx, cy = rect.center
        bob = int(2 * pulse)
        glow = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        pygame.draw.circle(glow, (255, 223, 164, 70), (rect.w // 2, rect.h // 2), 12 + bob)
        self.screen.blit(glow, rect.topleft)
        tip = (cx, cy + 10)
        left = (cx - 8, cy - 2)
        right = (cx + 8, cy - 2)
        pygame.draw.polygon(self.screen, (255, 154, 94), [left, right, tip])
        pygame.draw.polygon(self.screen, (232, 122, 74), [left, right, tip], 1)
        for oy in (-1, 3, 7):
            pygame.draw.line(self.screen, (255, 193, 131), (cx - 4, cy + oy), (cx + 4, cy + oy + 1), 1)
        pygame.draw.ellipse(self.screen, (121, 210, 119), (cx - 8, cy - 10 - bob, 7, 10))
        pygame.draw.ellipse(self.screen, (148, 229, 142), (cx - 1, cy - 12 - bob, 7, 11))
        pygame.draw.ellipse(self.screen, (105, 198, 104), (cx + 5, cy - 10 - bob, 7, 10))

    def _draw_cute_trap_icon(self, rect: pygame.Rect, pulse: float) -> None:
        cx, cy = rect.center
        # Thorn bush trap
        bush = pygame.Rect(cx - 10, cy - 5, 20, 14)
        pygame.draw.ellipse(self.screen, (201, 120, 146), bush)
        pygame.draw.ellipse(self.screen, (226, 154, 176), bush.inflate(-4, -4))
        spikes = [
            [(cx - 8, cy - 2), (cx - 12, cy - 10 - int(pulse * 2)), (cx - 4, cy - 6)],
            [(cx, cy - 4), (cx, cy - 13 - int(pulse * 2)), (cx + 4, cy - 5)],
            [(cx + 8, cy - 2), (cx + 12, cy - 10 - int(pulse * 2)), (cx + 4, cy - 6)],
        ]
        for tri in spikes:
            pygame.draw.polygon(self.screen, (187, 76, 108), tri)
        pygame.draw.circle(self.screen, (85, 70, 80), (cx - 3, cy - 1), 1)
        pygame.draw.circle(self.screen, (85, 70, 80), (cx + 3, cy - 1), 1)
        pygame.draw.arc(self.screen, (85, 70, 80), (cx - 4, cy + 1, 8, 5), math.pi, 2 * math.pi, 1)

    def _draw_cute_heal_icon(self, rect: pygame.Rect, pulse: float) -> None:
        cx, cy = rect.center
        # Potion bottle
        bottle = pygame.Rect(cx - 9, cy - 8, 18, 18)
        neck = pygame.Rect(cx - 4, cy - 12, 8, 6)
        pygame.draw.rect(self.screen, (233, 247, 255), bottle, border_radius=6)
        pygame.draw.rect(self.screen, (233, 247, 255), neck, border_radius=3)
        pygame.draw.rect(self.screen, (168, 230, 189), (bottle.x + 2, bottle.y + 8, bottle.w - 4, bottle.h - 4), border_radius=5)
        pygame.draw.rect(self.screen, (149, 214, 173), (bottle.x + 3, bottle.y + 10, bottle.w - 6, bottle.h - 6), border_radius=4)
        pygame.draw.rect(self.screen, (231, 193, 145), (neck.x, neck.y, neck.w, neck.h), border_radius=3)
        pygame.draw.circle(self.screen, (255, 255, 255), (cx - 3, cy + 3), 2 + int(pulse))
        pygame.draw.circle(self.screen, (255, 240, 250), (cx + 3, cy - 1), 2)
        pygame.draw.rect(self.screen, (154, 194, 170), bottle, 1, border_radius=6)

    def _draw_exit_house_icon(self, rect: pygame.Rect, unlocked: bool, pulse: float) -> None:
        base = rect.inflate(-6, -6)
        cx, cy = base.center
        roof = [(base.x + 2, base.y + 8), (cx, base.y), (base.right - 2, base.y + 8)]
        wall_color = (255, 244, 214) if unlocked else (220, 222, 226)
        roof_color = (248, 160, 166) if unlocked else (176, 178, 186)
        pygame.draw.polygon(self.screen, roof_color, roof)
        pygame.draw.polygon(self.screen, (214, 120, 130), roof, 1)
        house = pygame.Rect(base.x + 4, base.y + 8, base.w - 8, base.h - 10)
        pygame.draw.rect(self.screen, wall_color, house, border_radius=5)
        pygame.draw.rect(self.screen, (210, 198, 178), house, 1, border_radius=5)
        door = pygame.Rect(cx - 4, house.bottom - 9, 8, 10)
        door_color = (120, 190, 255) if unlocked else (130, 142, 160)
        pygame.draw.rect(self.screen, door_color, door, border_radius=3)
        pygame.draw.circle(self.screen, (255, 255, 255), (door.right - 2, door.centery), 1)
        if unlocked:
            halo = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            pygame.draw.circle(halo, (255, 239, 177, 70), (rect.w // 2, rect.h // 2), 12 + int(2 * pulse))
            self.screen.blit(halo, rect.topleft)

    def _draw_background(self) -> None:
        self.screen.fill((195, 234, 255))

        # Sky gradient.
        for i in range(self.window_h):
            t = i / max(1, self.window_h - 1)
            r = int(193 + 32 * t)
            g = int(234 + 14 * t)
            b = int(255 - 18 * t)
            pygame.draw.line(self.screen, (r, g, b), (0, i), (self.map_w, i))

        # Soft hills.
        pygame.draw.ellipse(self.screen, (170, 233, 177), (-80, self.window_h - 150, int(self.map_w * 0.8), 210))
        pygame.draw.ellipse(
            self.screen, (146, 221, 159), (int(self.map_w * 0.25), self.window_h - 145, int(self.map_w * 0.9), 200)
        )
        pygame.draw.ellipse(
            self.screen, (189, 241, 193), (int(self.map_w * 0.55), self.window_h - 120, int(self.map_w * 0.55), 160)
        )

        # Board container for better composition on smaller maps.
        board = pygame.Rect(0, self.map_offset_y, self.map_w, self.map_h)
        shadow = pygame.Rect(board.x + 4, board.y + 4, board.w, board.h)
        shadow_surf = pygame.Surface((shadow.w, shadow.h), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surf, (78, 110, 120, 35), shadow_surf.get_rect(), border_radius=16)
        self.screen.blit(shadow_surf, shadow.topleft)
        pygame.draw.rect(self.screen, (255, 252, 245), board, border_radius=16)
        pygame.draw.rect(self.screen, (227, 216, 198), board, 2, border_radius=16)

        # Clouds.
        cloud_offset = int(8 * math.sin(self.tick_count * 0.01))
        cloud_specs = [
            (90 + cloud_offset, 55, 1.0),
            (270 - cloud_offset, 80, 0.8),
            (470 + cloud_offset, 48, 1.1),
        ]
        for cx, cy, scale in cloud_specs:
            if cx > self.map_w - 20:
                continue
            for dx, dy, rr in [(-20, 4, 14), (-4, -2, 17), (14, 2, 13), (28, 6, 10)]:
                pygame.draw.circle(self.screen, (252, 252, 255), (int(cx + dx * scale), int(cy + dy * scale)), int(rr * scale))
            pygame.draw.ellipse(
                self.screen,
                (242, 247, 255),
                (int(cx - 28 * scale), int(cy + 4 * scale), int(62 * scale), int(16 * scale)),
            )

    def _draw_map(self) -> None:
        ts = self.cfg.tile_size
        pulse = (math.sin(self.tick_count * 0.08) + 1.0) * 0.5
        occupied_cells = set(self.traps) | set(self.medkits) | {self.player, self.key_pos, self.exit_pos}

        for y in range(self.cfg.rows):
            for x in range(self.cfg.cols):
                rect = self._cell_rect((x, y))
                if self.grid[y][x] == self.WALL:
                    self._draw_brick_tile(rect, wall=True, pulse=pulse)
                else:
                    self._draw_brick_tile(rect, wall=False, pulse=pulse, visited=(x, y) in self.visited)

                    if (x, y) not in occupied_cells:
                        phase = self.tick_count * 0.12 + x * 0.61 + y * 0.37
                        # Decorative animated flora on floor tiles.
                        if (x * 3 + y * 5) % 17 == 0:
                            self._draw_mushroom(rect.centerx + 7, rect.centery + 6, phase=phase, scale=0.9)
                        elif (x * 7 + y * 11) % 19 == 0:
                            self._draw_mushroom(rect.centerx - 6, rect.centery + 5, phase=phase + 0.8, scale=0.75)
                            self._draw_flower(rect.centerx + 7, rect.centery + 6, 3, phase=phase + 1.1)
                        elif (x + y) % 7 == 0:
                            self._draw_flower(rect.centerx + 8, rect.centery + 6, 3, phase=phase)
                        elif (x * 5 + y * 2) % 23 == 0:
                            self._draw_flower(rect.centerx - 7, rect.centery + 5, 3, phase=phase + 0.4)

        for trap in self.traps:
            self._draw_cute_trap_icon(self._cell_rect(trap), pulse)

        for med in self.medkits:
            self._draw_cute_heal_icon(self._cell_rect(med), pulse)

        if not self.has_key:
            self._draw_carrot_icon(self._cell_rect(self.key_pos), pulse)

        self._draw_exit_house_icon(self._cell_rect(self.exit_pos), self.has_key, pulse)
        self._draw_bunny_icon(self._cell_rect(self.player), pulse)
        pygame.draw.rect(
            self.screen,
            (235, 223, 204),
            pygame.Rect(0, self.map_offset_y, self.map_w, self.map_h),
            3,
            border_radius=16,
        )

    def _draw_path_hints(self) -> None:
        # Candy-color dotted hints (visual only).
        pts = [self.player]
        if not self.has_key:
            pts.append(self.key_pos)
        pts.append(self.exit_pos)

        for a, b in zip(pts, pts[1:]):
            ax, ay = self._cell_rect(a).center
            bx, by = self._cell_rect(b).center
            color = (255, 185, 116) if b == self.key_pos else (145, 199, 255)
            steps = max(6, int(math.hypot(bx - ax, by - ay) // 14))
            for i in range(1, steps):
                if i % 2 == 0:
                    continue
                t = i / steps
                px = int(ax + (bx - ax) * t)
                py = int(ay + (by - ay) * t)
                pygame.draw.circle(self.screen, color, (px, py), 2)
                pygame.draw.circle(self.screen, (255, 255, 255), (px - 1, py - 1), 1)

    def _draw_sidebar(self) -> None:
        x0 = self.map_w
        sidebar_h = self.screen.get_height()
        panel = pygame.Rect(x0, 0, self.cfg.sidebar_width, sidebar_h)
        pygame.draw.rect(self.screen, (240, 232, 252), panel)
        pygame.draw.line(self.screen, (206, 191, 230), (x0, 0), (x0, sidebar_h), 3)
        self.ui_buttons = {}
        mouse_pos = pygame.mouse.get_pos()

        def card(rect: pygame.Rect, fill: Tuple[int, int, int]) -> None:
            pygame.draw.rect(self.screen, fill, rect, border_radius=14)
            pygame.draw.rect(self.screen, (216, 202, 237), rect, 2, border_radius=14)

        def button(rect: pygame.Rect, label: str, key: str, active: bool = False) -> None:
            hovered = rect.collidepoint(mouse_pos)
            base = (236, 245, 255)
            if key == "train":
                base = (255, 242, 220)
            elif key == "auto":
                base = (226, 252, 229) if active else (242, 246, 250)
            elif key == "reset":
                base = (255, 236, 240)
            if hovered:
                base = tuple(min(255, c + 8) for c in base)
            pygame.draw.rect(self.screen, base, rect, border_radius=10)
            if active and key == "auto":
                border = (153, 201, 158)
            else:
                border = (211, 203, 228)
            pygame.draw.rect(self.screen, border, rect, 2, border_radius=10)
            txt = self.small_font.render(label, True, (90, 92, 108))
            self.screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))
            self.ui_buttons[key] = rect.copy()

        margin = 12
        card_w = self.cfg.sidebar_width - margin * 2
        y = 12

        header = pygame.Rect(x0 + margin, y, card_w, 74)
        card(header, (255, 247, 225))
        title = self.title_font.render("Bunny Carrot Run", True, (93, 82, 109))
        subtitle = self.small_font.render("Pastel Garden Edition", True, (115, 157, 120))
        self.screen.blit(title, (header.x + 14, header.y + 10))
        self.screen.blit(subtitle, (header.x + 14, header.y + 42))
        self._draw_bunny_icon(pygame.Rect(header.right - 58, header.y + 10, 28, 28), (math.sin(self.tick_count * 0.1) + 1) * 0.5)
        self._draw_carrot_icon(pygame.Rect(header.right - 30, header.y + 12, 22, 22), (math.sin(self.tick_count * 0.1) + 1) * 0.5)
        y = header.bottom + 8

        hp_box = pygame.Rect(x0 + margin, y, card_w, 50)
        card(hp_box, (236, 252, 236))
        hp_label = self.small_font.render(
            f"Bunny HP  {self.hp}/{self.max_hp}  |  Score {self.total_reward:+.1f}",
            True,
            (88, 103, 89),
        )
        self.screen.blit(hp_label, (hp_box.x + 12, hp_box.y + 8))
        bar = pygame.Rect(hp_box.x + 12, hp_box.y + 26, hp_box.w - 24, 14)
        pygame.draw.rect(self.screen, (219, 232, 220), bar, border_radius=7)
        fill = bar.copy()
        fill.w = int(bar.w * (self.hp / self.max_hp if self.max_hp else 0))
        hp_color = (255, 148, 154) if self.hp < 40 else (255, 182, 120)
        pygame.draw.rect(self.screen, hp_color, fill, border_radius=7)
        pygame.draw.rect(self.screen, (192, 214, 194), bar, 1, border_radius=7)
        y = hp_box.bottom + 8

        # Action buttons (mouse-friendly controls for training / auto-play / reset)
        btn_box = pygame.Rect(x0 + margin, y, card_w, 52)
        card(btn_box, (250, 246, 255))
        gap = 8
        btn_w = (btn_box.w - 24 - gap * 2) // 3
        btn_h = 30
        by = btn_box.y + 11
        bx = btn_box.x + 12
        button(pygame.Rect(bx, by, btn_w, btn_h), "Train x200", "train")
        button(pygame.Rect(bx + btn_w + gap, by, btn_w, btn_h), "Auto ON" if self.agent_autoplay else "Auto OFF", "auto", active=self.agent_autoplay)
        button(pygame.Rect(bx + (btn_w + gap) * 2, by, btn_w, btn_h), "Reset", "reset")
        y = btn_box.bottom + 8

        def draw_lines_in_box(
            box: pygame.Rect,
            title_text: str,
            lines: List[str],
            fill: Tuple[int, int, int],
            title_color: Tuple[int, int, int] = (118, 108, 152),
            text_color: Tuple[int, int, int] = (96, 98, 114),
        ) -> None:
            card(box, fill)
            title_surf = self.small_font.render(title_text, True, title_color)
            self.screen.blit(title_surf, (box.x + 12, box.y + 10))
            yy = box.y + 32
            for raw in lines:
                for line in self._wrap_text(raw, box.w - 24):
                    surf = self.small_font.render(line, True, text_color)
                    self.screen.blit(surf, (box.x + 12, yy))
                    yy += 18
                    if yy > box.bottom - 10:
                        return

        state_lines = self._state_lines_for_ui()
        state_lines.extend(
            [
                f"Carrot: {int(self.has_key)} | Steps: {self.steps}",
                f"Score: {self.total_reward:+.1f} | Last: {self.last_reward:+.1f}",
                f"TargetMode: {'Exit' if self.has_key else 'Carrot'}",
            ]
        )
        state_box = pygame.Rect(x0 + margin, y, card_w, 96)
        draw_lines_in_box(state_box, "Game Current State", state_lines, (236, 247, 255))
        y = state_box.bottom + 8

        action_lines = [
            f"Mode: {'Agent Auto' if self.agent_autoplay else 'Manual'}",
            # f"Done: {int(self.done)} | Win: {int(self.win)}",
            f"Last action: {self.last_action} ({self.last_action_source})",
            f"Eps: {self.agent.epsilon:.3f}",
            f"alpha={self.agent.alpha:.2f} | gamma={self.agent.gamma:.2f}",
            f"Episodes: {self.agent_episodes_trained}",
            "Keyboard: T/G/R (optional)",
        ]
        action_lines.extend(list(self.episode_result_log)[:1])
        action_lines.extend(list(self.agent_action_log)[:2])
        action_box = pygame.Rect(x0 + margin, y, card_w, 104)
        card(action_box, (255, 240, 244))
        title_surf = self.small_font.render("Agent Action", True, (118, 108, 152))
        self.screen.blit(title_surf, (action_box.x + 12, action_box.y + 10))

        yy = action_box.y + 34
        for raw in action_lines:
            for line in self._wrap_text(raw, action_box.w - 24):
                surf = self.small_font.render(line, True, (96, 98, 114))
                self.screen.blit(surf, (action_box.x + 12, yy))
                yy += 18
                if yy > action_box.bottom - 10:
                    break
            if yy > action_box.bottom - 10:
                break
        y = action_box.bottom + 8

        pos_lines: List[str] = []
        for ln in self.last_reward_positive[:]:
            pos_lines.append("Reward " + ln)
        for ln in list(self.reward_positive_log)[:4]:
            pos_lines.append("Reward " + ln)
        for ln in self.last_hp_positive[:]:
            pos_lines.append("HP " + ln)
        for ln in list(self.hp_positive_log)[:4]:
            pos_lines.append("HP " + ln)
        if not pos_lines:
            pos_lines = ["No rewards / HP gain yet"]
        # de-duplicate immediate duplicates while preserving order
        seen: Set[str] = set()
        pos_lines = [ln for ln in pos_lines if not (ln in seen or seen.add(ln))]
        neg_lines: List[str] = []
        for ln in self.last_reward_negative[:]:
            neg_lines.append("Penalty " + ln)
        for ln in list(self.reward_negative_log)[:5]:
            neg_lines.append("Penalty " + ln)
        for ln in self.last_hp_negative[:]:
            neg_lines.append("HP " + ln)
        for ln in list(self.hp_negative_log)[:6]:
            neg_lines.append("HP " + ln)
        if not neg_lines:
            neg_lines = ["No penalties / HP loss yet"]
        seen2: Set[str] = set()
        neg_lines = [ln for ln in neg_lines if not (ln in seen2 or seen2.add(ln))]

        gap2 = 8
        half_w = (card_w - gap2) // 2
        bottom_h = max(72, sidebar_h - y - 12)
        reward_box = pygame.Rect(x0 + margin, y, half_w, bottom_h)
        draw_lines_in_box(
            reward_box,
            "Rewards / HP Gain",
            pos_lines,
            (239, 252, 239),
            title_color=(93, 153, 103),
            text_color=(86, 111, 91),
        )
        penalty_box = pygame.Rect(reward_box.right + gap2, y, card_w - half_w - gap2, bottom_h)
        draw_lines_in_box(
            penalty_box,
            "Penalties / HP Loss",
            neg_lines,
            (255, 241, 241),
            title_color=(194, 99, 115),
            text_color=(118, 86, 94),
        )

    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        words = text.split()
        if not words:
            return [""]
        lines: List[str] = []
        cur = words[0]
        for w in words[1:]:
            candidate = cur + " " + w
            if self.small_font.size(candidate)[0] <= max_width:
                cur = candidate
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        return lines

    def valid_actions(self, pos: Pos) -> List[str]:
        """Return actions that are feasible from pos (including STAY)."""
        acts: List[str] = []
        for a in self.ACTIONS.keys():
            if a == "STAY":
                acts.append(a)
                continue
            nxt = self._next_pos_for_action(pos, a)
            if self._is_walkable(nxt):
                acts.append(a)
        return acts

    def agent_actions(self, pos: Pos) -> List[str]:
        """
        Action set used by RL updates/inference.
        Prefer movement actions to avoid idling loops; keep STAY only as fallback.
        """
        acts = self.valid_actions(pos)
        moving = [a for a in acts if a != "STAY"]
        return moving if moving else acts


    def greedy_q_action(self, state: Tuple[int, ...]) -> str:
        """
        Pure greedy policy based on Q-table.
        Pick the valid action with the highest Q(s,a). Break ties randomly.
        """
        qvals = self.agent.q_values(state)
        candidates = self.agent_actions(self.player)

        best_val = max(qvals[a] for a in candidates)
        best_actions = [a for a in candidates if qvals[a] == best_val]
        return random.choice(best_actions)

    # def _draw_banner(self) -> None:
    #     banner = pygame.Rect(10, 10, self.map_w - 20, 38)
    #     pygame.draw.rect(self.screen, (255, 251, 232), banner, border_radius=14)
    #     pygame.draw.rect(self.screen, (233, 218, 180), banner, 2, border_radius=14)
    #     target = "Hop to the cottage gate" if self.has_key else "Find the carrot first"
    #     txt = self.small_font.render(
    #         f"Goal: {target} | T=Train Q-learning | G=Agent Auto | Bunny | Poison | Potion",
    #         True,
    #         (110, 104, 92),
    #     )
    #     self.screen.blit(txt, (banner.x + 12, banner.y + 10))

    def _draw_end_overlay(self) -> None:
        overlay = pygame.Surface((self.map_w, self.window_h), pygame.SRCALPHA)
        overlay.fill((255, 242, 248, 135))
        self.screen.blit(overlay, (0, 0))

        text = "BUNNY WINS!" if self.win else "BUNNY TIRED!"
        color = (114, 212, 145) if self.win else (244, 129, 147)
        t1 = self.big_font.render(text, True, color)
        t2 = self.font.render("Press R to restart", True, (117, 110, 124))
        t3 = self.small_font.render(f"Final Reward: {self.total_reward:+.1f}", True, (224, 143, 93))

        cx = self.map_w // 2
        cy = self.window_h // 2
        panel = pygame.Rect(cx - 150, cy - 70, 300, 120)
        pygame.draw.rect(self.screen, (255, 255, 255), panel, border_radius=18)
        pygame.draw.rect(self.screen, (236, 214, 232), panel, 2, border_radius=18)
        self.screen.blit(t1, (cx - t1.get_width() // 2, cy - 44))
        self.screen.blit(t2, (cx - t2.get_width() // 2, cy + 2))
        self.screen.blit(t3, (cx - t3.get_width() // 2, cy + 34))

    def toggle_agent_autoplay(self) -> None:
        self.agent_autoplay = not self.agent_autoplay
        self.agent_thinking_cooldown = 0
        self.training_summary = (
            f"Agent autoplay {'ON' if self.agent_autoplay else 'OFF'} | "
            f"episodes={self.agent_episodes_trained} | eps={self.agent.epsilon:.3f}"
        )
        self._log(self.training_summary)

    def handle_mouse_click(self, pos: Tuple[int, int]) -> None:
        for key, rect in self.ui_buttons.items():
            if not rect.collidepoint(pos):
                continue
            if key == "train":
                self.train_agent(episodes=200)
                return
            if key == "auto":
                self.toggle_agent_autoplay()
                return
            if key == "reset":
                self.reset()
                return

    def draw(self) -> None:
        self._draw_background()
        self._draw_map()
        self._draw_path_hints()
        # self._draw_banner()
        self._draw_sidebar()
        if self.done:
            self._draw_end_overlay()
        pygame.display.flip()

    def _autoplay_restart_episode(self) -> None:
        if self.current_world_snapshot is not None:
            self._reset_episode_from_snapshot(self.current_world_snapshot, keep_logs=True)
        else:
            self.reset()
        self.last_action_source = "Agent"
        self.last_event = "Agent auto restart episode"
        self._log(self.last_event)
        self.agent_action_log.appendleft("AUTO: restart episode")

    def handle_key(self, key: int) -> None:
        if key == pygame.K_r:
            self.reset()
            return
        if key == pygame.K_t:
            self.train_agent(episodes=200)
            return
        if key == pygame.K_g:
            self.toggle_agent_autoplay()
            return
        if key == pygame.K_ESCAPE:
            raise SystemExit
        if self.done:
            return

        action = None
        if key in (pygame.K_UP, pygame.K_w):
            action = "UP"
        elif key in (pygame.K_DOWN, pygame.K_s):
            action = "DOWN"
        elif key in (pygame.K_LEFT, pygame.K_a):
            action = "LEFT"
        elif key in (pygame.K_RIGHT, pygame.K_d):
            action = "RIGHT"
        elif key == pygame.K_SPACE:
            action = "STAY"

        if action:
            self.step(action, source="Human")

    def run(self) -> None:
        running = True
        while running:
            self.tick_count += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    try:
                        self.handle_key(event.key)
                    except SystemExit:
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_mouse_click(event.pos)

            if self.agent_autoplay:
                if self.done:
                    if self.agent_thinking_cooldown <= 0:
                        self.agent_thinking_cooldown = self.agent_auto_restart_delay_frames
                    else:
                        self.agent_thinking_cooldown -= 1
                        if self.agent_thinking_cooldown == 0:
                            self._autoplay_restart_episode()
                else:
                    if self.agent_thinking_cooldown <= 0:
                        self.agent_step_once(learn=False, explore=False)
                        self.agent_thinking_cooldown = self.agent_action_delay_frames
                    else:
                        self.agent_thinking_cooldown -= 1
            elif self.agent_thinking_cooldown > 0:
                self.agent_thinking_cooldown -= 1
            self.draw()
            self.clock.tick(self.cfg.fps)
        pygame.quit()


if __name__ == "__main__":
    MazeKeyGame().run()
