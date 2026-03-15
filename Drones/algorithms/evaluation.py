from __future__ import annotations

from typing import TYPE_CHECKING
import algorithms.utils as utils

if TYPE_CHECKING:
    from world.game_state import GameState


def evaluation_function(state: GameState) -> float:
    """
    Evaluation function for non-terminal states of the drone vs. hunters game.

    A good evaluation function can consider multiple factors, such as:
      (a) BFS distance from drone to nearest delivery point (closer is better).
          Uses actual path distance so walls and terrain are respected.
      (b) BFS distance from each hunter to the drone, traversing only normal
          terrain ('.' / ' ').  Hunters blocked by mountains, fog, or storms
          are treated as unreachable (distance = inf) and pose no threat.
      (c) BFS distance to a "safe" position (i.e., a position that is not in the path of any hunter).
      (d) Number of pending deliveries (fewer is better).
      (e) Current score (higher is better).
      (f) Delivery urgency: reward the drone for being close to a delivery it can
          reach strictly before any hunter, so it commits to nearby pickups
          rather than oscillating in place out of excessive hunter fear.
      (g) Adding a revisit penalty can help prevent the drone from getting stuck in cycles.

    Returns a value in [-1000, +1000].

    Tips:
    - Use state.get_drone_position() to get the drone's current (x, y) position.
    - Use state.get_hunter_positions() to get the list of hunter (x, y) positions.
    - Use state.get_pending_deliveries() to get the set of pending delivery (x, y) positions.
    - Use state.get_score() to get the current game score.
    - Use state.get_layout() to get the current layout.
    - Use state.is_win() and state.is_lose() to check terminal states.
    - Use bfs_distance(layout, start, goal, hunter_restricted) from algorithms.utils
      for cached BFS distances. hunter_restricted=True for hunter-only terrain.
    - Use dijkstra(layout, start, goal) from algorithms.utils for cached
      terrain-weighted shortest paths, returning (cost, path).
    - Consider edge cases: no pending deliveries, no hunters nearby.
    - A good evaluation function balances delivery progress with hunter avoidance.
    """
    # Casos terminales
    if state.is_win():
        return 1000.0
    if state.is_lose():
        return -1000.0
    
    score = state.get_score()
    drone_pos = state.get_drone_position()
    layout = state.get_layout()
    hunter_positions = state.get_hunter_positions()
    pending_deliveries = state.get_pending_deliveries()
    
    eval_score = score
    
    # Penalización por entregas pendientes
    eval_score -= 100.0 * len(pending_deliveries)

    # Cercanía a la entrega más próxima
    min_delivery_dist = float('inf')
    for delivery_pos in pending_deliveries:
        dist = utils.bfs_distance(layout, drone_pos, delivery_pos, hunter_restricted=False)
        if dist < min_delivery_dist:
            min_delivery_dist = dist
            
    if min_delivery_dist != float('inf'):
        # se suma 1 para evitar división por cero. 
        eval_score += 60.0 / (min_delivery_dist + 1)

    # Gestión del peligro de los cazadores
    min_hunter_dist = float('inf')
    for hunter_pos in hunter_positions:
        dist = utils.bfs_distance(layout, drone_pos, hunter_pos, hunter_restricted=True)
        if dist < min_hunter_dist:
            min_hunter_dist = dist
            
    if min_hunter_dist != float('inf'):
        if min_hunter_dist <= 2:
            # Peligro inminente: penalización masiva para obligarlo a huir
            eval_score -= 500.0 / (min_hunter_dist + 1)
        elif min_hunter_dist <= 4:
            # Riesgo moderado
            eval_score -= 100.0 / (min_hunter_dist + 1)
        elif min_hunter_dist >= 6:
            # Posición segura. Recompensa por mantener distancia.
            eval_score += 20.0

    # Urgencia de entrega
    # Si el dron está más cerca de la entrega que el cazador más cercano, se da un bono por ir a la entrega
    if min_delivery_dist != float('inf') and min_hunter_dist != float('inf'):
        if min_delivery_dist < min_hunter_dist:
            eval_score += 40.0 / (min_delivery_dist + 1)
    return max(-1000.0, min(1000.0, eval_score))
