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
    
    # Obtener información básica del estado
    score = state.get_score()
    drone_pos = state.get_drone_position()
    layout = state.get_layout()
    hunter_positions = state.get_hunter_positions()
    pending_deliveries = state.get_pending_deliveries()
    
    # Inicializar componentes de evaluación
    hunter_penalty = 0.0
    delivery_bonus = 0.0
    
    # Calcular penalización por hunters cercanos
    if hunter_positions:
        min_hunter_distance = float('inf')
        for hunter_pos in hunter_positions:
            # Distancia BFS
            distance = utils.bfs_distance(layout, drone_pos, hunter_pos, hunter_restricted=True)
            if distance < min_hunter_distance:
                min_hunter_distance = distance
        
        # Penalizar más si los hunters están muy cerca
        if min_hunter_distance != float('inf'):
            hunter_penalty = -100.0 / (min_hunter_distance)
    
    # Calcular bonificación por entregas cercanas
    if pending_deliveries:
        min_delivery_distance = float('inf')
        for delivery_pos in pending_deliveries:
            distance = utils.bfs_distance(layout, drone_pos, delivery_pos, hunter_restricted=False)
            if distance < min_delivery_distance:
                min_delivery_distance = distance
        
        # Bonificar más si las entregas están cerca
        if min_delivery_distance != float('inf'):
            delivery_bonus = 50.0 / (min_delivery_distance)
  
    final_score = score + hunter_penalty + delivery_bonus
    
    
    if final_score > 1000.0:
        final_score = 1000.0
    elif final_score < -1000.0:
        final_score = -1000.0
    
    return final_score
