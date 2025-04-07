import pygame
import random
import numpy as np
from collections import deque

# Initialisation Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 20)

# Variables de suivi des temps d'attente
total_wait_fixed = 0
total_wait_adaptive = 0
count_fixed = 0
count_adaptive = 0

# Création des attractions (nom, position x, y, capacité, durée)
attractions = {
    "Space Mountain": (200, 200, 55, 1),
    "Big Thunder Mountain": (600, 150, 45, 2),
    "Pirates des Caraïbes": (300, 400, 20, 3),
    "Peter Pan’s Flight": (500, 500, 40, 3),
    "Splash Mountain": (700, 300, 50, 3),
    "Indiana Jones Adventure": (350, 150, 50, 3),
    "PPE journey": (650, 500, 7, 2),
}
exit_gate = (400, 250)

# Création des files d’attente et suivi des cycles
queues = {name: deque() for name in attractions.keys()}
in_attraction = {name: deque() for name in attractions.keys()}
cycle_timer = {name: 0 for name in attractions.keys()}

# Liste des visiteurs
visitors = []
visitor_id_counter = 0  # Compteur pour assigner un ID unique

# Variables de gestion des spawns
visitor_spawn_timer = 0
total_time_elapsed = 0
spawn_curve_factor = 100
spawn_interval = 15
visitor_count = 0

# Fonction pour choisir la prochaine attraction avec le moins d'attente
def update_visitor_next_destination(visitor):
    if visitor["fixed_path"]:
        return
    if visitor["desires"]:
        best_next = min(visitor["desires"], key=lambda dest: len(queues[dest]) * attractions[dest][3])
        visitor["desires"].remove(best_next)
        visitor["destination"] = best_next

# Fonction de génération des visiteurs
def generate_new_visitor():
    global visitor_id_counter

    attraction_weights = {
        "Space Mountain": 10,
        "Big Thunder Mountain": 9,
        "Pirates des Caraïbes": 4,
        "Peter Pan’s Flight": 7,
        "Splash Mountain": 5,
        "Indiana Jones Adventure": 3,
        "PPE journey": 1,
    }

    attractions_list = list(attraction_weights.keys())
    weights = list(attraction_weights.values())

    num_visits = random.randint(3, len(attractions_list) + 7)
    desires = random.choices(attractions_list, weights=weights, k=num_visits)

    visitor = {
        "id": visitor_id_counter,
        "position": np.array(exit_gate, dtype=np.float64),
        "desires": desires[:],
        "destination": desires[0],
        "speed": random.uniform(1, 2),
        "finished": False,
        "going_to_exit": False,
        "in_queue": False,
        "inside_timer": 0,
        "cooldown_timer": 0,
        "last_attraction": None,
        "exiting_attraction": False,
        "fixed_path": random.random() < 0.5,
        "queue_entry_time": None  # Ajout du suivi du temps d'entrée en file
    }
    visitor_id_counter += 1
    return visitor

# Simulation principale
running = True
while running:
    screen.fill((30, 30, 30))

    total_time_elapsed += 1
    visitor_spawn_timer -= 1
    spawn_probability = max(0.1, min(1.0, 1 - np.exp(-total_time_elapsed / spawn_curve_factor)))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Apparition des visiteurs
    if visitor_spawn_timer <= 0:
        num_visitors = random.randint(2, 5)
        for _ in range(num_visitors):
            if random.random() < spawn_probability:
                visitors.append(generate_new_visitor())
        visitor_spawn_timer = spawn_interval

    # Gestion des attractions et files d'attente
    for name, (x, y, capacity, duration) in attractions.items():
        queue_text = font.render(f"{len(queues[name])} | {len(in_attraction[name])}", True, (255, 255, 255))
        screen.blit(queue_text, (x - 40, y + 10))

    pygame.draw.circle(screen, (255, 255, 255), exit_gate, 15)

    # Gestion des visiteurs
    for visitor in visitors:
        dest_pos = np.array(attractions.get(visitor["destination"], exit_gate)[:2], dtype=np.float64)
        direction = dest_pos - visitor["position"]
        distance = np.linalg.norm(direction)

        if distance > 5:
            visitor["position"] += (direction / distance) * visitor["speed"]
        else:
            if visitor["destination"] in attractions:
                queue = queues[visitor["destination"]]
                if not visitor["in_queue"]:
                    visitor["queue_entry_time"] = total_time_elapsed
                    queue.append(visitor)
                    visitor["in_queue"] = True

    # Calcul des temps d'attente
    for attraction in queues:
        _, _, capacity, duration = attractions[attraction]
        if cycle_timer[attraction] == 0 and len(queues[attraction]) > 0:
            for _ in range(min(capacity, len(queues[attraction]))):
                visitor = queues[attraction].popleft()
                if visitor["queue_entry_time"] is not None:
                    wait_time = total_time_elapsed - visitor["queue_entry_time"]
                    if visitor["fixed_path"]:
                        total_wait_fixed += wait_time
                        count_fixed += 1
                    else:
                        total_wait_adaptive += wait_time
                        count_adaptive += 1

                visitor["inside_timer"] = duration * 60
                in_attraction[attraction].append(visitor)
            cycle_timer[attraction] = duration * 60

    # Calcul des temps d'attente moyens
    average_wait_fixed = (total_wait_fixed / count_fixed) / 60 if count_fixed > 0 else 0
    average_wait_adaptive = (total_wait_adaptive / count_adaptive) / 60 if count_adaptive > 0 else 0

    # Affichage des stats
    fixed_wait_text = font.render(f"Fixed Path Avg Wait: {average_wait_fixed:.2f} min", True, (0, 255, 255))
    adaptive_wait_text = font.render(f"Adaptive Path Avg Wait: {average_wait_adaptive:.2f} min", True, (255, 140, 0))

    screen.blit(fixed_wait_text, (10, 110))
    screen.blit(adaptive_wait_text, (10, 130))

    pygame.display.flip()
    clock.tick(120)

pygame.quit()
