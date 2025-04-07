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

# Création des attractions (simulation de wagons)
attractions = {
    "Space Mountain": (200, 100, 55, 1),
    "Big Thunder Mountain": (600, 150, 45, 2),
    "Pirates des Caraïbes": (300, 400, 20, 3),
    "Peter Pan’s Flight": (500, 500, 40, 3),
    "Splash Mountain": (700, 300, 50, 3),
    "Indiana Jones Adventure": (350, 150, 50, 3),
    "PPE journey": (650, 500, 7, 2),
}
exit_gate = (400, 250)

# Création des files d’attente et des visiteurs
queues = {name: deque() for name in attractions.keys()}
visitors = []

# Fonction pour générer un visiteur avec une liste d'attractions aléatoire
def generate_new_visitor():
    attractions_list = list(attractions.keys())
    num_visits = random.randint(3, len(attractions_list))
    desires = random.sample(attractions_list, num_visits)
    return {
        "position": np.array(exit_gate, dtype=np.float64),
        "desires": desires,
        "destination": desires[0],
        "speed": random.uniform(1, 2),
        "finished": False,
        "going_to_exit": False,
        "last_attraction": None,
    }

# Fonction pour calculer les poids dynamiques
def calculate_dynamic_weights():
    weights = {}
    for a1 in attractions:
        weights[a1] = {}
        for a2 in attractions:
            if a1 != a2:
                pos1, pos2 = np.array(attractions[a1][:2]), np.array(attractions[a2][:2])
                travel_time = np.linalg.norm(pos2 - pos1) / 50  # Temps trajet arbitraire
                wait_time = len(queues[a2]) * attractions[a2][3]  # Temps attente estimé
                weights[a1][a2] = travel_time + wait_time
    return weights

# Fonction pour choisir la meilleure attraction suivante après chaque attraction
def update_visitor_desires(visitor, weights):
    if visitor["desires"]:
        current_location = visitor["last_attraction"] if visitor["last_attraction"] else exit_gate
        best_next = min(visitor["desires"], key=lambda dest: weights[current_location].get(dest, float('inf')))
        visitor["desires"].remove(best_next)
        visitor["destination"] = best_next

# Simulation
running = True
while running:
    screen.fill((30, 30, 30))
    dynamic_weights = calculate_dynamic_weights()

    # Ajouter de nouveaux visiteurs progressivement
    if random.random() < 0.02:
        visitors.append(generate_new_visitor())

    # Dessiner les attractions
    for name, (x, y, _, _) in attractions.items():
        pygame.draw.circle(screen, (255, 0, 0), (x, y), 7)
        screen.blit(font.render(name, True, (255, 255, 255)), (x - 30, y + 15))

    # Déplacer et gérer les visiteurs
    for visitor in visitors:
        dest_name = visitor["destination"]
        dest_pos = np.array(attractions.get(dest_name, exit_gate)[:2], dtype=np.float64)
        direction = dest_pos - visitor["position"]
        distance = np.linalg.norm(direction)

        if distance > 5:
            visitor["position"] += (direction / distance) * visitor["speed"]
        else:
            visitor["last_attraction"] = visitor["destination"]
            if visitor["desires"]:
                update_visitor_desires(visitor, dynamic_weights)
            else:
                visitor["destination"] = "Exit"
                visitor["going_to_exit"] = True

        pygame.draw.circle(screen, (0, 255, 255), visitor["position"].astype(int), 3)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
