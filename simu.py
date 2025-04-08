import pygame
import random
import numpy as np
from collections import deque
import heapq
import time

# Initialisation Pygame
pygame.init()


# Obtention de la taille de l'écran
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
clock = pygame.time.Clock()
font = pygame.font.Font(None, 20)

# Temps d'attente total pour chaque groupe
total_wait_fixed = 0
total_wait_adaptive = 0
count_fixed = 0
count_adaptive = 0

# Compteur des visiteurs qui ont quitté le parc
exited_count = 0

# Constante pour limiter le temps passé dans le parc (en ticks)
MAX_TIME_IN_PARK = 36000  # à ajuster selon le comportement souhaité


# Définition des positions en pourcentage de la taille de l'écran
def get_coords(x_percent, y_percent):
    return (int(WIDTH * x_percent), int(HEIGHT * y_percent))


# Redéfinir les coordonnées des attractions en fonction de la taille de l'écran
attractions = {
    "Space Mountain": (*get_coords(0.875, 0.41), 95, 1),
    "Big Thunder Mountain": (*get_coords(0.78, 0.795), 45, 2),
    "Pirates des Caraïbes": (*get_coords(0.56, 0.92), 20, 3),
    "Peter Pan's Flight": (*get_coords(0.34, 0.795), 40, 3),
    "roller coaster": (*get_coords(0.25, 0.41), 20, 4),
    "bluefire": (*get_coords(0.34, 0.205), 20, 2),
    "silverstar": (*get_coords(0.56, 0.08), 20, 1),
    "euromir": (*get_coords(0.78, 0.205), 20, 3),
    "eurosat": (*get_coords(0.875, 0.59), 40, 3),
    "toutatis": (*get_coords(0.25, 0.59), 20, 4),
}

# Définir la position de la porte de sortie au centre
exit_gate = get_coords(0.56, 0.5)

# Création des files d'attente et des visiteurs à l'intérieur des attractions
queues = {name: deque() for name in attractions.keys()}
in_attraction = {name: deque() for name in attractions.keys()}
cycle_timer = {name: 0 for name in attractions.keys()}

# Création des visiteurs
visitors = []

# Paramètres d'apparition progressive des visiteurs
visitor_spawn_timer = 0
total_time_elapsed = 0
spawn_curve_factor = 100  # Facteur pour ralentir l'apparition progressivement
spawn_interval = 15  # Temps minimum entre chaque apparition de visiteurs (modifiable)
visitor_count = 0  # Nombre actuel de visiteurs dans le parc

# Chargement et redimensionnement des images d'attractions
attraction_images = {
    "Space Mountain": pygame.image.load("logo.png").convert_alpha(),
    "Big Thunder Mountain": pygame.image.load("logo.png").convert_alpha(),
    "Pirates des Caraïbes": pygame.image.load("logo.png").convert_alpha(),
    "Peter Pan's Flight": pygame.image.load("logo.png").convert_alpha(),
    "roller coaster": pygame.image.load("logo.png").convert_alpha(),
    "bluefire": pygame.image.load("logo.png").convert_alpha(),
    "silverstar": pygame.image.load("logo.png").convert_alpha(),
    "euromir": pygame.image.load("logo.png").convert_alpha(),
    "eurosat": pygame.image.load("logo.png").convert_alpha(),
    "toutatis": pygame.image.load("logo.png").convert_alpha(),
}

# Redimensionner les images des attractions selon la taille de l'écran
image_size = int(min(WIDTH, HEIGHT) * 0.05)  # 5% de la plus petite dimension
for name in attraction_images:
    attraction_images[name] = pygame.transform.smoothscale(attraction_images[name], (image_size, image_size))

wait_time_history = {name: deque(maxlen=55) for name in attractions.keys()}


def build_graph(last_attraction_for_adaptive=None):
    """Construit un graphe où chaque attraction est connectée entre elles et à l'entrée/sortie."""
    graph = {name: {} for name in attractions}

    # Connecter chaque attraction à toutes les autres
    for a1 in attractions:
        for a2 in attractions:
            if a1 != a2:
                distance = np.linalg.norm(np.array(attractions[a1][:2]) - np.array(attractions[a2][:2]))
                current_queue = len(queues[a2])
                _, _, capacity, duration = attractions[a2]
                # Pénalité si a2 est la dernière attraction visitée par un adaptatif
                penalty = 1000 if last_attraction_for_adaptive and a2 == last_attraction_for_adaptive else 0
                cost = (distance / 10) + (current_queue / max(1, capacity)) * duration + penalty
                graph[a1][a2] = cost
                graph[a2][a1] = cost  # Connexion bidirectionnelle

    # Ajouter exit_gate pour représenter l'entrée du parc
    graph["exit_gate"] = {}
    for attraction in attractions:
        distance = np.linalg.norm(np.array(exit_gate) - np.array(attractions[attraction][:2]))
        gate_cost = distance / 10
        graph["exit_gate"][attraction] = gate_cost
        graph[attraction]["exit_gate"] = gate_cost

    # Ajouter "Exit" comme sortie finale
    graph["Exit"] = {}
    for attraction in attractions:
        distance = np.linalg.norm(np.array(attractions[attraction][:2]) - np.array(exit_gate))
        exit_cost = distance / 10
        graph["Exit"][attraction] = exit_cost
        graph[attraction]["Exit"] = exit_cost

    return graph


def dijkstra(graph, start, targets):
    """Trouve le chemin optimal en visitant toutes les attractions désirées"""
    queue = []
    heapq.heappush(queue, (0, start, []))  # (coût total, attraction actuelle, chemin parcouru)
    visited = {}
    best_path = None

    while queue:
        cost, current, path = heapq.heappop(queue)
        if current in visited and visited[current] <= cost:
            continue
        visited[current] = cost
        path = path + [current]
        if set(targets).issubset(set(path)):
            return path
        for neighbor, weight in graph.get(current, {}).items():
            if neighbor not in path or neighbor in targets:
                heapq.heappush(queue, (cost + weight, neighbor, path))
        if best_path is None or len(set(path).intersection(set(targets))) > len(
                set(best_path).intersection(set(targets))):
            best_path = path
    # Si aucun chemin complet n'est trouvé, retourner le meilleur chemin partiel
    return best_path if best_path else ["Exit"]


def update_visitor_next_destination(visitor):
    """Met à jour la destination du visiteur adaptatif en recalculant avec Dijkstra uniquement si nécessaire"""
    if visitor["fixed_path"] or visitor["commit_to_destination"]:
        return

    # Si le visiteur a déjà effectué toutes ses attractions, il part
    if not visitor["desires"]:
        visitor["destination"] = "Exit"
        visitor["going_to_exit"] = True
        return

    # Retirer l'attraction que l'on vient de quitter (si présente)
    if visitor["last_attraction"] in visitor["desires"]:
        visitor["desires"].remove(visitor["last_attraction"])

    # Éliminer les doublons
    visitor["desires"] = list(dict.fromkeys(visitor["desires"]))

    """Met à jour la destination du visiteur adaptatif en recalculant avec Dijkstra"""

    if visitor["fixed_path"]:
        return  # Les visiteurs fixes ne changent jamais leur chemin

    if not visitor["desires"]:
        visitor["destination"] = "Exit"
        visitor["going_to_exit"] = True
        print(f"🏁 Visiteur {id(visitor)} a terminé son parcours et quitte le parc.")
        return

    visitor["desires"] = list(dict.fromkeys(visitor["desires"]))  # Supprime les doublons
    graph = build_graph()

    valid_destinations = [d for d in visitor["desires"] if d != visitor["last_attraction"]]

    if not valid_destinations:
        visitor["destination"] = "Exit"
        visitor["going_to_exit"] = True
        return

    graph = build_graph(visitor["last_attraction"])

    # 🕒 Mesurer le temps que prend Dijkstra
    start_time = time.time()

    optimal_path = dijkstra(graph, visitor["last_attraction"], valid_destinations)
    elapsed_time = time.time() - start_time
    print(f"🧠 Dijkstra pour visiteur {id(visitor)} a pris {elapsed_time:.4f} secondes")

    if optimal_path and len(optimal_path) > 1:
        visitor["planned_route"] = optimal_path[1:]
        visitor["destination"] = visitor["planned_route"].pop(0)
        visitor["commit_to_destination"] = True  # Bloquer les recalculs jusqu'à ce que la destination soit atteinte
        # Debug (peut être activé)
        # print(f"🔄 {id(visitor)} - Replanification vers {visitor['destination']} (désirs restants : {visitor['desires']})")
    else:
        visitor["destination"] = "Exit"
        visitor["going_to_exit"] = True


def generate_new_visitor():
    """Crée un visiteur avec un itinéraire basé sur Dijkstra"""
    attraction_weights = {
        "Space Mountain": 10,
        "Big Thunder Mountain": 9,
        "Pirates des Caraïbes": 4,
        "Peter Pan's Flight": 8,
        "roller coaster": 3,
        "bluefire": 7,
        "silverstar": 8,
        "euromir": 4,
        "eurosat": 7,
        "toutatis": 8
    }
    attractions_list = list(attraction_weights.keys())
    weights = list(attraction_weights.values())
    num_visits = random.randint(3, len(attractions_list) + 2)
    desires = random.choices(attractions_list, weights=weights, k=num_visits)
    fixed_path = random.random() < 0.8

    visitor = {
        "position": np.array(exit_gate, dtype=np.float64),
        "desires": desires[:],
        "original_desires": desires[:],
        "destination": desires[0],
        "speed": random.uniform(1, 2),
        "finished": False,
        "going_to_exit": False,
        "in_queue": False,
        "inside_timer": 0,
        "cooldown_timer": 0,
        "last_attraction": "exit_gate",  # Commence depuis l'entrée
        "exiting_attraction": False,
        "destination_changes": [],
        "fixed_path": fixed_path,
        "planned_route": [],
        "commit_to_destination": True,  # Par défaut, on suit la destination initiale
        "start_time": total_time_elapsed,  # Pour suivre le temps passé dans le parc
        "stuck_timer": 0,  # Compteur de stagnation
        "prev_position": np.array(exit_gate, dtype=np.float64),
        "prev_destination": None,
        "counted_finished": False  # Pour éviter de compter plusieurs fois
    }

    if not fixed_path:
        graph = build_graph()
        optimal_path = dijkstra(graph, "exit_gate", desires)
        if optimal_path and len(optimal_path) > 1:
            visitor["planned_route"] = optimal_path[1:]
            visitor["destination"] = visitor["planned_route"].pop(0)
        visitor["commit_to_destination"] = True
    return visitor


# Redimensionner le panneau d'information
info_panel_width = int(WIDTH * 0.25)
info_panel_height = int(HEIGHT * 0.25)

# Simulation principale
running = True
while running:
    screen.fill((150, 200, 0))

    # Dessiner un carré noir pour l'Exit Gate en redimensionnant selon l'écran
    pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0, 0, info_panel_width, info_panel_height))
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(0, 0, info_panel_width, info_panel_height), 3)

    graph = build_graph()

    positions = {name: (x, y) for name, (x, y, _, _) in attractions.items()}
    positions["exit_gate"] = exit_gate
    positions["Exit"] = exit_gate  # Tu peux ajuster si la sortie est ailleurs

    drawn = set()
    for node in graph:
        for neighbor in graph[node]:
            key = tuple(sorted((node, neighbor)))
            if key not in drawn:
                x1, y1 = positions[node]
                x2, y2 = positions[neighbor]
                pygame.draw.line(screen, (220, 220, 220), (x1, y1), (x2, y2), 1)
                drawn.add(key)

    # Apparition progressive des visiteurs depuis l'entrée
    total_time_elapsed += 1
    visitor_spawn_timer -= 1
    spawn_probability = max(0.1, min(1.0, 1 - np.exp(-total_time_elapsed / spawn_curve_factor)))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:  # Ajout de la possibilité de quitter avec Échap
                running = False
            elif event.key == pygame.K_UP:
                spawn_curve_factor = max(10, spawn_curve_factor - 10)
            elif event.key == pygame.K_DOWN:
                spawn_curve_factor = min(500, spawn_curve_factor + 10)
            elif event.key == pygame.K_LEFT:
                spawn_interval = max(1, spawn_interval - 1)
            elif event.key == pygame.K_RIGHT:
                spawn_interval = min(100, spawn_interval + 1)

    if visitor_spawn_timer <= 0:
        num_visitors = random.randint(2, 5)
        for _ in range(num_visitors):
            if random.random() < spawn_probability:
                visitors.append(generate_new_visitor())
        visitor_spawn_timer = spawn_interval

    # Forcer la sortie des visiteurs qui passent trop de temps dans le parc
    for visitor in visitors:
        if total_time_elapsed - visitor["start_time"] > MAX_TIME_IN_PARK:
            visitor["destination"] = "Exit"
            visitor["going_to_exit"] = True
            print(f"⏰ {id(visitor)} - Trop longtemps dans le parc, départ forcé.")

    # Dessiner les attractions et afficher la file d'attente
    for name, (x, y, capacity, duration) in attractions.items():
        if name in attraction_images:
            image_height = attraction_images[name].get_height()
            screen.blit(attraction_images[name], (x - image_size // 2, y - image_height - 5))

        # Calculer la longueur de la file d'attente
        queue_length = len(queues[name])

        # Définir la couleur du texte en fonction de la longueur de la file
        if queue_length > 15:
            queue_color = (255, 0, 0)  # Rouge si la file est longue
        elif queue_length > 7:
            queue_color = (255, 165, 0)  # Orange si la file est moyenne
        else:
            queue_color = (255, 255, 255)  # Blanc si la file est courte

        # Afficher le texte de la file d'attente avec la couleur appropriée
        queue_text = font.render(f"🕒 {cycle_timer[name]}s | {queue_length} | {len(in_attraction[name])}", True,
                                 queue_color)
        text_width = queue_text.get_width()
        screen.blit(queue_text, (x - text_width // 2, y + 10))

        # Afficher le nom de l'attraction
        attraction_text = font.render(name, True, (255, 255, 255))
        screen.blit(attraction_text, (x - attraction_text.get_width() // 2, y + 30))

    # Dessiner la sortie
    exit_radius = int(min(WIDTH, HEIGHT) * 0.025)
    pygame.draw.circle(screen, (255, 255, 255), exit_gate, exit_radius)

    # Déplacement et affichage des visiteurs
    for visitor in visitors:
        # Mise à jour du compteur de stagnation
        if np.linalg.norm(visitor["position"] - visitor["prev_position"]) < 1 and visitor["prev_destination"] == \
                visitor["destination"]:
            visitor["stuck_timer"] += 1
        else:
            visitor["stuck_timer"] = 0
        visitor["prev_position"] = visitor["position"].copy()
        visitor["prev_destination"] = visitor["destination"]

        # Choix de la couleur selon l'état
        if visitor["finished"]:
            color = (0, 255, 0)
        elif visitor["going_to_exit"]:
            color = (255, 255, 0)
        elif visitor["destination"] in in_attraction and any(
                v is visitor for v in in_attraction[visitor["destination"]]):
            color = (75, 0, 130)
        elif visitor["destination"] in queues and any(v is visitor for v in queues[visitor["destination"]]):
            color = (255, 165, 0)
        else:
            color = (0, 255, 255) if visitor["fixed_path"] else (255, 140, 0)

        dest_name = visitor["destination"]
        dest_pos = np.array(attractions.get(dest_name, exit_gate)[:2], dtype=np.float64)
        direction = dest_pos - visitor["position"]
        distance = np.linalg.norm(direction)

        if visitor["exiting_attraction"]:
            visitor["position"] += visitor["exit_direction"] * visitor["speed"]
            visitor["cooldown_timer"] -= 1
            if visitor["cooldown_timer"] <= 0:
                visitor["exiting_attraction"] = False
        elif distance > 5:
            visitor["position"] += (direction / distance) * visitor["speed"]
        else:
            if dest_name in attractions and visitor["cooldown_timer"] == 0:
                queue = queues[dest_name]
                _, _, capacity, duration = attractions[dest_name]
                # Pour les adaptatifs, si la file est trop longue, autoriser un recalcul
                if (not visitor["fixed_path"]) and (len(queue) > capacity * 1.5) and (not visitor["in_queue"]):
                    visitor["commit_to_destination"] = False
                    update_visitor_next_destination(visitor)
                    continue
                if not visitor["in_queue"]:
                    visitor["queue_entry_time"] = total_time_elapsed
                    queue.append(visitor)
                    visitor["in_queue"] = True
                    visitor["commit_to_destination"] = False
            elif dest_name == "Exit":
                visitor["finished"] = True
                if not visitor["counted_finished"]:
                    visitor["counted_finished"] = True
                    exited_count += 1
                    print(f"🏁 Visiteur {id(visitor)} - Arrivé à Exit, parc terminé. Total quittés: {exited_count}")

        # Adapter la taille des points des visiteurs à la résolution
        visitor_size = max(3, int(min(WIDTH, HEIGHT) * 0.005))
        pygame.draw.circle(screen, color, visitor["position"].astype(int), visitor_size)

    # Supprimer les visiteurs ayant fini leur parcours
    visitors = [visitor for visitor in visitors if not visitor["finished"]]
    visitor_count = len(visitors)

    # Gestion des cycles des attractions
    for attraction in queues:
        _, _, capacity, duration = attractions[attraction]
        if cycle_timer[attraction] == 0 and len(queues[attraction]) > 0:
            for _ in range(min(capacity, len(queues[attraction]))):
                visitor = queues[attraction].popleft()

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

    for attraction in in_attraction:
        to_remove = []
        for visitor in list(in_attraction[attraction]):
            if visitor["inside_timer"] > 0:
                visitor["inside_timer"] -= 1
            if visitor["inside_timer"] == 0:
                to_remove.append(visitor)
        for visitor in to_remove:
            in_attraction[attraction].remove(visitor)
            visitor["in_queue"] = False
            visitor["cooldown_timer"] = 120
            # Pour les adaptatifs, retirer l'attraction visitée de leurs désirs
            if not visitor["fixed_path"]:
                if attraction in visitor["desires"]:
                    visitor["desires"].remove(attraction)
            visitor["last_attraction"] = attraction
            visitor["exiting_attraction"] = True
            wait_time = total_time_elapsed - visitor["queue_entry_time"]
            wait_time_history[attraction].append(wait_time)
            if visitor["fixed_path"]:
                total_wait_fixed += wait_time
                count_fixed += 1
            else:
                total_wait_adaptive += wait_time
                count_adaptive += 1

            # Choix de la prochaine destination
            if visitor["fixed_path"]:
                if visitor["desires"]:
                    visitor["destination"] = visitor["desires"].pop(0)
                else:
                    visitor["destination"] = "Exit"
                    visitor["going_to_exit"] = True
            else:
                if visitor["desires"]:
                    update_visitor_next_destination(visitor)
                    if visitor["destination"] == attraction and visitor["desires"]:
                        update_visitor_next_destination(visitor)
                else:
                    visitor["destination"] = "Exit"
                    visitor["going_to_exit"] = True

            visitor["exiting_attraction"] = False
            visitor["cooldown_timer"] = 0
            visitor["in_queue"] = False
            new_dest_pos = np.array(attractions.get(visitor["destination"], exit_gate)[:2], dtype=np.float64)
            direction_to_next = new_dest_pos - visitor["position"]
            distance_to_next = np.linalg.norm(direction_to_next)
            if distance_to_next > 0:
                visitor["exit_direction"] = direction_to_next / distance_to_next
                visitor["position"] += visitor["exit_direction"] * visitor["speed"]
            else:
                visitor["exit_direction"] = np.array([0, 0])

    # Calcul des temps d'attente moyens
    average_wait_fixed = 10 * (total_wait_fixed / count_fixed) / 60 if count_fixed > 0 else 0
    average_wait_adaptive = 10 * (total_wait_adaptive / count_adaptive) / 60 if count_adaptive > 0 else 0

    # Affichage des statistiques
    margin = 10
    y_pos = margin

    # Afficher les paramètres avec un espacement relatif à la taille de l'écran
    line_height = int(HEIGHT * 0.03)

    spawn_text = font.render(f"Spawn Curve Factor: {spawn_curve_factor}", True, (255, 255, 255))
    screen.blit(spawn_text, (margin, y_pos))
    y_pos += line_height

    interval_text = font.render(f"Spawn Interval: {spawn_interval}", True, (255, 255, 255))
    screen.blit(interval_text, (margin, y_pos))
    y_pos += line_height

    prob_text = font.render(f"Spawn Probability: {spawn_probability:.2f}", True, (255, 255, 255))
    screen.blit(prob_text, (margin, y_pos))
    y_pos += line_height

    visitor_count_text = font.render(f"Visitors in Park: {visitor_count}", True, (255, 255, 255))
    screen.blit(visitor_count_text, (margin, y_pos))
    y_pos += line_height

    exited_text = font.render(f"Exited Visitors: {exited_count}", True, (255, 0, 0))
    screen.blit(exited_text, (margin, y_pos))
    y_pos += line_height

    fixed_wait_text = font.render(f"Fixed Path Avg Wait: {average_wait_fixed:.2f} min", True, (0, 255, 255))
    screen.blit(fixed_wait_text, (margin, y_pos))
    y_pos += line_height

    adaptive_wait_text = font.render(f"Adaptive Path Avg Wait: {average_wait_adaptive:.2f} min", True, (255, 140, 0))
    screen.blit(adaptive_wait_text, (margin, y_pos))

    # Mettre à jour les minuteurs d'attraction
    for attraction in cycle_timer:
        if cycle_timer[attraction] > 0:
            cycle_timer[attraction] -= 1

    pygame.display.flip()
    clock.tick(60)

pygame.quit()