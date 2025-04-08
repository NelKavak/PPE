import pygame
import random
import numpy as np
from collections import deque
import heapq

# Initialisation Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 20)

# Temps d'attente total pour chaque groupe
total_wait_fixed = 0
total_wait_adaptive = 0
count_fixed = 0
count_adaptive = 0

# Compteur des visiteurs qui ont quitté le parc
exited_count = 0
rejected_count = 0
# Constante pour limiter le temps passé dans le parc (en ticks)
MAX_TIME_IN_PARK = 36000  # à ajuster selon le comportement souhaité

# Création des attractions (simulation de wagons) avec restrictions de taille
# Format: (x, y, capacité, durée, taille_min, taille_max)
attractions = {
    "Space Mountain": (200, 200, 95, 1, 1.3, 2),
    "Big Thunder Mountain": (600, 200, 45, 2, 1.35, 2),
    "Pirates des Caraïbes": (200, 500, 20, 3, 1.35, 1.9),
    "Peter Pan's Flight": (600, 500, 40, 3, 1.4, 1.90),
    "A": (300, 500, 20, 3, 1.4, 1.9),
    "B": (300, 100, 40, 3, 1.45, 1.95),
    "C": (600, 300, 20, 3, 1.4, 2),
    "D": (500, 150, 40, 3, 1.35, 2.1),
    "E": (700, 200, 20, 3, 1.2, 1.85),
    "F": (400, 200, 40, 3, 1.35, 1.95),
}
exit_gate = (400, 50)

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
spawn_interval = 15       # Temps minimum entre chaque apparition de visiteurs (modifiable)
visitor_count = 0         # Nombre actuel de visiteurs dans le parc

# Chargement et redimensionnement des images d'attractions
attraction_images = {
    "Space Mountain": pygame.image.load("logo.png").convert_alpha(),
    "Big Thunder Mountain": pygame.image.load("logo.png").convert_alpha(),
    "Pirates des Caraïbes": pygame.image.load("logo.png").convert_alpha(),
    "Peter Pan's Flight": pygame.image.load("logo.png").convert_alpha(),
    "A": pygame.image.load("logo.png").convert_alpha(),
    "B": pygame.image.load("logo.png").convert_alpha(),
    "C": pygame.image.load("logo.png").convert_alpha(),
    "D": pygame.image.load("logo.png").convert_alpha(),
    "E": pygame.image.load("logo.png").convert_alpha(),
    "F": pygame.image.load("logo.png").convert_alpha(),
}
for name in attraction_images:
    attraction_images[name] = pygame.transform.smoothscale(attraction_images[name], (30, 30))
    
wait_time_history = {name: deque(maxlen=55) for name in attractions.keys()}

def is_attraction_accessible(visitor, attraction_name):
    """Vérifie si un visiteur peut accéder à une attraction en fonction de sa taille."""
    if attraction_name == "Exit" or attraction_name == "exit_gate":
        return True
        
    _, _, _, _, min_height, max_height = attractions[attraction_name]
    height = visitor["height"]
    
    if (min_height is not None and height < min_height) or (max_height is not None and height > max_height):
        return False
    return True

def filter_accessible_attractions(visitor):
    """Filtre les attractions accessibles en fonction de la taille du visiteur."""
    return [name for name in attractions.keys() if is_attraction_accessible(visitor, name)]

def build_graph(visitor=None, last_attraction_for_adaptive=None):
    """Construit un graphe où chaque attraction est connectée entre elles et à l'entrée/sortie."""
    graph = {name: {} for name in attractions}
    
    # Déterminer les attractions accessibles pour ce visiteur
    accessible_attractions = filter_accessible_attractions(visitor) if visitor else list(attractions.keys())
    
    # Connecter chaque attraction à toutes les autres accessibles
    for a1 in attractions:
        for a2 in accessible_attractions:
            if a1 != a2:
                distance = np.linalg.norm(np.array(attractions[a1][:2]) - np.array(attractions[a2][:2]))
                current_queue = len(queues[a2])
                _, _, capacity, duration, _, _ = attractions[a2]
                # Pénalité si a2 est la dernière attraction visitée par un adaptatif
                penalty = 1000 if last_attraction_for_adaptive and a2 == last_attraction_for_adaptive else 0
                cost = (distance / 10) + (current_queue / max(1, capacity)) * duration + penalty
                graph[a1][a2] = cost
                graph[a2][a1] = cost  # Connexion bidirectionnelle

    # Ajouter exit_gate pour représenter l'entrée du parc
    graph["exit_gate"] = {}
    for attraction in accessible_attractions:
        distance = np.linalg.norm(np.array(exit_gate) - np.array(attractions[attraction][:2]))
        gate_cost = distance / 10
        graph["exit_gate"][attraction] = gate_cost
        graph[attraction]["exit_gate"] = gate_cost

    # Ajouter "Exit" comme sortie finale
    graph["Exit"] = {}
    for attraction in accessible_attractions:
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
        if best_path is None or len(set(path).intersection(set(targets))) > len(set(best_path).intersection(set(targets))):
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
    
    # Filtrer les attractions selon la taille
    valid_destinations = [d for d in visitor["desires"] 
                         if d != visitor["last_attraction"] 
                         and is_attraction_accessible(visitor, d)]

    if not valid_destinations:
        visitor["destination"] = "Exit"
        visitor["going_to_exit"] = True
        return

    graph = build_graph(visitor, visitor["last_attraction"])
    optimal_path = dijkstra(graph, visitor["last_attraction"], valid_destinations)

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
    """Crée un visiteur avec un itinéraire basé sur Dijkstra et une taille aléatoire"""
    global rejected_count  # Déclarer rejected_count comme variable globale
    
    attraction_weights = {
        "Space Mountain": 10,
        "Big Thunder Mountain": 9,
        "Pirates des Caraïbes": 4,
        "Peter Pan's Flight": 7,
        "A": 5,
        "B": 8,
        "C": 3,
        "D": 2,
        "E": 6,
        "F": 9,
    }
    
    # Générer une taille aléatoire entre 1m20 et 2m
    height = round(random.uniform(1.20, 2.0), 2)
    print(f"Nouveau visiteur généré avec taille: {height}m")
    
    # Création du visiteur de base
    visitor = {
        "position": np.array(exit_gate, dtype=np.float64),
        "height": height,
        "desires": [],
        "original_desires": [],
        "destination": None,
        "speed": random.uniform(1, 2),
        "finished": False,
        "going_to_exit": False,
        "in_queue": False,
        "inside_timer": 0,
        "cooldown_timer": 0,
        "last_attraction": "exit_gate",  # Commence depuis l'entrée
        "exiting_attraction": False,
        "destination_changes": [],
        "fixed_path": random.random() < 0.8,
        "planned_route": [],
        "commit_to_destination": True,  # Par défaut, on suit la destination initiale
        "start_time": total_time_elapsed,  # Pour suivre le temps passé dans le parc
        "stuck_timer": 0,                # Compteur de stagnation
        "prev_position": np.array(exit_gate, dtype=np.float64),
        "prev_destination": None,
        "counted_finished": False,       # Pour éviter de compter plusieurs fois
        "rejections": 0,                # Compteur de refus d'accès aux attractions
    }
    
    # Filtrer les attractions selon la taille du visiteur
    accessible_attractions = filter_accessible_attractions(visitor)
    
    # Générer une liste de souhaits aléatoire parmi toutes les attractions
    all_attractions_list = list(attraction_weights.keys())
    all_weights = [attraction_weights[a] for a in all_attractions_list]
    num_visits = min(random.randint(3, len(all_attractions_list)), len(all_attractions_list))
    
    # Générer la liste de souhaits sans restrictions
    unrestricted_desires = random.choices(all_attractions_list, weights=all_weights, k=num_visits)
    # Éliminer les doublons potentiels tout en préservant l'ordre
    unrestricted_desires = list(dict.fromkeys(unrestricted_desires))
    
    # Filtrer cette liste pour ne garder que les attractions accessibles
    filtered_desires = [d for d in unrestricted_desires if d in accessible_attractions]
    
    # Affichage du type de visiteur et de sa taille
    visitor_type = "Fixe" if visitor["fixed_path"] else "Adaptatif"
    print(f"\n--- Nouveau visiteur {id(visitor)} - Type: {visitor_type}, Taille: {height}m ---")
    
    # Affichage des attractions accessibles et inaccessibles
    print(f"✅ Attractions accessibles: {', '.join(accessible_attractions)}")
    inaccessible_attractions = [a for a in all_attractions_list if a not in accessible_attractions]
    if inaccessible_attractions:
        print(f"❌ Attractions inaccessibles: {', '.join(inaccessible_attractions)}")
    
    # Afficher la liste de souhaits originale
    print(f"📋 Liste souhaitée (sans restrictions): {', '.join(unrestricted_desires)}")
    
    # Identifier les attractions rejetées pour cause d'inaccessibilité
    would_be_rejected = [d for d in unrestricted_desires if d not in accessible_attractions]
    if would_be_rejected:
        visitor["rejections"] += len(would_be_rejected)
        rejected_count += len(would_be_rejected)  # Mettre à jour le compteur global
        print(f"🚫 Attractions filtrées car inaccessibles: {', '.join(would_be_rejected)}")
    
    # Définir les désirs du visiteur comme étant uniquement les attractions accessibles de sa liste originale
    visitor["desires"] = filtered_desires
    visitor["original_desires"] = unrestricted_desires.copy()  # Stocke la liste originale complète
    
    # Afficher le plan final après filtrage
    if filtered_desires:
        print(f"🎯 Plan final (avec restrictions): {', '.join(filtered_desires)}")
    else:
        print(f"🎯 Plan final: Aucune attraction accessible dans la liste de souhaits, direction sortie")
    
    if not filtered_desires:
        # Si aucune attraction n'est accessible dans la liste de souhaits, le visiteur va à la sortie
        visitor["destination"] = "Exit"
        visitor["going_to_exit"] = True
    elif visitor["fixed_path"]:
        # Pour les visiteurs à chemin fixe
        visitor["destination"] = filtered_desires[0]
    else:
        # Pour les visiteurs adaptatifs
        graph = build_graph(visitor)
        optimal_path = dijkstra(graph, "exit_gate", filtered_desires)
        if optimal_path and len(optimal_path) > 1:
            visitor["planned_route"] = optimal_path[1:]
            visitor["destination"] = visitor["planned_route"].pop(0)
            print(f"🧭 Parcours optimisé: {' → '.join(optimal_path)}")
        else:
            visitor["destination"] = "Exit"
            visitor["going_to_exit"] = True
    
    # Vérification de débogage
    if set(accessible_attractions) == set(all_attractions_list) and set(filtered_desires) != set(unrestricted_desires):
        print("⚠️ ERREUR: Le visiteur n'a pas de restrictions de taille mais le plan final diffère de la liste de souhaits!")
        
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
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                spawn_curve_factor = max(10, spawn_curve_factor - 10)
            elif event.key == pygame.K_DOWN:
                spawn_curve_factor = min(500, spawn_curve_factor + 10)
            elif event.key == pygame.K_LEFT:
                spawn_interval = max(1, spawn_interval - 1)
            elif event.key == pygame.K_RIGHT:
                spawn_interval = min(100, spawn_interval + 1)
            elif event.key == pygame.K_SPACE:
                # Générer un visiteur
                new_visitor = generate_new_visitor()
                visitors.append(new_visitor)
                print(f"Nouveau visiteur généré avec taille: {new_visitor['height']}m")
            elif event.key == pygame.K_v:
                # Générer 5 visiteurs
                for _ in range(5):
                    new_visitor = generate_new_visitor()
                    visitors.append(new_visitor)
                    print(f"Nouveau visiteur généré avec taille: {new_visitor['height']}m")
            elif event.key == pygame.K_b:
                # Générer 50 visiteurs
                for _ in range(50):
                    new_visitor = generate_new_visitor()
                    visitors.append(new_visitor)
                    print(f"Nouveau visiteur généré avec taille: {new_visitor['height']}m")

    # GENERATION AUTOMATIQUE DES VISITEURS
    # if visitor_spawn_timer <= 0:
    #     num_visitors = random.randint(1, 3)
    #     for _ in range(num_visitors):
    #         if random.random() < spawn_probability:
    #             visitors.append(generate_new_visitor())
    #     visitor_spawn_timer = spawn_interval
    visitor_spawn_timer = spawn_interval
    
    # Forcer la sortie des visiteurs qui passent trop de temps dans le parc
    for visitor in visitors:
        if total_time_elapsed - visitor["start_time"] > MAX_TIME_IN_PARK:
            visitor["destination"] = "Exit"
            visitor["going_to_exit"] = True
            print(f"⏰ {id(visitor)} - Trop longtemps dans le parc, départ forcé.")

    # Affichage des attractions et files d'attente
    for name, (x, y, capacity, duration, min_height, max_height) in attractions.items():
        if name in attraction_images:
            image_height = attraction_images[name].get_height()
            screen.blit(attraction_images[name], (x - 17, y - image_height - 7))
        queue_text = font.render(f"🕒 {cycle_timer[name]}s | {len(queues[name])} | {len(in_attraction[name])}", True, (255, 255, 255))
        screen.blit(queue_text, (x - 40, y + 10))
        
        # Afficher les restrictions de taille
        height_restriction = ""
        if min_height is not None and max_height is not None:
            height_restriction = f"{min_height}m-{max_height}m"
        elif min_height is not None:
            height_restriction = f"Min {min_height}m"
        elif max_height is not None:
            height_restriction = f"Max {max_height}m"
        
        attraction_text = font.render(f"{name}", True, (255, 255, 255))
        screen.blit(attraction_text, (x - attraction_text.get_width() // 2, y + 20))
        height_text = font.render(f"{height_restriction}", True, (255, 255, 255))
        screen.blit(height_text, (x - height_text.get_width() // 2, y + 35))
    pygame.draw.circle(screen, (255, 255, 255), exit_gate, 15)

    # Déplacement et affichage des visiteurs
    for visitor in visitors:
        # Choix de la couleur selon l'état
        if visitor["finished"]:
            color = (0, 255, 0)
        elif visitor["going_to_exit"]:
            color = (255, 255, 0)
        elif visitor["destination"] in in_attraction and any(v is visitor for v in in_attraction[visitor["destination"]]):
            color = (75, 0, 130)
        elif visitor["destination"] in queues and any(v is visitor for v in queues[visitor["destination"]]):
            color = (255, 165, 0)
        else:
            # Différencier par couleur selon le type de visiteur (fixe ou adaptatif)
            base_color = (0, 255, 255) if visitor["fixed_path"] else (255, 140, 0)
            # Si le visiteur a une taille restrictive, modifier légèrement la couleur
            if visitor["height"] < 1.40:  # Très petit
                color = (base_color[0], base_color[1], min(255, base_color[2] + 50))
            elif visitor["height"] > 1.90:  # Très grand
                color = (min(255, base_color[0] + 50), base_color[1], base_color[2])
            else:
                color = base_color

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
                # L'utilisateur arrive à l'attraction
                queue = queues[dest_name]
                _, _, capacity, duration, _, _ = attractions[dest_name]
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
                    print(f"🏁 Visiteur {id(visitor)} - Arrivé à Exit, parc terminé. Taille: {visitor['height']}m, Refus: {visitor['rejections']}. Total quittés: {exited_count}")

        pygame.draw.circle(screen, color, visitor["position"].astype(int), 3)

    # Supprimer les visiteurs ayant fini leur parcours
    visitors = [visitor for visitor in visitors if not visitor["finished"]]
    visitor_count = len(visitors)
    
    # Gestion des cycles des attractions
    for attraction in queues:
        _, _, capacity, duration, _, _ = attractions[attraction]
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
                # Pour les visiteurs à chemin fixe, choisir la prochaine attraction accessible
                next_attraction = None
                while visitor["desires"] and next_attraction is None:
                    candidate = visitor["desires"][0]
                    visitor["desires"].pop(0)
                    if is_attraction_accessible(visitor, candidate):
                        next_attraction = candidate
                    else:
                        visitor["rejections"] += 1
                        print(f"🚫 Visiteur {id(visitor)} - Skipping {candidate} (taille: {visitor['height']}m incompatible)")
                
                if next_attraction:
                    visitor["destination"] = next_attraction
                else:
                    visitor["destination"] = "Exit"
                    visitor["going_to_exit"] = True
            else:
                # Pour les adaptatifs, mettre à jour avec la prochaine destination optimale
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

    average_wait_fixed = (total_wait_fixed / count_fixed) / 60 if count_fixed > 0 else 0
    average_wait_adaptive = (total_wait_adaptive / count_adaptive) / 60 if count_adaptive > 0 else 0
    fixed_wait_text = font.render(f"Fixed Path Avg Wait: {average_wait_fixed:.2f} min", True, (0, 255, 255))
    adaptive_wait_text = font.render(f"Adaptive Path Avg Wait: {average_wait_adaptive:.2f} min", True, (255, 140, 0))
    screen.blit(fixed_wait_text, (10, 110))
    screen.blit(adaptive_wait_text, (10, 130))

    for attraction in cycle_timer:
        if cycle_timer[attraction] > 0:
            cycle_timer[attraction] -= 1

    spawn_text = font.render(f"Spawn Curve Factor: {spawn_curve_factor}", True, (255, 255, 255))
    interval_text = font.render(f"Spawn Interval: {spawn_interval}", True, (255, 255, 255))
    prob_text = font.render(f"Spawn Probability: {spawn_probability:.2f}", True, (255, 255, 255))
    screen.blit(spawn_text, (10, 10))
    screen.blit(interval_text, (10, 30))
    screen.blit(prob_text, (10, 50))
    visitor_count_text = font.render(f"Visitors in Park: {visitor_count}", True, (255, 255, 255))
    screen.blit(visitor_count_text, (10, 70))
    exited_text = font.render(f"Exited Visitors: {exited_count}", True, (255, 0, 0))
    screen.blit(exited_text, (10, 90))
    pygame.display.flip()
    clock.tick(240)

pygame.quit()