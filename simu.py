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


# Création des attractions (simulation de wagons)
attractions = {
    "Space Mountain": (200, 200, 95, 1),
    "Big Thunder Mountain": (600, 200, 45, 2),
    "Pirates des Caraïbes": (200, 500, 20, 3),
    "Peter Pan’s Flight": (600, 500, 40, 3),
}
exit_gate = (400, 50)

# Création des files d’attente et des visiteurs à l'intérieur des attractions
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
attraction_images = {
    "Space Mountain": pygame.image.load("logo.png").convert_alpha(),
    "Big Thunder Mountain": pygame.image.load("logo.png").convert_alpha(),
    "Pirates des Caraïbes": pygame.image.load("logo.png").convert_alpha(),
    "Peter Pan’s Flight": pygame.image.load("logo.png").convert_alpha(),
}
for name in attraction_images:
    attraction_images[name] = pygame.transform.smoothscale(attraction_images[name], (30, 30))
wait_time_history = {name: deque(maxlen=35) for name in attractions.keys()}


def build_graph():
    """Construit un graphe où chaque attraction est connectée entre elles et à l'entrée/sortie."""
    graph = {name: {} for name in attractions}  # ✅ Initialise toutes les attractions dans le graphe

    # ✅ Connecter chaque attraction à toutes les autres
    for a1 in attractions:
        for a2 in attractions:
            if a1 != a2:
                distance = np.linalg.norm(np.array(attractions[a1][:2]) - np.array(attractions[a2][:2]))
                estimated_wait = np.mean(wait_time_history[a2]) if wait_time_history[a2] else 0
                attraction_capacity = attractions[a2][2]

                cost = (distance / 10) + (estimated_wait / max(1, attraction_capacity))
                graph[a1][a2] = cost
                graph[a2][a1] = cost  # ✅ Connexion bidirectionnelle

    # ✅ Ajouter `exit_gate` pour représenter l'entrée du parc
    graph["exit_gate"] = {}
    for attraction in attractions:
        distance = np.linalg.norm(np.array(exit_gate) - np.array(attractions[attraction][:2]))
        gate_cost = distance / 10  # Pondération basée sur la distance
        graph["exit_gate"][attraction] = gate_cost
        graph[attraction]["exit_gate"] = gate_cost  # ✅ Connexion bidirectionnelle

    # ✅ Ajouter `Exit` comme sortie finale
    graph["Exit"] = {}
    for attraction in attractions:
        distance = np.linalg.norm(np.array(attractions[attraction][:2]) - np.array(exit_gate))
        exit_cost = distance / 10
        graph["Exit"][attraction] = exit_cost
        graph[attraction]["Exit"] = exit_cost  # ✅ Connexion bidirectionnelle

    print("📌 Vérification des connexions du graphe :")
    for node, edges in graph.items():
        print(f"🔹 {node} -> {list(edges.keys())}")

    return graph


def dijkstra(graph, start, targets):
    """Trouve le chemin optimal en visitant toutes les attractions désirées"""
    queue = []
    heapq.heappush(queue, (0, start, []))  # (coût total, attraction actuelle, chemin parcouru)
    visited = {}  # Stocke le meilleur coût pour chaque nœud

    best_path = None

    while queue:
        cost, current, path = heapq.heappop(queue)

        if current in visited and visited[current] <= cost:
            continue
        visited[current] = cost

        path = path + [current]

        if set(targets).issubset(set(path)):  # ✅ Vérifie si toutes les attractions désirées sont visitées
            return path

        for neighbor, weight in graph.get(current, {}).items():
            if neighbor not in path or neighbor in targets:
                heapq.heappush(queue, (cost + weight, neighbor, path))

        # ✅ Sauvegarde du chemin le plus avancé même s'il est incomplet
        if best_path is None or len(set(path).intersection(set(targets))) > len(set(best_path).intersection(set(targets))):
            best_path = path

    print(f"⚠️ Dijkstra : Aucun chemin complet trouvé de {start} vers {targets}, chemin partiel : {best_path}")
    return best_path if best_path else ["Exit"]  # ✅ Toujours retourner un chemin viable


def update_visitor_next_destination(visitor):
    """Met à jour la destination du visiteur adaptatif en recalculant avec Dijkstra"""

    if visitor["fixed_path"]:
        return  # Les visiteurs fixes ne changent jamais leur chemin

    if not visitor["desires"]:  # Si plus de désirs => aller à la sortie
        visitor["destination"] = "Exit"
        visitor["going_to_exit"] = True
        print(f"🏁 Visiteur {id(visitor)} a terminé son parcours et quitte le parc.")
        return

    # ✅ Supprime les attractions en double pour éviter les boucles
    visitor["desires"] = list(dict.fromkeys(visitor["desires"]))

    # ⚡ Mise à jour du graphe
    graph = build_graph()

    # ❌ Empêcher le retour immédiat à la dernière attraction
    valid_destinations = [d for d in visitor["desires"] if d != visitor["last_attraction"]]

    if not valid_destinations:
        visitor["destination"] = "Exit"
        visitor["going_to_exit"] = True
        print(f"🚨 Visiteur {id(visitor)} n'a plus d'attractions accessibles, il quitte le parc.")
        return

    # 🔍 Trouver le chemin optimal avec Dijkstra
    optimal_path = dijkstra(graph, visitor["last_attraction"], valid_destinations)

    if optimal_path and len(optimal_path) > 1:
        visitor["planned_route"] = optimal_path[1:]  # ⚠️ Ne pas inclure la position actuelle
        visitor["destination"] = visitor["planned_route"].pop(0)
        print(f"🆕 Visiteur {id(visitor)} recalcul son trajet ➝ {visitor['destination']}")
    else:
        visitor["destination"] = "Exit"
        visitor["going_to_exit"] = True
        print(f"🚨 Visiteur {id(visitor)} n'a pas trouvé de bon chemin, il quitte le parc.")


def generate_new_visitor():
    """Crée un visiteur avec un itinéraire basé sur Dijkstra"""

    attraction_weights = {
        "Space Mountain": 10,
        "Big Thunder Mountain": 9,
        "Pirates des Caraïbes": 4,
        "Peter Pan’s Flight": 7,
    }

    attractions_list = list(attraction_weights.keys())
    weights = list(attraction_weights.values())

    num_visits = random.randint(3, len(attractions_list) + 7)
    desires = random.choices(attractions_list, weights=weights, k=num_visits)

    fixed_path = random.random() < 0.7

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
        "last_attraction": "exit_gate",  # ✅ Commence bien depuis l'entrée
        "exiting_attraction": False,
        "destination_changes": [],
        "fixed_path": fixed_path,
        "planned_route": []
    }

    if not fixed_path:
        graph = build_graph()
        optimal_path = dijkstra(graph, "exit_gate", desires)  # ✅ Part bien de `exit_gate`
        if optimal_path:
            visitor["planned_route"] = optimal_path[1:]  # ⚠️ Ne pas inclure `exit_gate`
            visitor["destination"] = visitor["planned_route"].pop(0)

    return visitor



# Simulation principale

running = True
while running:
    screen.fill((30, 30, 30))
    screen.fill((30, 30, 30))

    # Apparition progressive des visiteurs depuis l'entrée
    total_time_elapsed += 1
    visitor_spawn_timer -= 1
    spawn_probability = max(0.1,min(1.0, 1 - np.exp(-total_time_elapsed / spawn_curve_factor)))  # Limité entre 0.1 et 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:  #  Augmente la vitesse d'apparition des visiteurs
                spawn_curve_factor = max(10, spawn_curve_factor - 10)  #  Limite à 10 min
            elif event.key == pygame.K_DOWN:  #  Diminue la vitesse d'apparition des visiteurs
                spawn_curve_factor = min(500, spawn_curve_factor + 10)  #  Max 500 pour éviter le freeze
            elif event.key == pygame.K_LEFT:  #  Réduit l'intervalle entre les spawns (plus rapide)
                spawn_interval = max(1, spawn_interval - 1)  #  Min 1 pour éviter 0
            elif event.key == pygame.K_RIGHT:  #  Augmente l'intervalle entre les spawns (plus lent)
                spawn_interval = min(100, spawn_interval + 1)  #  Max 100 pour éviter un arrêt total

    if visitor_spawn_timer <= 0:
        num_visitors = random.randint(2, 5)  # Entre 2 et 5 visiteurs arrivent d'un coup
        for _ in range(num_visitors):
           if random.random() < spawn_probability:  # Probabilité de spawn ajustée
                visitors.append(generate_new_visitor())
        visitor_spawn_timer = spawn_interval  # Temps réduit entre les spawns

    # Dessiner les attractions et afficher la file d’attente
    for name, (x, y, capacity, duration) in attractions.items():
        #pygame.draw.circle(screen, (255, 0, 0), (x, y), 7)
        if name in attraction_images:
            image_height = attraction_images[name].get_height()
            screen.blit(attraction_images[name], (x - 17, y - image_height - 7))
        queue_text = font.render(f"🕒 {cycle_timer[name]}s |  {len(queues[name])} |  {len(in_attraction[name])}", True, (255, 255, 255))
        screen.blit(queue_text, (x - 40, y + 10))

        attraction_text = font.render(name, True, (255, 255, 255))
        screen.blit(attraction_text, (x - attraction_text.get_width() // 2, y + 20))

    pygame.draw.circle(screen, (255, 255, 255), exit_gate, 15)

    # Déplacer et dessiner les visiteurs
    for visitor in visitors:
        if visitor["finished"]:
            color = (0, 255, 0)  # Sortie du parc
        elif visitor["going_to_exit"]:
            color = (255, 255, 0)  # Direction sortie
        elif visitor["destination"] in in_attraction and any(
                v is visitor for v in in_attraction[visitor["destination"]]):
            color = (75, 0, 130)  # Dans l'attraction
        elif visitor["destination"] in queues and any(v is visitor for v in queues[visitor["destination"]]):
            color = (255, 165, 0)  # En file d’attente
        else:
            # Différencier les visiteurs fixes et adaptatifs
            if visitor["fixed_path"]:
                color = (0, 255, 255)  # 🔵 Bleu clair pour les visiteurs à chemin fixe
            else:
                color = (255, 140, 0)  # 🟠 Orange pour les visiteurs adaptatifs

        dest_name = visitor["destination"]
        dest_pos = np.array(attractions.get(dest_name, exit_gate)[:2], dtype=np.float64)

        direction = dest_pos - visitor["position"]
        distance = np.linalg.norm(direction)

        # Phase de sortie de l'attraction : marche un peu avant d'aller ailleurs
        if visitor["exiting_attraction"]:
            visitor["position"] += visitor["exit_direction"] * visitor["speed"]
            visitor["cooldown_timer"] -= 1
            if visitor["cooldown_timer"] <= 0:
                visitor["exiting_attraction"] = False
        elif distance > 5:
            visitor["position"] += (direction / distance) * visitor["speed"]
        else:
            if visitor["destination"] in attractions and visitor["cooldown_timer"] == 0:
                queue = queues[visitor["destination"]]
                _, _, capacity, duration = attractions[visitor["destination"]]

                if not visitor["in_queue"]:
                    visitor["queue_entry_time"] = total_time_elapsed  # Stocker le moment d'entrée en file
                    queue.append(visitor)
                    visitor["in_queue"] = True


            elif visitor["destination"] == "Exit":
                visitor["finished"] = True

        pygame.draw.circle(screen, color, visitor["position"].astype(int), 3)




    # Supprimer les visiteurs qui ont fini leur parcours pour éviter le lag
    visitors = [visitor for visitor in visitors if not visitor["finished"]]
    visitor_count = len(visitors)  # Nombre de visiteurs actifs dans le parc
    # Calcul du temps moyen d'attente dans toutes les attractions
    total_people_waiting = sum(len(queues[attraction]) for attraction in queues)
    average_wait_time = (total_people_waiting / len(attractions)) * 2 if total_people_waiting > 0 else 0


    # Gestion des cycles des attractions
    for attraction in queues:
        _, _, capacity, duration = attractions[attraction]

        if cycle_timer[attraction] == 0 and len(queues[attraction]) > 0:
            for _ in range(min(capacity, len(queues[attraction]))):
                visitor = queues[attraction].popleft()
                wait_time = total_time_elapsed - visitor["queue_entry_time"]  # Temps passé en file

                # Ajouter le temps d'attente au bon groupe
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

                # Ajouter le temps d'attente au total selon le type de visiteur
                if visitor["inside_timer"] == 0:  # Il quitte l'attraction
                    wait_time = duration * 60  # Temps qu'il a passé dans l'attraction
                    #if visitor["fixed_path"]:
                        # total_wait_fixed += wait_time
                        #count_fixed += 1

                    #else:
                        # total_wait_adaptive += wait_time
                        #count_adaptive += 1

            if visitor["inside_timer"] == 0:
                to_remove.append(visitor)

        for visitor in to_remove:
            in_attraction[attraction].remove(visitor)
            visitor["in_queue"] = False
            visitor["cooldown_timer"] = 120
            visitor["last_attraction"] = attraction
            visitor["exiting_attraction"] = True

            wait_time = total_time_elapsed - visitor["queue_entry_time"]
            wait_time_history[attraction].append(wait_time)  # ✅ Ajout du temps d’attente dans l’historique

            if visitor["fixed_path"]:
                total_wait_fixed += wait_time
                count_fixed += 1
            else:
                total_wait_adaptive += wait_time
                count_adaptive += 1

            # Dès qu'un visiteur quitte une attraction, il va directement vers la prochaine attraction
            if visitor["fixed_path"]:
                if visitor["desires"]:
                    visitor["destination"] = visitor["desires"].pop(0)
                else:
                    visitor["destination"] = "Exit"
                    visitor["going_to_exit"] = True
            else:
                if visitor["desires"]:
                    update_visitor_next_destination(visitor)  # ❗ Ne change que quand il a fini une attraction
                    new_dest = visitor["destination"]
                    if new_dest == visitor["last_attraction"] and visitor["desires"]:
                        update_visitor_next_destination(visitor)
            #if len(visitor["destination_changes"]) > 3:  # Plus de 3 changements
                #print(f"⚠️ [{total_time_elapsed}s] Visiteur {id(visitor)} change trop souvent !")

            # Mise à jour immédiate de la position pour éviter une errance inutile
            visitor["exiting_attraction"] = False  # Désactive la phase de sortie aléatoire
            visitor["cooldown_timer"] = 0  # Pas besoin d'attente avant de bouger
            visitor["in_queue"] = False  # Il n'est plus en file d'attente

            # Déplacement instantané vers la nouvelle destination
            new_dest_pos = np.array(attractions.get(visitor["destination"], exit_gate)[:2], dtype=np.float64)
            direction_to_next = new_dest_pos - visitor["position"]
            distance_to_next = np.linalg.norm(direction_to_next)

            # Normalisation du vecteur directionnel
            if distance_to_next > 0:
                visitor["exit_direction"] = direction_to_next / distance_to_next
                visitor["position"] += visitor["exit_direction"] * visitor["speed"]  # Déplacement immédiat
            else:
                visitor["exit_direction"] = np.array([0, 0])
    average_wait_fixed = (total_wait_fixed / count_fixed) / 60 if count_fixed > 0 else 0  # Converti en minutes
    average_wait_adaptive = (total_wait_adaptive / count_adaptive) / 60 if count_adaptive > 0 else 0  # Converti en minutes

    # Calculer les temps d'attente moyens pour chaque population
    fixed_wait_text = font.render(f"Fixed Path Avg Wait: {average_wait_fixed:.2f} min", True, (0, 255, 255))
    adaptive_wait_text = font.render(f"Adaptive Path Avg Wait: {average_wait_adaptive:.2f} min", True, (255, 140, 0))

    screen.blit(fixed_wait_text, (10, 110))  # Position en bas des autres textes
    screen.blit(adaptive_wait_text, (10, 130))  # Juste en dessous

    for attraction in cycle_timer:
        if cycle_timer[attraction] > 0:
            cycle_timer[attraction] -= 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # Affichage de la valeur actuelle de spawn_curve_factor et spawn_interval
    # Affichage des paramètres en haut à gauche
    spawn_text = font.render(f"Spawn Curve Factor: {spawn_curve_factor}", True, (255, 255, 255))
    interval_text = font.render(f"Spawn Interval: {spawn_interval}", True, (255, 255, 255))
    prob_text = font.render(f"Spawn Probability: {spawn_probability:.2f}", True,
                            (255, 255, 255))  # Ajout de la probabilité

    screen.blit(spawn_text, (10, 10))  # Position en haut à gauche
    screen.blit(interval_text, (10, 30))  # Juste en dessous
    screen.blit(prob_text, (10, 50))  # Affichage de la probabilité de spawn
    # Affichage du nombre de visiteurs et du temps moyen d'attente
    visitor_count_text = font.render(f"Visitors in Park: {visitor_count}", True, (255, 255, 255))
    #average_wait_text = font.render(f"Avg Wait Time: {average_wait_time:.2f} min", True, (255, 255, 255))

    screen.blit(visitor_count_text, (10, 70))  # En bas des autres textes
    #screen.blit(average_wait_text, (10, 90))


    pygame.display.flip()
    clock.tick(120)

pygame.quit()



