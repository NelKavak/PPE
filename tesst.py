import pygame
import random
import numpy as np
from collections import deque
import heapq
import time
from datetime import datetime, timedelta

# Initialisation Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulation de parc d'attractions")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 20)

# Variables pour calculer les temps d'attente
total_wait_fixed = 0
total_wait_adaptive = 0
count_fixed = 0
count_adaptive = 0

# Création des attractions avec des données plus réalistes
# (x, y, capacité par cycle, durée en minutes, popularité)
attractions = {
    "Space Mountain": (200, 150, 24, 3.5, 10),
    "Big Thunder Mountain": (550, 200, 30, 4, 9),
    "Pirates des Caraïbes": (150, 350, 50, 8, 8),
    "Peter Pan's Flight": (550, 350, 15, 2.5, 9),
    "Haunted Mansion": (350, 150, 35, 5, 7),
    "Jungle Cruise": (300, 450, 30, 10, 6),
    "Small World": (450, 450, 20, 12, 5),
    "Buzz Lightyear": (350, 250, 25, 4, 7),
}

# Points d'entrée/sortie du parc
entrance_gate = (400, 50)
exit_gate = (450, 50)

# Création des files d'attente et des visiteurs à l'intérieur des attractions
queues = {name: deque() for name in attractions.keys()}
in_attraction = {name: deque() for name in attractions.keys()}
cycle_timer = {name: 0 for name in attractions.keys()}

# Historique des temps d'attente pour chaque attraction
wait_time_history = {name: deque(maxlen=20) for name in attractions.keys()}

# Visiteurs actuellement dans le parc
visitors = []

# Paramètres de simulation
SIMULATION_SPEED = 1.0  # Facteur de vitesse (1.0 = temps réel)
VISITOR_LIMIT = 1000  # Limite de visiteurs simultanés dans le parc

# Paramètres d'apparition des visiteurs
visitor_spawn_timer = 0
current_time = datetime(2025, 4, 7, 8, 0)  # Commence à 8h

# Courbe d'affluence pendant la journée (heures -> facteur d'affluence)
attendance_curve = {
    0: 0.4,   # Minuit
    1: 0.3,  # Très faible affluence
    2: 0.2,
    3: 0.2,
    4: 0.1,
    5: 0.2,   # Début de matinée
    6: 0.3,
    7: 0.4,
    8: 0.8,
    9: 0.9,   # Matinée
    10: 1.1,
    11: 1.6,    # Fin de matinée
    12: 1.7,  # Midi
    13: 1.6,  # Début d'après-midi
    14: 1.6,  # Après-midi (pic)
    15: 1.5,
    16: 1.4,  # Milieu d'après-midi
    17: 1.2,    # Fin d'après-midi
    18: 1, # Début de soirée
    19: 1,  # Soirée
    20: 0.8,
    21: 0.7,  # Fin de soirée
    22: 0.6,
    23: 0.5,  # Tard dans la nuit
}

# Chargement des images des attractions (utilise un placeholder ici)
attraction_images = {}
for name in attractions.keys():
    # Dans un code complet, vous utiliseriez des images différentes pour chaque attraction
    # Par exemple: pygame.image.load(f"{name.lower().replace(' ', '_')}.png").convert_alpha()
    attraction_images[name] = pygame.image.load("logo.png").convert_alpha()
    attraction_images[name] = pygame.transform.smoothscale(attraction_images[name], (30, 30))


def calculate_current_attendance_factor():
    """Calcule le facteur d'affluence en fonction de l'heure actuelle"""
    hour = current_time.hour
    minute = current_time.minute

    # Interpolation entre les heures
    if hour in attendance_curve and (hour + 1) % 24 in attendance_curve:
        factor1 = attendance_curve[hour]
        factor2 = attendance_curve[(hour + 1) % 24]  # Utiliser modulo 24 pour boucler sur 24h
        # Interpolation linéaire
        factor = factor1 + (factor2 - factor1) * (minute / 60)
        return factor
    elif hour in attendance_curve:
        return attendance_curve[hour]
    else:
        return 0.1  # Valeur par défaut


def build_graph():
    """Construit un graphe où chaque attraction est connectée aux autres avec des poids réalistes"""
    graph = {name: {} for name in attractions}

    # Ajouter entrée et sortie
    graph["entrance"] = {}
    graph["exit"] = {}

    # Connecter chaque attraction à toutes les autres
    for a1 in attractions:
        pos1 = np.array(attractions[a1][:2])

        # Distance et temps d'attente pour chaque attraction
        for a2 in attractions:
            if a1 != a2:
                pos2 = np.array(attractions[a2][:2])

                # Distance physique entre les attractions
                distance = np.linalg.norm(pos1 - pos2)

                # Temps de marche estimé (1 unité = ~10 secondes)
                walk_time = distance / 10

                # Estimation du temps d'attente basée sur l'historique
                wait_time = np.mean(wait_time_history[a2]) if wait_time_history[a2] else 30

                # Coût total = temps de marche + temps d'attente prévu
                cost = walk_time + (wait_time / 60)  # Convertir en minutes

                graph[a1][a2] = cost

        # Connecter à l'entrée et à la sortie
        entrance_dist = np.linalg.norm(pos1 - np.array(entrance_gate))
        exit_dist = np.linalg.norm(pos1 - np.array(exit_gate))

        graph["entrance"][a1] = entrance_dist / 10
        graph[a1]["entrance"] = entrance_dist / 10
        graph[a1]["exit"] = exit_dist / 10
        graph["exit"][a1] = exit_dist / 10

    # Connecter entrée et sortie directement
    gate_distance = np.linalg.norm(np.array(entrance_gate) - np.array(exit_gate))
    graph["entrance"]["exit"] = gate_distance / 10
    graph["exit"]["entrance"] = gate_distance / 10

    return graph


def dijkstra(graph, start, targets):
    """
    Trouve le chemin optimal pour visiter une liste d'attractions cibles.

    Args:
        graph: Graphe pondéré des attractions
        start: Point de départ
        targets: Liste des attractions à visiter

    Returns:
        Liste représentant le chemin optimal
    """
    # Si aucune cible, aller directement à la sortie
    if not targets:
        return [start, "exit"]

    # Conversion en ensemble pour accélérer les recherches
    targets_set = set(targets)

    # File de priorité pour Dijkstra
    queue = [(0, start, [])]  # (coût, nœud actuel, chemin)
    visited = {}  # {(nœud, attractions_visitées): coût}

    # Suivi du meilleur chemin partiel
    best_path = None
    best_coverage = -1

    while queue:
        cost, current, path = heapq.heappop(queue)

        # Créer une clé d'état qui inclut les attractions visitées
        current_visited = frozenset(p for p in path if p in targets_set)
        state = (current, current_visited)

        # Ignorer si on a déjà trouvé un meilleur chemin pour cet état
        if state in visited and visited[state] <= cost:
            continue

        # Enregistrer ce chemin comme le meilleur pour cet état
        visited[state] = cost

        # Mettre à jour le chemin
        new_path = path + [current]

        # Vérifier si on a atteint toutes les cibles
        current_coverage = len(set(new_path).intersection(targets_set))

        # Mettre à jour le meilleur chemin partiel si nécessaire
        if current_coverage > best_coverage:
            best_coverage = current_coverage
            best_path = new_path

        if current_coverage == len(targets_set):
            # Si toutes les attractions sont visitées, ajouter la sortie
            if "exit" not in new_path:
                return new_path + ["exit"]
            return new_path

        # Explorer les voisins
        for neighbor, edge_cost in graph.get(current, {}).items():
            # Éviter les boucles sauf pour les attractions cibles non visitées
            should_visit = (
                    neighbor not in new_path or
                    (neighbor in targets_set and neighbor not in current_visited)
            )

            if should_visit:
                heapq.heappush(queue, (cost + edge_cost, neighbor, new_path))

    # Retourner le meilleur chemin partiel si on n'a pas pu atteindre toutes les cibles
    if best_path:
        if "exit" not in best_path:
            return best_path + ["exit"]
        return best_path

    # Chemin par défaut si rien n'a été trouvé
    return [start, "exit"]


def generate_realistic_visitor():
    """Crée un visiteur avec des préférences et comportements plus réalistes"""

    # Déterminer le nombre d'attractions que le visiteur souhaite faire
    # La plupart des visiteurs ne font que 4-7 attractions par jour
    num_attractions = random.choices(
        [2, 3, 4, 5, 6, 7, 8],
        weights=[5, 10, 20, 30, 20, 10, 5],  # Pondération centrée sur 4-6 attractions
        k=1
    )[0]

    # Sélectionner les attractions en fonction de leur popularité
    popularity = {name: attractions[name][4] for name in attractions}
    attractions_list = list(popularity.keys())
    weights = list(popularity.values())

    # Sélection pondérée des attractions basée sur la popularité
    desires = random.choices(attractions_list, weights=weights, k=num_attractions)
    # Éliminer les doublons tout en préservant l'ordre
    unique_desires = []
    for d in desires:
        if d not in unique_desires:
            unique_desires.append(d)

    # Déterminer si le visiteur utilise un itinéraire fixe ou adaptatif
    # 70% des visiteurs ont tendance à suivre un plan fixe
    fixed_path = random.random() < 0.5

    # Caractéristiques du visiteur
    visitor = {
        "position": np.array(entrance_gate, dtype=np.float64),
        "desires": unique_desires[:],
        "original_desires": unique_desires[:],
        "destination": None,
        "speed": random.uniform(0.8, 1.5),  # Vitesse variable entre visiteurs
        "finished": False,
        "going_to_exit": False,
        "in_queue": False,
        "queue_entry_time": 0,
        "inside_timer": 0,
        "cooldown_timer": 0,
        "last_attraction": "entrance",
        "exiting_attraction": False,
        "fixed_path": fixed_path,
        "planned_route": [],
        "fatigue": 0,  # Niveau de fatigue (0-100)
        "patience": random.uniform(0.7, 1.3),  # Multiplicateur de patience
        "arrival_time": current_time,  # Heure d'arrivée dans le parc
        "group_size": random.choices([1, 2, 3, 4, 5], weights=[20, 40, 25, 10, 5], k=1)[0]  # Taille du groupe
    }

    # Planifier l'itinéraire initial
    if visitor["desires"]:
        graph = build_graph()
        if fixed_path:
            # Pour les visiteurs à itinéraire fixe, planifier tout le parcours dès le début
            optimal_path = dijkstra(graph, "entrance", visitor["desires"])
            visitor["planned_route"] = optimal_path[1:]  # Exclure l'entrée
        else:
            # Pour les visiteurs adaptatifs, juste la prochaine destination
            if visitor["desires"]:
                visitor["destination"] = visitor["desires"][0]
            else:
                visitor["destination"] = "exit"
                visitor["going_to_exit"] = True
    else:
        # Si pas d'attractions désirées, aller directement à la sortie
        visitor["destination"] = "exit"
        visitor["going_to_exit"] = True

    # Définir la première destination
    if visitor["planned_route"]:
        visitor["destination"] = visitor["planned_route"].pop(0)

    return visitor


def update_visitor_next_destination(visitor):
    """Met à jour la destination du visiteur adaptatif en recalculant avec Dijkstra"""

    # Si c'est un visiteur à parcours fixe, ne pas changer son plan
    if visitor["fixed_path"]:
        if visitor["desires"]:
            visitor["destination"] = visitor["desires"].pop(0)
        else:
            visitor["destination"] = "exit"
            visitor["going_to_exit"] = True
        return

    # Si le visiteur est trop fatigué, il peut décider de quitter le parc
    fatigue_exit_chance = min(visitor["fatigue"] / 200, 0.3)  # Max 30% de chance
    if random.random() < fatigue_exit_chance:
        visitor["destination"] = "exit"
        visitor["going_to_exit"] = True
        visitor["desires"] = []  # Annuler les attractions restantes
        return

    # Si plus de désirs, aller à la sortie
    if not visitor["desires"]:
        visitor["destination"] = "exit"
        visitor["going_to_exit"] = True
        return

    # Mise à jour du graphe avec les temps d'attente actuels
    graph = build_graph()

    # Recalculer l'itinéraire optimal basé sur les conditions actuelles
    # Éviter de retourner à la dernière attraction
    valid_destinations = [d for d in visitor["desires"] if d != visitor["last_attraction"]]

    if not valid_destinations:
        visitor["destination"] = "exit"
        visitor["going_to_exit"] = True
        return

    # Trouver le chemin optimal
    optimal_path = dijkstra(graph, visitor["last_attraction"], valid_destinations)

    if optimal_path and len(optimal_path) > 1:
        next_stop = optimal_path[1]  # Premier arrêt après la position actuelle

        # Vérifier si cette attraction fait partie des désirs
        if next_stop in visitor["desires"]:
            visitor["desires"].remove(next_stop)

        visitor["destination"] = next_stop
    else:
        visitor["destination"] = "exit"
        visitor["going_to_exit"] = True


def draw_park():
    """Dessine le parc et ses éléments"""
    # Fond gris foncé
    screen.fill((30, 30, 30))

    # Dessiner l'entrée et la sortie
    pygame.draw.circle(screen, (0, 200, 0), entrance_gate, 15)  # Entrée en vert
    entry_text = font.render("Entrée", True, (255, 255, 255))
    screen.blit(entry_text, (entrance_gate[0] - 20, entrance_gate[1] - 30))

    pygame.draw.circle(screen, (200, 0, 0), exit_gate, 15)  # Sortie en rouge
    exit_text = font.render("Sortie", True, (255, 255, 255))
    screen.blit(exit_text, (exit_gate[0] - 20, exit_gate[1] - 30))

    # Dessiner les attractions
    for name, (x, y, capacity, duration, popularity) in attractions.items():
        # Dessiner l'image de l'attraction
        if name in attraction_images:
            screen.blit(attraction_images[name], (x - 15, y - 15))

        # Afficher le nom de l'attraction
        attraction_text = font.render(name, True, (255, 255, 255))
        screen.blit(attraction_text, (x - attraction_text.get_width() // 2, y + 20))

        # Afficher les infos de file d'attente
        queue_length = len(queues[name])
        wait_time_minutes = (queue_length / max(1, capacity)) * (duration * 60) / 60

        # Couleur basée sur le temps d'attente
        if wait_time_minutes < 15:
            color = (0, 255, 0)  # Vert
        elif wait_time_minutes < 30:
            color = (255, 255, 0)  # Jaune
        else:
            color = (255, 0, 0)  # Rouge

        queue_text = font.render(f"{queue_length} pers. • {int(wait_time_minutes)} min", True, color)
        screen.blit(queue_text, (x - queue_text.get_width() // 2, y + 40))


def draw_visitors():
    """Dessine les visiteurs sur la carte"""
    for visitor in visitors:
        # Déterminer la couleur en fonction du type et de l'état
        if visitor["finished"]:
            color = (100, 100, 100)  # Gris pour les visiteurs qui ont terminé
        elif visitor["going_to_exit"]:
            color = (255, 200, 0)  # Jaune pour ceux qui vont vers la sortie
        elif visitor["in_queue"] or any(v is visitor for attraction in in_attraction.values() for v in attraction):
            color = (255, 100, 0)  # Orange pour ceux en file d'attente
        elif visitor["fixed_path"]:
            color = (0, 150, 255)  # Bleu pour les visiteurs à chemin fixe
        else:
            color = (255, 0, 100)  # Rose pour les visiteurs adaptatifs

        # Dessiner le visiteur
        pygame.draw.circle(screen, color, visitor["position"].astype(int), 2 + visitor["group_size"] // 2)


def draw_hud():
    """Affiche les informations sur la simulation"""
    # Heure actuelle
    time_text = font.render(f"Heure: {current_time.strftime('%H:%M')}", True, (255, 255, 255))
    screen.blit(time_text, (10, 10))

    # Informations sur les visiteurs
    visitor_count = len(visitors)
    visitor_text = font.render(f"Visiteurs dans le parc: {visitor_count}", True, (255, 255, 255))
    screen.blit(visitor_text, (10, 30))

    # Temps d'attente moyens
    fixed_wait = (total_wait_fixed / max(1, count_fixed)) / 60 if count_fixed > 0 else 0
    adaptive_wait = (total_wait_adaptive / max(1, count_adaptive)) / 60 if count_adaptive > 0 else 0

    wait_fixed_text = font.render(f"Temps d'attente moyen (fixe): {fixed_wait:.1f} min", True, (0, 150, 255))
    wait_adaptive_text = font.render(f"Temps d'attente moyen (adaptatif): {adaptive_wait:.1f} min", True, (255, 0, 100))

    screen.blit(wait_fixed_text, (10, 50))
    screen.blit(wait_adaptive_text, (10, 70))

    # État du parc et affluence
    status_text = font.render(f"PARC OUVERT 24h/24 - Affluence: {calculate_current_attendance_factor():.2f}", True, (0, 255, 0))
    screen.blit(status_text, (WIDTH - status_text.get_width() - 10, 10))

    # Instructions
    instructions = [
        "Contrôles:",
        "ESPACE - Pause/Reprendre",
        "F - Accélérer la simulation",
        "S - Ralentir la simulation",
        "R - Réinitialiser la simulation"
    ]

    for i, instruction in enumerate(instructions):
        inst_text = font.render(instruction, True, (200, 200, 200))
        screen.blit(inst_text, (WIDTH - inst_text.get_width() - 10, 40 + i * 20))


def manage_attraction_cycles():
    """Gère les cycles des attractions (embarquement, durée, débarquement)"""
    global total_wait_fixed, total_wait_adaptive, count_fixed, count_adaptive

    for attraction_name in attractions:
        x, y, capacity, duration, popularity = attractions[attraction_name]
        duration_ticks = int(duration * 60)  # Conversion en ticks

        # Si le cycle est terminé et qu'il y a des visiteurs en attente
        if cycle_timer[attraction_name] <= 0 and queues[attraction_name]:
            # Nombre de visiteurs à embarquer (min entre capacité et taille de la file)
            boarding_count = min(capacity, len(queues[attraction_name]))

            # Embarquer les visiteurs
            for _ in range(boarding_count):
                if queues[attraction_name]:
                    visitor = queues[attraction_name].popleft()

                    # Calculer le temps d'attente
                    wait_time = (current_time - visitor["queue_entry_time"]).total_seconds()

                    # Enregistrer le temps d'attente selon le type de visiteur
                    if visitor["fixed_path"]:
                        total_wait_fixed += wait_time
                        count_fixed += 1
                    else:
                        total_wait_adaptive += wait_time
                        count_adaptive += 1

                    # Ajouter à l'historique des temps d'attente
                    wait_time_history[attraction_name].append(wait_time)

                    # Définir le temps dans l'attraction
                    visitor["inside_timer"] = duration_ticks
                    in_attraction[attraction_name].append(visitor)

            # Démarrer un nouveau cycle
            cycle_timer[attraction_name] = duration_ticks

        # Réduire le temps du cycle
        elif cycle_timer[attraction_name] > 0:
            cycle_timer[attraction_name] -= 1

        # Gérer les visiteurs dans l'attraction
        visitors_leaving = []
        for visitor in in_attraction[attraction_name]:
            visitor["inside_timer"] -= 1

            # Si le temps est écoulé, le visiteur quitte l'attraction
            if visitor["inside_timer"] <= 0:
                visitors_leaving.append(visitor)

        # Traiter les visiteurs qui quittent l'attraction
        for visitor in visitors_leaving:
            in_attraction[attraction_name].remove(visitor)
            visitor["in_queue"] = False
            visitor["exiting_attraction"] = True
            visitor["last_attraction"] = attraction_name

            # Augmenter la fatigue après chaque attraction
            visitor["fatigue"] += random.uniform(5, 15)

            # Temps de cooldown avant la prochaine attraction
            visitor["cooldown_timer"] = random.randint(60, 180)  # 1-3 minutes

            # Direction de sortie aléatoire
            angle = random.uniform(0, 2 * np.pi)
            visitor["exit_direction"] = np.array([np.cos(angle), np.sin(angle)])

            # Mettre à jour la destination
            if attraction_name in visitor["desires"]:
                visitor["desires"].remove(attraction_name)

            # Planifier la prochaine destination
            update_visitor_next_destination(visitor)


def spawn_visitors():
    """Fait apparaître de nouveaux visiteurs en fonction de l'heure et de l'affluence"""
    global visitor_spawn_timer

    # Réduire le timer
    visitor_spawn_timer -= 1

    if visitor_spawn_timer <= 0:
        # Calculer le facteur d'affluence actuel
        attendance_factor = calculate_current_attendance_factor()

        # Nombre de visiteurs à faire apparaître
        spawn_count = random.choices(
            [1, 2, 3, 4, 5, 6, 7 ,8],
            weights=[2.5, 5, 20, 30, 20, 20 , 7.5, 5],
            k=1
        )[0]

        # Ajuster en fonction de l'affluence
        spawn_count = int(spawn_count * attendance_factor)

        # Limiter le nombre total de visiteurs
        spawn_count = min(spawn_count, VISITOR_LIMIT - len(visitors))

        # Faire apparaître les visiteurs
        for _ in range(spawn_count):
            visitors.append(generate_realistic_visitor())

        # Réinitialiser le timer (varie selon l'affluence)
        visitor_spawn_timer = random.randint(10, 17) // max(0.1, attendance_factor)


def update_time():
    """Met à jour l'heure de la simulation"""
    global current_time

    # Avancer le temps (1 seconde réelle = 1 minute dans la simulation)
    current_time += timedelta(minutes=1 * SIMULATION_SPEED)


def main():
    """Fonction principale de la simulation"""
    global SIMULATION_SPEED, visitors

    paused = False
    running = True

    while running:
        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_f:
                    SIMULATION_SPEED = min(5.0, SIMULATION_SPEED + 0.5)
                elif event.key == pygame.K_s:
                    SIMULATION_SPEED = max(0.5, SIMULATION_SPEED - 0.5)
                elif event.key == pygame.K_r:
                    # Réinitialiser la simulation
                    visitors = []
                    for name in queues:
                        queues[name].clear()
                        in_attraction[name].clear()

        if not paused:
            # Mettre à jour le temps
            update_time()

            # Faire apparaître des visiteurs
            spawn_visitors()

            # Gérer les attractions
            manage_attraction_cycles()

            # Mettre à jour les visiteurs
            for visitor in visitors[:]:
                # Si le visiteur est en file d'attente ou dans une attraction, passer au suivant
                if visitor["in_queue"] or any(v is visitor for attraction in in_attraction.values() for v in attraction):
                    continue

                # Si le visiteur est en train de sortir d'une attraction
                if visitor["exiting_attraction"]:
                    visitor["cooldown_timer"] -= 1
                    visitor["position"] += 0

                    if visitor["cooldown_timer"] <= 0:
                        visitor["exiting_attraction"] = False
                    continue

                # Déterminer la position de la destination
                if visitor["destination"] == "entrance":
                    dest_pos = np.array(entrance_gate)
                elif visitor["destination"] == "exit":
                    dest_pos = np.array(exit_gate)
                elif visitor["destination"] in attractions:
                    dest_pos = np.array(attractions[visitor["destination"]][:2])
                else:
                    # Destination invalide, diriger vers la sortie
                    visitor["destination"] = "exit"
                    dest_pos = np.array(exit_gate)

                # Calculer la direction et la distance
                direction = dest_pos - visitor["position"]
                distance = np.linalg.norm(direction)

                # Déplacer le visiteur vers sa destination
                if distance > 5:
                    visitor["position"] += (direction / distance) * visitor["speed"]
                else:
                    # Le visiteur est arrivé à destination
                    if visitor["destination"] == "exit":
                        visitor["finished"] = True
                    elif visitor["destination"] in attractions:
                        # Rejoindre la file d'attente
                        visitor["in_queue"] = True
                        visitor["queue_entry_time"] = current_time
                        queues[visitor["destination"]].append(visitor)

                    # Si c'est un visiteur à chemin fixe, passer à la prochaine étape
                    if visitor["fixed_path"] and visitor["planned_route"]:
                        visitor["destination"] = visitor["planned_route"].pop(0)

            # Retirer les visiteurs qui ont terminé
            visitors = [v for v in visitors if not v["finished"]]

        # Affichage
        draw_park()
        draw_visitors()
        draw_hud()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()