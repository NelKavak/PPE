import pygame
import random
import numpy as np
from collections import deque
import heapq
import time
import pygame

# Initialisation Pygame
pygame.init()


# Obtention de la taille de l'écran
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
clock = pygame.time.Clock()
font = pygame.font.Font(None, 20)
large_font = pygame.font.Font(None, 36)

# Temps d'attente total pour chaque groupe
total_wait_fixed = 0
total_wait_adaptive = 0
count_fixed = 0
count_adaptive = 0

# Compteur des visiteurs qui ont quitté le parc
exited_count = 0
MAX_TIME_IN_PARK = 36000


def get_coords(x_percent, y_percent):
    return (int(WIDTH * x_percent), int(HEIGHT * y_percent))

rejected_count = 0
# Constante pour limiter le temps passé dans le parc (en ticks)
# Création des attractions (simulation de wagons)
attractions = {
    "Space Mountain": (*get_coords(0.875, 0.41), 95, 1, 1.3, 2),
    "Big Thunder Mountain": (*get_coords(0.78, 0.795), 45, 2, 1.35, 2),
    "Pirates des Caraïbes": (*get_coords(0.56, 0.92), 20, 3, 1.35, 1.9),
    "Peter Pan's Flight": (*get_coords(0.34, 0.795), 40, 3, 1.4, 1.90),
    "roller coaster": (*get_coords(0.25, 0.41), 20, 4, 1.4, 1.9),
    "bluefire": (*get_coords(0.34, 0.205), 20, 2, 1.45, 1.95),
    "silverstar": (*get_coords(0.56, 0.08), 20, 1, 1.4, 2),
    "euromir": (*get_coords(0.78, 0.205), 20, 3, 1.35, 2.1),
    "eurosat": (*get_coords(0.875, 0.59), 40, 3, 1.2, 1.85),
    "toutatis": (*get_coords(0.25, 0.59), 20, 4, 1.35, 1.95),
}

exit_gate = get_coords(0.56, 0.5)

# Création des files d'attente et des visiteurs à l'intérieur des attractions
queues = {name: deque() for name in attractions.keys()}
in_attraction = {name: deque() for name in attractions.keys()}
cycle_timer = {name: 0 for name in attractions.keys()}

visitors = []
visitor_spawn_timer = 0
total_time_elapsed = 0
spawn_curve_factor = 100
spawn_interval = 15
visitor_count = 0

attraction_images = {name: pygame.image.load("logo.png").convert_alpha() for name in attractions.keys()}

image_size = int(min(WIDTH, HEIGHT) * 0.05)
# Image de maintenance
maintenance_image = pygame.image.load("maintenance.png").convert_alpha()

# Redimensionner les images des attractions selon la taille de l'écran
image_size = int(min(WIDTH, HEIGHT) * 0.05)  # 5% de la plus petite dimension
for name in attraction_images:
    attraction_images[name] = pygame.transform.smoothscale(attraction_images[name], (image_size, image_size))

# Redimensionner l'image de maintenance
maintenance_image = pygame.transform.smoothscale(maintenance_image, (image_size, image_size))

wait_time_history = {name: deque(maxlen=55) for name in attractions.keys()}

# Variable pour suivre l'état de maintenance de "toutatis"
toutatis_maintenance = False

class Button:
    def __init__(self, x, y, width, height, text, color=(100, 100, 100), hover_color=(150, 150, 150)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False
        self.is_clicked = False

    def draw(self, surface):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, (255, 255, 255), self.rect, 2)

        text_surf = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def update(self, mouse_pos, mouse_clicked):
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        if self.is_hovered and mouse_clicked:
            self.is_clicked = True
        else:
            self.is_clicked = False
        return self.is_clicked


class Checkbox:
    def __init__(self, x, y, size, text):
        self.rect = pygame.Rect(x, y, size, size)
        self.text = text
        self.is_checked = False
        self.size = size

    def draw(self, surface):
        pygame.draw.rect(surface, (255, 255, 255), self.rect, 2)
        if self.is_checked:
            inner_rect = pygame.Rect(self.rect.x + 4, self.rect.y + 4, self.rect.width - 8, self.rect.height - 8)
            pygame.draw.rect(surface, (0, 255, 0), inner_rect)

        text_surf = font.render(self.text, True, (255, 255, 255))
        surface.blit(text_surf, (self.rect.x + self.size + 10, self.rect.y + (self.size - text_surf.get_height()) // 2))

    def update(self, mouse_pos, mouse_clicked):
        if self.rect.collidepoint(mouse_pos) and mouse_clicked:
            self.is_checked = not self.is_checked
            return True
        return False

class UserInterface:
    def __init__(self):
        self.default_panel_width = int(WIDTH * 0.35)
        self.default_panel_height = int(HEIGHT * 0.7)

        self.panel_width = self.default_panel_width
        self.panel_height = self.default_panel_height
        self.panel_x = (WIDTH - self.panel_width) // 2
        self.panel_y = (HEIGHT - self.panel_height) // 2

        self.active = False
        self.route_calculated = False
        self.show_only_result = False
        self.optimized_route = []

        checkbox_size = 20
        checkbox_spacing = 40

        self.attraction_checkboxes = []
        for i, name in enumerate(attractions.keys()):
            y_pos = self.panel_y + 100 + i * checkbox_spacing
            self.attraction_checkboxes.append(Checkbox(self.panel_x + 50, y_pos, checkbox_size, name))

        button_width = 200
        button_height = 40

        self.calculate_button = Button(
            self.panel_x + (self.panel_width - button_width) // 2,
            self.panel_y + self.panel_height - 70,
            button_width, button_height, "Calculer l'itinéraire"
        )

        self.close_button = Button(
            self.panel_x + self.panel_width - 30,
            self.panel_y + 10,
            20, 20, "X", (255, 0, 0), (200, 0, 0)
        )

        self.back_button = Button(
            self.panel_x + 30,
            self.panel_y + self.panel_height - 70,
            button_width // 2, button_height, "Retour",
            (100, 100, 100), (150, 150, 150)
        )

        self.route_nodes = []
        self.route_edges = []

    def toggle(self):
        self.active = not self.active
        if not self.active:
            self.route_calculated = False
            self.show_only_result = False

    def update(self, mouse_pos, mouse_clicked):
        if not self.active:
            return

        if self.close_button.update(mouse_pos, mouse_clicked):
            self.toggle()
            return

        if self.show_only_result:
            if self.back_button.update(mouse_pos, mouse_clicked):
                self.show_only_result = False
            return

        checkbox_clicked = False
        for checkbox in self.attraction_checkboxes:
            if checkbox.update(mouse_pos, mouse_clicked):
                checkbox_clicked = True

        if self.calculate_button.update(mouse_pos, mouse_clicked) and not checkbox_clicked:
            selected_attractions = [cb.text for cb in self.attraction_checkboxes if cb.is_checked]
            if selected_attractions:
                self.calculate_optimal_route(selected_attractions)
                self.route_calculated = True
                self.show_only_result = True
            else:
                self.route_calculated = False

    def calculate_optimal_route(self, selected_attractions):
        if not selected_attractions:
            self.optimized_route = []
            self.route_nodes = []
            self.route_edges = []
            return

        graph = build_graph()
        optimal_path = dijkstra(graph, "exit_gate", selected_attractions)

        if optimal_path:
            self.optimized_route = optimal_path
            self.route_nodes = []
            self.route_edges = []

            for node in optimal_path:
                if node in attractions:
                    pos = attractions[node][:2]
                elif node == "exit_gate" or node == "Exit":
                    pos = exit_gate
                self.route_nodes.append(pos)

            for i in range(len(self.route_nodes) - 1):
                self.route_edges.append((self.route_nodes[i], self.route_nodes[i + 1]))
        else:
            self.optimized_route = []
            self.route_nodes = []
            self.route_edges = []

    def draw(self, surface):
        if not self.active:
            return

        # Redimensionner dynamiquement si on affiche le parcours optimisé
        if self.show_only_result and self.route_calculated:
            # Calculer une hauteur adaptée au nombre d'attractions
            displayable_route = [attraction for attraction in self.optimized_route if
                                 attraction not in ["exit_gate", "Exit"]]
            result_height = max(len(displayable_route) * 30 + 150, 150)  # Minimum 200px

            self.panel_width = int(WIDTH * 0.2)
            self.panel_height = result_height

            # Positionner en bas à droite
            self.panel_x = WIDTH - self.panel_width + 20  # 20px de marge
            self.panel_y = HEIGHT - self.panel_height + 20  # 20px de marge
        else:
            self.panel_width = self.default_panel_width
            self.panel_height = self.default_panel_height
            self.panel_x = (WIDTH - self.panel_width) // 2
            self.panel_y = (HEIGHT - self.panel_height) // 2

        # Mise à jour des positions des boutons APRÈS le repositionnement du panneau
        if self.show_only_result:
            # Positions pour le panneau de résultat
            self.back_button.x = self.panel_x + (self.panel_width - 200) // 2
            self.back_button.y = self.panel_y + self.panel_height - 50
        else:
            # Positions pour le panneau principal
            self.calculate_button.x = self.panel_x + (self.panel_width - 200) // 2
            self.calculate_button.y = self.panel_y + self.panel_height - 70

        # La croix reste toujours en haut à droite du panneau, peu importe son état
        self.close_button = Button(
                self.panel_x + self.panel_width - 30,
                self.panel_y + 10,
                20, 20, "X", (255, 0, 0), (200, 0, 0)
        )

        self.back_button = Button(
            self.panel_x + self.panel_width - 30,
            self.panel_y + 10,
            20, 20, "X", (255, 0, 0), (200, 0, 0)
        )

        panel_rect = pygame.Rect(self.panel_x, self.panel_y, self.panel_width, self.panel_height)
        pygame.draw.rect(surface, (50, 50, 50, 200), panel_rect)
        pygame.draw.rect(surface, (255, 255, 255), panel_rect, 2)

        if self.show_only_result and self.route_calculated:
            # Afficher uniquement le résultat du parcours optimisé
            title_text = large_font.render("Votre parcours optimisé", True, (255, 255, 0))
            surface.blit(title_text,
                         (self.panel_x + (self.panel_width - title_text.get_width()) // 2, self.panel_y + 20))

            result_y = self.panel_y + 70
            displayable_route = [attraction for attraction in self.optimized_route if
                                 attraction not in ["exit_gate", "Exit"]]

            for i, attraction in enumerate(displayable_route):
                route_text = font.render(f"{i + 1}. {attraction}", True, (255, 255, 255))
                surface.blit(route_text, (self.panel_x + 30, result_y + i * 30))

            self.back_button.draw(surface)
            self.close_button.draw(surface)

        else:
            # Afficher l'interface de sélection
            title_text = large_font.render("Planifiez votre parcours", True, (255, 255, 255))
            surface.blit(title_text,
                         (self.panel_x + (self.panel_width - title_text.get_width()) // 2, self.panel_y + 20))

            instruction_text = font.render("Sélectionnez les attractions que vous souhaitez visiter:", True,
                                           (255, 255, 255))
            surface.blit(instruction_text, (self.panel_x + 50, self.panel_y + 70))

            for i, checkbox in enumerate(self.attraction_checkboxes):
                checkbox.x = self.panel_x + 50
                checkbox.y = self.panel_y + 100 + i * 40
                checkbox.draw(surface)

            self.calculate_button.draw(surface)
            self.close_button.draw(surface)

        # Toujours afficher le parcours optimisé s'il est calculé
        if self.route_calculated and self.route_edges:
            for start_pos, end_pos in self.route_edges:
                pygame.draw.line(surface, (255, 255, 0), start_pos, end_pos, 3)

            for i, pos in enumerate(self.route_nodes):
                pygame.draw.circle(surface, (0, 255, 0), pos, 8)
                if i < len(self.optimized_route):
                    node_name = self.optimized_route[i]
                    if node_name not in ["exit_gate", "Exit"]:
                        filtered_idx = [a for a in self.optimized_route if a not in ["exit_gate", "Exit"]].index(
                            node_name)
                        num_text = font.render(str(filtered_idx + 1), True, (0, 0, 0))
                        surface.blit(num_text,
                                     (pos[0] - num_text.get_width() // 2, pos[1] - num_text.get_height() // 2))


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
    if not targets:
        return ["exit_gate", "Exit"]

    current = start
    path = [current]
    remaining_targets = targets.copy()

    while remaining_targets:
        best_next = None
        best_cost = float('inf')

        for target in remaining_targets:
            if target in graph.get(current, {}):
                cost = graph[current][target]
                if cost < best_cost:
                    best_cost = cost
                    best_next = target

        if best_next:
            current = best_next
            path.append(current)
            remaining_targets.remove(current)
        else:
            current = remaining_targets[0]
            path.append(current)
            remaining_targets.remove(current)

    path.append("Exit")
    return path

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

    if toutatis_maintenance:
        print("Toutatis est en maintenance.")
        # Retirer "toutatis" des désirs des visiteurs et rediriger ceux qui s'y rendent
        for visitor in visitors:
            if "toutatis" in visitor["desires"]:
                visitor["desires"].remove("toutatis")
            if visitor["destination"] == "toutatis":
                # Pour les visiteurs à chemin fixe, forcer le passage à la prochaine attraction
                if visitor["fixed_path"]:
                    if visitor["desires"]:
                        visitor["destination"] = visitor["desires"].pop(0)
                    else:
                        visitor["destination"] = "Exit"
                        visitor["going_to_exit"] = True
                else:
                    # Pour les adaptatifs, recalculer le trajet
                    visitor["commit_to_destination"] = False
                    update_visitor_next_destination(visitor)
    visitor["desires"] = list(dict.fromkeys(visitor["desires"]))

    if not visitor["desires"]:
        visitor["destination"] = "Exit"
        visitor["going_to_exit"] = True
        return

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
    start_time = time.time()

    optimal_path = dijkstra(graph, visitor["last_attraction"], valid_destinations)
    elapsed_time = time.time() - start_time

    if optimal_path and len(optimal_path) > 1:
        visitor["planned_route"] = optimal_path[1:]
        visitor["destination"] = visitor["planned_route"].pop(0)
        visitor["commit_to_destination"] = True
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
        "Peter Pan's Flight": 8,
        "roller coaster": 3,
        "bluefire": 7,
        "silverstar": 8,
        "euromir": 4,
        "eurosat": 7,

    }
    if not toutatis_maintenance:
        attraction_weights["toutatis"] = 8


    attractions_list = list(attraction_weights.keys())
    weights = list(attraction_weights.values())
    num_visits = random.randint(3, len(attractions_list) + 2)
    desires = random.choices(attractions_list, weights=weights, k=num_visits)
    fixed_path = random.random() < 0.8


    # Générer une taille aléatoire entre 1m20 et 2m
    height = round(random.uniform(1.20, 2.0), 2)

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
        "last_attraction": "exit_gate",
        "exiting_attraction": False,
        "destination_changes": [],
        "fixed_path": random.random() < 0.8,
        "planned_route": [],
        "commit_to_destination": True,
        "start_time": total_time_elapsed,
        "stuck_timer": 0,
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


ui = UserInterface()

plan_route_button = Button(
    10, HEIGHT - 50, 180, 40,
    "Planifier mon parcours",
    (50, 50, 200), (80, 80, 220)
)

info_panel_width = int(WIDTH * 0.2)
info_panel_height = int(HEIGHT * 0.5)

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
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_UP:
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
            elif event.key == pygame.K_v:
                # Générer 5 visiteurs
                for _ in range(5):
                    new_visitor = generate_new_visitor()
                    visitors.append(new_visitor)
            elif event.key == pygame.K_b:
                # Générer 50 visiteurs
                for _ in range(50):
                    new_visitor = generate_new_visitor()
                    visitors.append(new_visitor)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_clicked = True

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            # Vérifier si le clic est dans la zone du bouton de maintenance
            if 50 <= mouse_x <= 200 and 600 <= mouse_y <= 650:
                toutatis_maintenance = not toutatis_maintenance
                if toutatis_maintenance:
                    print("Toutatis est en maintenance.")
                    # Retirer "toutatis" des désirs des visiteurs
                    for visitor in visitors:
                        if "toutatis" in visitor["desires"]:
                            visitor["desires"].remove("toutatis")
                        if visitor["destination"] == "toutatis":
                            update_visitor_next_destination(visitor)
                else:
                    print("Toutatis est de nouveau opérationnel.")

    if visitor_spawn_timer <= 0 and not ui.active:
        num_visitors = random.randint(2, 5)
        for _ in range(num_visitors):
            if random.random() < spawn_probability:
                visitors.append(generate_new_visitor())
        visitor_spawn_timer = spawn_interval


    # GENERATION AUTOMATIQUE DES VISITEURS
    if visitor_spawn_timer <= 0:
        num_visitors = random.randint(1, 3)
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

    # Affichage des attractions et files d'attente
    for name, (x, y, capacity, duration, min_height, max_height) in attractions.items():
        if name in attraction_images:
            image_height = attraction_images[name].get_height()
            if name == "toutatis" and toutatis_maintenance:
                screen.blit(maintenance_image, (x - image_size // 2, y - image_height - 5))
            else:
                screen.blit(attraction_images[name], (x - image_size // 2, y - image_height - 5))

        queue_length = len(queues[name])

        if queue_length > 15:
            queue_color = (255, 0, 0)
        elif queue_length > 7:
            queue_color = (255, 165, 0)
        else:
            queue_color = (255, 255, 255)

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
    exit_text = font.render("Entrée/Sortie", True, (0, 0, 0))
    screen.blit(exit_text, (exit_gate[0] - exit_text.get_width() // 2, exit_gate[1] - exit_text.get_height() // 2))

    # Dessiner le bouton de maintenance en haut à droite
    pygame.draw.rect(screen, (255, 0, 0) if toutatis_maintenance else (0, 0, 0), pygame.Rect(50, 600, 150, 50))
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(50, 600, 150, 50), 3)

    maintenance_text = font.render("Maintenance", True, (255, 255, 255))
    screen.blit(maintenance_text, (70, 620))

    # Déplacement et affichage des visiteurs
    for visitor in visitors:
        if np.linalg.norm(visitor["position"] - visitor["prev_position"]) < 1 and visitor["prev_destination"] == \
                visitor["destination"]:
            visitor["stuck_timer"] += 1
        else:
            visitor["stuck_timer"] = 0
        visitor["prev_position"] = visitor["position"].copy()
        visitor["prev_destination"] = visitor["destination"]

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

        visitor_size = max(3, int(min(WIDTH, HEIGHT) * 0.005))
        pygame.draw.circle(screen, color, visitor["position"].astype(int), visitor_size)

    visitors = [visitor for visitor in visitors if not visitor["finished"]]
    visitor_count = len(visitors)

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

    average_wait_fixed = 10 * (total_wait_fixed / count_fixed) / 60 if count_fixed > 0 else 0
    average_wait_adaptive = 10 * (total_wait_adaptive / count_adaptive) / 60 if count_adaptive > 0 else 0

    margin = 10
    y_pos = margin
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
    y_pos += line_height

    for attraction in wait_time_history:
        average_wait = sum(wait_time_history[attraction]) / len(wait_time_history[attraction]) / 60 if wait_time_history[attraction] else 0
        wait_text = font.render(f"{attraction} Wait: {average_wait:.2f} min", True, (255, 255, 255))
        screen.blit(wait_text, (margin, y_pos))
        y_pos += line_height

    for attraction in cycle_timer:
        if cycle_timer[attraction] > 0:
            cycle_timer[attraction] -= 1

    if plan_route_button.update(mouse_pos, mouse_clicked):
        ui.toggle()

    plan_route_button.draw(screen)
    ui.update(mouse_pos, mouse_clicked)
    ui.draw(screen)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()