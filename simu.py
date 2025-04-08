import pygame
import random
import numpy as np
from collections import deque
import heapq
import time

pygame.init()

info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
clock = pygame.time.Clock()
font = pygame.font.Font(None, 20)
large_font = pygame.font.Font(None, 36)

total_wait_fixed = 0
total_wait_adaptive = 0
count_fixed = 0
count_adaptive = 0
exited_count = 0
MAX_TIME_IN_PARK = 36000


def get_coords(x_percent, y_percent):
    return (int(WIDTH * x_percent), int(HEIGHT * y_percent))


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

exit_gate = get_coords(0.56, 0.5)

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
for name in attraction_images:
    attraction_images[name] = pygame.transform.smoothscale(attraction_images[name], (image_size, image_size))

wait_time_history = {name: deque(maxlen=55) for name in attractions.keys()}


# Interface utilisateur pour le parcours personnalisé
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
        self.panel_width = int(WIDTH * 0.35)
        self.panel_height = int(HEIGHT * 0.7)
        self.panel_x = (WIDTH - self.panel_width) // 2
        self.panel_y = (HEIGHT - self.panel_height) // 2

        self.active = False
        self.route_calculated = False
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

        # Pour visualiser le chemin calculé
        self.route_nodes = []
        self.route_edges = []

    def toggle(self):
        self.active = not self.active
        if not self.active:
            self.route_calculated = False

    def update(self, mouse_pos, mouse_clicked):
        if not self.active:
            return

        if self.close_button.update(mouse_pos, mouse_clicked):
            self.toggle()
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

            # Préparer la visualisation du parcours
            for node in optimal_path:
                if node in attractions:
                    pos = attractions[node][:2]
                elif node == "exit_gate" or node == "Exit":
                    pos = exit_gate
                self.route_nodes.append(pos)

            # Créer les segments entre chaque point consécutif
            for i in range(len(self.route_nodes) - 1):
                self.route_edges.append((self.route_nodes[i], self.route_nodes[i + 1]))
        else:
            self.optimized_route = []
            self.route_nodes = []
            self.route_edges = []

    def draw(self, surface):
        if not self.active:
            return

        # Dessiner le panneau de fond
        panel_rect = pygame.Rect(self.panel_x, self.panel_y, self.panel_width, self.panel_height)
        pygame.draw.rect(surface, (50, 50, 50, 200), panel_rect)
        pygame.draw.rect(surface, (255, 255, 255), panel_rect, 2)

        # Titre
        title_text = large_font.render("Planifiez votre parcours", True, (255, 255, 255))
        surface.blit(title_text, (self.panel_x + (self.panel_width - title_text.get_width()) // 2, self.panel_y + 20))

        # Instructions
        instruction_text = font.render("Sélectionnez les attractions que vous souhaitez visiter:", True,
                                       (255, 255, 255))
        surface.blit(instruction_text, (self.panel_x + 50, self.panel_y + 70))

        # Checkboxes
        for checkbox in self.attraction_checkboxes:
            checkbox.draw(surface)

        # Bouton de calcul
        self.calculate_button.draw(surface)

        # Bouton de fermeture
        self.close_button.draw(surface)

        # Affichage du résultat du calcul
        if self.route_calculated and self.optimized_route:
            result_y = self.panel_y + 100 + len(self.attraction_checkboxes) * 40 + 20
            result_text = font.render("Parcours optimisé :", True, (255, 255, 0))
            surface.blit(result_text, (self.panel_x + 50, result_y))

            # Filtrer "exit_gate" de l'affichage et commencer le compteur à 1
            displayable_route = [attraction for attraction in self.optimized_route if
                                 attraction != "exit_gate" and attraction != "Exit"]

            for i, attraction in enumerate(displayable_route):
                route_text = font.render(f"{i + 1}. {attraction}", True, (255, 255, 255))
                surface.blit(route_text, (self.panel_x + 70, result_y + 30 + i * 25))

        # Visualisation du parcours sur la carte si calculé
        if self.route_calculated and self.route_edges:
            for start_pos, end_pos in self.route_edges:
                pygame.draw.line(surface, (255, 255, 0), start_pos, end_pos, 3)

            for i, pos in enumerate(self.route_nodes):
                pygame.draw.circle(surface, (0, 255, 0), pos, 8)
                # N'afficher un numéro que pour les attractions (pas pour exit_gate/Exit)
                if i < len(self.optimized_route):
                    node_name = self.optimized_route[i]
                    if node_name != "exit_gate" and node_name != "Exit":
                        # Trouver l'index dans la liste filtrable pour l'affichage du numéro
                        filtered_idx = [a for a in self.optimized_route if a != "exit_gate" and a != "Exit"].index(
                            node_name)
                        num_text = font.render(str(filtered_idx + 1), True, (0, 0, 0))
                        surface.blit(num_text,
                                     (pos[0] - num_text.get_width() // 2, pos[1] - num_text.get_height() // 2))

def build_graph(last_attraction_for_adaptive=None):
    graph = {name: {} for name in attractions}

    for a1 in attractions:
        for a2 in attractions:
            if a1 != a2:
                distance = np.linalg.norm(np.array(attractions[a1][:2]) - np.array(attractions[a2][:2]))
                current_queue = len(queues[a2])
                _, _, capacity, duration = attractions[a2]
                penalty = 1000 if last_attraction_for_adaptive and a2 == last_attraction_for_adaptive else 0
                cost = (distance / 30) + (current_queue / max(1, capacity)) * duration + penalty
                graph[a1][a2] = cost
                graph[a2][a1] = cost

    graph["exit_gate"] = {}
    for attraction in attractions:
        distance = np.linalg.norm(np.array(exit_gate) - np.array(attractions[attraction][:2]))
        gate_cost = distance / 10
        graph["exit_gate"][attraction] = gate_cost
        graph[attraction]["exit_gate"] = gate_cost

    graph["Exit"] = {}
    for attraction in attractions:
        distance = np.linalg.norm(np.array(attractions[attraction][:2]) - np.array(exit_gate))
        exit_cost = distance / 10
        graph["Exit"][attraction] = exit_cost
        graph[attraction]["Exit"] = exit_cost

    return graph


def dijkstra(graph, start, targets):
    """
    Calculate the shortest path that visits all target attractions.

    Args:
        graph: The weighted graph of attractions
        start: Starting point (usually 'exit_gate')
        targets: List of attractions to visit

    Returns:
        Path that visits all targets in optimal order
    """
    if not targets:
        return ["exit_gate", "Exit"]

    # Create a complete path that visits all targets
    current = start
    path = [current]
    remaining_targets = targets.copy()

    while remaining_targets:
        # Find the closest next target
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
            # No direct connection, just add the first remaining target
            # This is a simplified approach
            current = remaining_targets[0]
            path.append(current)
            remaining_targets.remove(current)

    # Add Exit to the end of path
    path.append("Exit")
    return path

def update_visitor_next_destination(visitor):
    if visitor["fixed_path"] or visitor["commit_to_destination"]:
        return

    if not visitor["desires"]:
        visitor["destination"] = "Exit"
        visitor["going_to_exit"] = True
        return

    if visitor["last_attraction"] in visitor["desires"]:
        visitor["desires"].remove(visitor["last_attraction"])

    visitor["desires"] = list(dict.fromkeys(visitor["desires"]))

    if not visitor["desires"]:
        visitor["destination"] = "Exit"
        visitor["going_to_exit"] = True
        return

    visitor["desires"] = list(dict.fromkeys(visitor["desires"]))
    graph = build_graph()

    valid_destinations = [d for d in visitor["desires"] if d != visitor["last_attraction"]]

    if not valid_destinations:
        visitor["destination"] = "Exit"
        visitor["going_to_exit"] = True
        return

    graph = build_graph(visitor["last_attraction"])

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
        "last_attraction": "exit_gate",
        "exiting_attraction": False,
        "destination_changes": [],
        "fixed_path": fixed_path,
        "planned_route": [],
        "commit_to_destination": True,
        "start_time": total_time_elapsed,
        "stuck_timer": 0,
        "prev_position": np.array(exit_gate, dtype=np.float64),
        "prev_destination": None,
        "counted_finished": False
    }

    if not fixed_path:
        graph = build_graph()
        optimal_path = dijkstra(graph, "exit_gate", desires)
        if optimal_path and len(optimal_path) > 1:
            visitor["planned_route"] = optimal_path[1:]
            visitor["destination"] = visitor["planned_route"].pop(0)
        visitor["commit_to_destination"] = True
    return visitor


# Créer une instance de l'interface utilisateur
ui = UserInterface()

# Ajouter un bouton pour ouvrir l'interface de planification
plan_route_button = Button(
    10, HEIGHT - 50, 180, 40,
    "Planifier mon parcours",
    (50, 50, 200), (80, 80, 220)
)

info_panel_width = int(WIDTH * 0.25)
info_panel_height = int(HEIGHT * 0.25)

running = True
while running:
    screen.fill((150, 200, 0))

    pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0, 0, info_panel_width, info_panel_height))
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(0, 0, info_panel_width, info_panel_height), 3)

    graph = build_graph()

    positions = {name: (x, y) for name, (x, y, _, _) in attractions.items()}
    positions["exit_gate"] = exit_gate
    positions["Exit"] = exit_gate

    drawn = set()
    for node in graph:
        for neighbor in graph[node]:
            key = tuple(sorted((node, neighbor)))
            if key not in drawn:
                x1, y1 = positions.get(node, exit_gate)
                x2, y2 = positions.get(neighbor, exit_gate)
                pygame.draw.line(screen, (220, 220, 220), (x1, y1), (x2, y2), 1)
                drawn.add(key)

    # Gestion des événements
    mouse_pos = pygame.mouse.get_pos()
    mouse_clicked = False

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
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Clic gauche
                mouse_clicked = True

    # Apparition progressive des visiteurs
    total_time_elapsed += 1
    visitor_spawn_timer -= 1
    spawn_probability = max(0.1, min(1.0, 1 - np.exp(-total_time_elapsed / spawn_curve_factor)))

    if visitor_spawn_timer <= 0 and not ui.active:
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

    # Affichage des attractions et files d'attente
    for name, (x, y, capacity, duration) in attractions.items():
        if name in attraction_images:
            image_height = attraction_images[name].get_height()
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

        attraction_text = font.render(name, True, (255, 255, 255))
        screen.blit(attraction_text, (x - attraction_text.get_width() // 2, y + 30))

    # Dessiner la sortie
    exit_radius = int(min(WIDTH, HEIGHT) * 0.025)
    pygame.draw.circle(screen, (255, 255, 255), exit_gate, exit_radius)
    exit_text = font.render("Entrée/Sortie", True, (0, 0, 0))
    screen.blit(exit_text, (exit_gate[0] - exit_text.get_width() // 2, exit_gate[1] - exit_text.get_height() // 2))

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
        average_wait = sum(wait_time_history[attraction]) / len(wait_time_history[attraction]) / 60 if \
        wait_time_history[attraction] else 0
        wait_text = font.render(f"{attraction}: {average_wait:.1f} min", True, (255, 255, 255))
        screen.blit(wait_text, (margin, y_pos))
        y_pos += line_height

    # Mettre à jour les cycles des attractions
    for attraction in cycle_timer:
        if cycle_timer[attraction] > 0:
            cycle_timer[attraction] -= 1

    # Mettre à jour et dessiner le bouton de planification
    if plan_route_button.update(mouse_pos, mouse_clicked):
        ui.toggle()
    plan_route_button.draw(screen)

    # Mettre à jour et dessiner l'interface utilisateur de planification
    ui.update(mouse_pos, mouse_clicked)
    ui.draw(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()