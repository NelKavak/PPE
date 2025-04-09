import pygame
import random
import numpy as np
from collections import deque
import heapq
import time

pygame.init()

info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
screen = pygame.display.set_mode((WIDTH+300, HEIGHT+300), pygame.FULLSCREEN)
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

# Définition des attractions : (x, y, capacité, durée)
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

# Visiteurs dédiés : route_demo_visitor (magenta) et witness_visitor (bleu)
route_demo_visitor = None
witness_visitor = None

# ===============================
# INTERFACE UTILISATEUR
# ===============================
class Button:
    def __init__(self, x, y, width, height, text, color=(100,100,100), hover_color=(150,150,150)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False
        self.is_clicked = False

    def draw(self, surface):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, (255,255,255), self.rect, 2)
        text_surf = font.render(self.text, True, (255,255,255))
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
        pygame.draw.rect(surface, (255,255,255), self.rect, 2)
        if self.is_checked:
            inner_rect = pygame.Rect(self.rect.x+4, self.rect.y+4, self.rect.width-8, self.rect.height-8)
            pygame.draw.rect(surface, (0,255,0), inner_rect)
        text_surf = font.render(self.text, True, (255,255,255))
        surface.blit(text_surf, (self.rect.x+self.size+10, self.rect.y+(self.size-text_surf.get_height())//2))

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
        self.optimized_route = []  # Liste des nœuds (chaînes)
        self.route_nodes = []      # Liste des positions (x,y)
        self.route_edges = []      # Liste de segments

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

        for checkbox in self.attraction_checkboxes:
            checkbox.update(mouse_pos, mouse_clicked)

        if self.calculate_button.update(mouse_pos, mouse_clicked):
            selected = [cb.text for cb in self.attraction_checkboxes if cb.is_checked]
            if selected:
                self.calculate_optimal_route(selected)
                self.route_calculated = True
                self.active = False  # Fermer la popup
            else:
                self.route_calculated = False

    def calculate_optimal_route(self, selected_attractions):
        if not selected_attractions:
            self.optimized_route = []
            self.route_nodes = []
            self.route_edges = []
            return
        g = build_graph()
        path = dijkstra(g, "exit_gate", selected_attractions)
        if path:
            self.optimized_route = path
            self.route_nodes = []
            self.route_edges = []
            for node in path:
                if node in attractions:
                    pos = attractions[node][:2]
                elif node in ["exit_gate", "Exit"]:
                    pos = exit_gate
                self.route_nodes.append(pos)
            for i in range(len(self.route_nodes)-1):
                self.route_edges.append((self.route_nodes[i], self.route_nodes[i+1]))
        else:
            self.optimized_route = []
            self.route_nodes = []
            self.route_edges = []

    def draw(self, surface):
        if not self.active:
            return

        panel_rect = pygame.Rect(self.panel_x, self.panel_y, self.panel_width, self.panel_height)
        pygame.draw.rect(surface, (50,50,50), panel_rect)
        pygame.draw.rect(surface, (255,255,255), panel_rect, 2)

        title_text = large_font.render("Planifiez votre parcours", True, (255,255,255))
        surface.blit(title_text, (self.panel_x + (self.panel_width - title_text.get_width())//2, self.panel_y+20))
        instruction = font.render("Sélectionnez les attractions :", True, (255,255,255))
        surface.blit(instruction, (self.panel_x+50, self.panel_y+70))
        for checkbox in self.attraction_checkboxes:
            checkbox.draw(surface)
        self.calculate_button.draw(surface)
        self.close_button.draw(surface)
        if self.optimized_route:
            result_y = self.panel_y + 100 + len(self.attraction_checkboxes)*40 + 20
            result_text = font.render("Parcours optimisé :", True, (255,255,0))
            surface.blit(result_text, (self.panel_x+50, result_y))
            displayable = [a for a in self.optimized_route if a not in ["exit_gate","Exit"]]
            for i, a in enumerate(displayable):
                rt = font.render(f"{i+1}. {a}", True, (255,255,255))
                surface.blit(rt, (self.panel_x+70, result_y+30+i*25))
            # Toujours afficher les lignes et nœuds
            for start, end in self.route_edges:
                pygame.draw.line(surface, (255,255,0), start, end, 3)
            for pos in self.route_nodes:
                pygame.draw.circle(surface, (0,255,0), pos, 8)

# Instanciation de l'interface
ui = UserInterface()
checkbox_size = 20
checkbox_spacing = 40
ui.attraction_checkboxes = []
for i, name in enumerate(attractions.keys()):
    y = ui.panel_y + 100 + i*checkbox_spacing
    ui.attraction_checkboxes.append(Checkbox(ui.panel_x+50, y, checkbox_size, name))
btn_width = 200
btn_height = 40
ui.calculate_button = Button(ui.panel_x + (ui.default_panel_width - btn_width)//2,
                             ui.panel_y + ui.default_panel_height - 70,
                             btn_width, btn_height, "Calculer l'itinéraire")
ui.close_button = Button(ui.panel_x + ui.default_panel_width - 30,
                         ui.panel_y + 10,
                         20, 20, "X", (255,0,0), (200,0,0))

plan_route_button = Button(10, HEIGHT - 50, 180, 40,
                           "Planifier mon parcours", (50,50,200), (80,80,220))

# ===============================
# FONCTIONS DE PATHFINDING
# ===============================
def build_graph(last_attraction_for_adaptive=None):
    g = {name: {} for name in attractions}
    for a1 in attractions:
        for a2 in attractions:
            if a1 != a2:
                d = np.linalg.norm(np.array(attractions[a1][:2]) - np.array(attractions[a2][:2]))
                q = len(queues[a2])
                _, cap, dur = attractions[a2][1:]
                pen = 1000 if last_attraction_for_adaptive and a2 == last_attraction_for_adaptive else 0
                cost = (d/30) + (q/max(1, cap))*dur + pen
                g[a1][a2] = cost
                g[a2][a1] = cost
    g["exit_gate"] = {}
    for a in attractions:
        d = np.linalg.norm(np.array(exit_gate) - np.array(attractions[a][:2]))
        g["exit_gate"][a] = d/10
        g[a]["exit_gate"] = d/10
    g["Exit"] = {}
    for a in attractions:
        d = np.linalg.norm(np.array(attractions[a][:2]) - np.array(exit_gate))
        g["Exit"][a] = d/10
        g[a]["Exit"] = d/10
    return g

def dijkstra(graph, start, targets):
    if not targets:
        return ["exit_gate", "Exit"]
    current = start
    path = [current]
    rem = targets.copy()
    while rem:
        best_next = None
        best_cost = float('inf')
        for t in rem:
            if t in graph.get(current, {}):
                c = graph[current][t]
                if c < best_cost:
                    best_cost = c
                    best_next = t
        if best_next:
            current = best_next
            path.append(current)
            rem.remove(current)
        else:
            current = rem[0]
            path.append(current)
            rem.remove(current)
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
    g = build_graph(visitor["last_attraction"])
    opt = dijkstra(g, visitor["last_attraction"], [d for d in visitor["desires"] if d != visitor["last_attraction"]])
    if opt and len(opt) > 1:
        visitor["planned_route"] = opt[1:]
        visitor["destination"] = visitor["planned_route"].pop(0)
        visitor["commit_to_destination"] = True
    else:
        visitor["destination"] = "Exit"
        visitor["going_to_exit"] = True

def generate_new_visitor():
    weights = {"Space Mountain":10, "Big Thunder Mountain":9, "Pirates des Caraïbes":4,
               "Peter Pan's Flight":8, "roller coaster":3, "bluefire":7,
               "silverstar":8, "euromir":4, "eurosat":7, "toutatis":8}
    alist = list(weights.keys())
    w = list(weights.values())
    num = random.randint(3, len(alist)+2)
    desires = random.choices(alist, weights=w, k=num)
    fixed_path = random.random() < 0.8
    v = {
        "position": np.array(exit_gate, dtype=np.float64),
        "desires": desires[:],
        "original_desires": desires[:],
        "destination": desires[0],
        "speed": random.uniform(1,2),
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
        "counted_finished": False,
        "generated": False
    }
    if not fixed_path:
        g = build_graph()
        opt = dijkstra(g, "exit_gate", desires)
        if opt and len(opt) > 1:
            v["planned_route"] = opt[1:]
            v["destination"] = v["planned_route"].pop(0)
        v["commit_to_destination"] = True
    return v

# Pour générer un visiteur fixe (utilisé pour le témoin et le demo)
def generate_fixed_visitor(optimal_path):
    # Retirer "exit_gate" et "Exit" du début
    route = [node for node in optimal_path if node not in ["exit_gate", "Exit"]]
    v = generate_new_visitor()
    v["position"] = np.array(exit_gate, dtype=np.float64)
    if route:
        v["destination"] = route[0]
        v["planned_route"] = route[1:] if len(route) > 1 else []
    else:
        v["destination"] = "Exit"
        v["planned_route"] = []
    return v

# Global pour le visiteur témoin
witness_visitor = None

# ====================================================================
# BOUCLE PRINCIPALE
# ====================================================================
running = True
while running:
    screen.fill((150,200,0))
    
    # Panneau d'info en haut à gauche
    info_panel_width = int(WIDTH * 0.25)
    info_panel_height = int(HEIGHT * 0.25)
    pygame.draw.rect(screen, (0,0,0), pygame.Rect(0,0,info_panel_width, info_panel_height))
    pygame.draw.rect(screen, (255,255,255), pygame.Rect(0,0,info_panel_width, info_panel_height), 3)
    
    # Affichage du graphe
    g = build_graph()
    positions = {name: (x,y) for name, (x,y,_,_) in attractions.items()}
    positions["exit_gate"] = exit_gate
    positions["Exit"] = exit_gate
    drawn = set()
    for node in g:
        for neighbor in g[node]:
            key = tuple(sorted((node, neighbor)))
            if key not in drawn:
                x1,y1 = positions.get(node, exit_gate)
                x2,y2 = positions.get(neighbor, exit_gate)
                pygame.draw.line(screen, (220,220,220), (x1,y1), (x2,y2), 1)
                drawn.add(key)
    
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
            if event.button == 1:
                mouse_clicked = True
    
    total_time_elapsed += 1
    visitor_spawn_timer -= 1
    spawn_probability = max(0.1, min(1.0, 1 - np.exp(-total_time_elapsed/spawn_curve_factor)))
    if visitor_spawn_timer <= 0:
        for _ in range(random.randint(2,5)):
            if random.random() < spawn_probability:
                visitors.append(generate_new_visitor())
        visitor_spawn_timer = spawn_interval
    
    for visitor in visitors:
        if total_time_elapsed - visitor["start_time"] > MAX_TIME_IN_PARK:
            visitor["destination"] = "Exit"
            visitor["going_to_exit"] = True
    
    for name, (x, y, cap, dur) in attractions.items():
        if name in attraction_images:
            ih = attraction_images[name].get_height()
            screen.blit(attraction_images[name], (x - image_size//2, y - ih - 5))
        ql = len(queues[name])
        if ql > 15:
            qc = (255,0,0)
        elif ql > 7:
            qc = (255,165,0)
        else:
            qc = (255,255,255)
        qt = font.render(f"🕒 {cycle_timer[name]}s | {ql} | {len(in_attraction[name])}", True, qc)
        screen.blit(qt, (x - qt.get_width()//2, y+10))
        at_text = font.render(name, True, (255,255,255))
        screen.blit(at_text, (x - at_text.get_width()//2, y+30))
    
    pygame.draw.circle(screen, (255,255,255), exit_gate, int(min(WIDTH,HEIGHT)*0.025))
    ex_text = font.render("Entrée/Sortie", True, (0,0,0))
    screen.blit(ex_text, (exit_gate[0]-ex_text.get_width()//2, exit_gate[1]-ex_text.get_height()//2))
    
    # Affichage et mise à jour des visiteurs standards
    for visitor in visitors:
        if np.linalg.norm(visitor["position"] - visitor["prev_position"]) < 1 and visitor["prev_destination"] == visitor["destination"]:
            visitor["stuck_timer"] += 1
        else:
            visitor["stuck_timer"] = 0
        visitor["prev_position"] = visitor["position"].copy()
        visitor["prev_destination"] = visitor["destination"]
        dest = visitor["destination"]
        dpos = np.array(attractions.get(dest, exit_gate)[:2], dtype=np.float64)
        dir_vect = dpos - visitor["position"]
        dist = np.linalg.norm(dir_vect)
        if visitor.get("exiting_attraction", False):
            visitor["position"] += visitor["exit_direction"] * visitor["speed"]
            visitor["cooldown_timer"] -= 1
            if visitor["cooldown_timer"] <= 0:
                visitor["exiting_attraction"] = False
        elif dist > 5:
            visitor["position"] += (dir_vect / dist) * visitor["speed"]
        else:
            if dest in attractions and visitor["cooldown_timer"] == 0:
                q = queues[dest]
                _, _, cap, dur = attractions[dest]
                if (not visitor["fixed_path"]) and (len(q) > cap * 1.5) and (not visitor["in_queue"]):
                    visitor["commit_to_destination"] = False
                    update_visitor_next_destination(visitor)
                    continue
                if not visitor["in_queue"]:
                    visitor["queue_entry_time"] = total_time_elapsed
                    q.append(visitor)
                    visitor["in_queue"] = True
                    visitor["commit_to_destination"] = False
            elif dest == "Exit":
                visitor["finished"] = True
                if not visitor.get("counted_finished", False):
                    visitor["counted_finished"] = True
                    exited_count += 1
        vsize = max(3, int(min(WIDTH,HEIGHT)*0.005))
        if visitor["finished"]:
            col = (0,255,0)
        elif visitor["going_to_exit"]:
            col = (255,255,0)
        elif visitor["fixed_path"]:
            col = (0,255,255)
        else:
            col = (255,140,0)
        pygame.draw.circle(screen, col, visitor["position"].astype(int), vsize)
    
    visitors = [v for v in visitors if not v["finished"]]
    visitor_count = len(visitors)
    
    # Mise à jour et affichage du visiteur en surbrillance (route_demo_visitor)
    if route_demo_visitor is not None:
        if np.linalg.norm(route_demo_visitor["position"] - route_demo_visitor["prev_position"]) < 1 and \
           route_demo_visitor["prev_destination"] == route_demo_visitor["destination"]:
            route_demo_visitor["stuck_timer"] += 1
        else:
            route_demo_visitor["stuck_timer"] = 0
        route_demo_visitor["prev_position"] = route_demo_visitor["position"].copy()
        route_demo_visitor["prev_destination"] = route_demo_visitor["destination"]
        dname = route_demo_visitor["destination"]
        dpos = np.array(attractions.get(dname, exit_gate)[:2], dtype=np.float64)
        dir_vect = dpos - route_demo_visitor["position"]
        dist = np.linalg.norm(dir_vect)
        if dist > 15:
            route_demo_visitor["position"] += (dir_vect/ dist) * route_demo_visitor["speed"]
        else:
            if route_demo_visitor.get("planned_route") and len(route_demo_visitor["planned_route"]) > 0:
                route_demo_visitor["destination"] = route_demo_visitor["planned_route"].pop(0)
                route_demo_visitor["cooldown_timer"] = 0
            else:
                route_demo_visitor["destination"] = "Exit"
        dsize = max(5, int(min(WIDTH,HEIGHT)*0.007))
        pygame.draw.circle(screen, (255,0,255), route_demo_visitor["position"].astype(int), dsize)
    
    # Mise à jour et affichage du visiteur témoin (witness_visitor)
    if ui.optimized_route and len(ui.optimized_route) > 1:
        if witness_visitor is None:
            witness_visitor = generate_fixed_visitor(ui.optimized_route)
            witness_visitor["witness"] = True
        if np.linalg.norm(witness_visitor["position"] - witness_visitor["prev_position"]) < 1 and \
           witness_visitor["prev_destination"] == witness_visitor["destination"]:
            witness_visitor["stuck_timer"] += 1
        else:
            witness_visitor["stuck_timer"] = 0
        witness_visitor["prev_position"] = witness_visitor["position"].copy()
        witness_visitor["prev_destination"] = witness_visitor["destination"]
        dname = witness_visitor["destination"]
        dpos = np.array(attractions.get(dname, exit_gate)[:2], dtype=np.float64)
        dir_vect = dpos - witness_visitor["position"]
        dist = np.linalg.norm(dir_vect)
        if dist > 15:
            witness_visitor["position"] += (dir_vect/ dist) * witness_visitor["speed"]
        else:
            if witness_visitor.get("planned_route") and len(witness_visitor["planned_route"]) > 0:
                witness_visitor["destination"] = witness_visitor["planned_route"].pop(0)
                witness_visitor["cooldown_timer"] = 0
            else:
                witness_visitor["destination"] = "Exit"
        wsize = max(5, int(min(WIDTH,HEIGHT)*0.007))
        pygame.draw.circle(screen, (0,0,255), witness_visitor["position"].astype(int), wsize)
    else:
        route_demo_visitor = None
        witness_visitor = None
    
    # Dessin permanent du trajet optimisé (lignes et nœuds)
    if ui.route_calculated and ui.optimized_route:
        for start, end in ui.route_edges:
            pygame.draw.line(screen, (255,255,0), start, end, 3)
        for pos in ui.route_nodes:
            pygame.draw.circle(screen, (0,255,0), pos, 8)
    
    # Gestion des cycles des attractions
    for attraction in queues:
        _, _, cap, dur = attractions[attraction]
        if cycle_timer[attraction] == 0 and len(queues[attraction]) > 0:
            for _ in range(min(cap, len(queues[attraction]))):
                v = queues[attraction].popleft()
                wt = total_time_elapsed - v["queue_entry_time"]
                if v["fixed_path"]:
                    total_wait_fixed += wt
                    count_fixed += 1
                else:
                    total_wait_adaptive += wt
                    count_adaptive += 1
                v["inside_timer"] = dur * 60
                in_attraction[attraction].append(v)
            cycle_timer[attraction] = dur * 60
        else:
            if cycle_timer[attraction] > 0:
                cycle_timer[attraction] -= 1
    
    for attraction in in_attraction:
        to_rem = []
        for v in list(in_attraction[attraction]):
            if v["inside_timer"] > 0:
                v["inside_timer"] -= 1
            if v["inside_timer"] == 0:
                to_rem.append(v)
        for v in to_rem:
            in_attraction[attraction].remove(v)
            v["in_queue"] = False
            v["cooldown_timer"] = 120
            if not v["fixed_path"]:
                if attraction in v["desires"]:
                    v["desires"].remove(attraction)
            v["last_attraction"] = attraction
            v["exiting_attraction"] = True
            wt = total_time_elapsed - v["queue_entry_time"]
            wait_time_history[attraction].append(wt)
            if v["fixed_path"]:
                total_wait_fixed += wt
                count_fixed += 1
            else:
                total_wait_adaptive += wt
                count_adaptive += 1
            if v["fixed_path"]:
                if v["desires"]:
                    v["destination"] = v["desires"].pop(0)
                else:
                    v["destination"] = "Exit"
                    v["going_to_exit"] = True
            else:
                if v["desires"]:
                    update_visitor_next_destination(v)
                    if v["destination"] == attraction and v["desires"]:
                        update_visitor_next_destination(v)
                else:
                    v["destination"] = "Exit"
                    v["going_to_exit"] = True
            v["exiting_attraction"] = False
            v["cooldown_timer"] = 0
            v["in_queue"] = False
            new_dpos = np.array(attractions.get(v["destination"], exit_gate)[:2], dtype=np.float64)
            dvec = new_dpos - v["position"]
            dnorm = np.linalg.norm(dvec)
            if dnorm > 0:
                v["exit_direction"] = dvec / dnorm
                v["position"] += v["exit_direction"] * v["speed"]
            else:
                v["exit_direction"] = np.array([0,0])
    
    avg_wait_fixed = 10 * (total_wait_fixed / count_fixed) / 60 if count_fixed > 0 else 0
    avg_wait_adaptive = 10 * (total_wait_adaptive / count_adaptive) / 60 if count_adaptive > 0 else 0
    
    margin = 10
    y_pos = margin
    lh = int(HEIGHT * 0.03)
    st = font.render(f"Spawn Curve Factor: {spawn_curve_factor}", True, (255,255,255))
    screen.blit(st, (margin, y_pos)); y_pos += lh
    it = font.render(f"Spawn Interval: {spawn_interval}", True, (255,255,255))
    screen.blit(it, (margin, y_pos)); y_pos += lh
    pt = font.render(f"Spawn Probability: {spawn_probability:.2f}", True, (255,255,255))
    screen.blit(pt, (margin, y_pos)); y_pos += lh
    vc = font.render(f"Visitors in Park: {visitor_count}", True, (255,255,255))
    screen.blit(vc, (margin, y_pos)); y_pos += lh
    ex = font.render(f"Exited Visitors: {exited_count}", True, (255,0,0))
    screen.blit(ex, (margin, y_pos)); y_pos += lh
    fwt = font.render(f"Fixed Path Avg Wait: {avg_wait_fixed:.2f} min", True, (0,255,255))
    screen.blit(fwt, (margin, y_pos)); y_pos += lh
    awt = font.render(f"Adaptive Path Avg Wait: {avg_wait_adaptive:.2f} min", True, (255,140,0))
    screen.blit(awt, (margin, y_pos)); y_pos += lh
    for a in wait_time_history:
        aw = sum(wait_time_history[a]) / len(wait_time_history[a]) / 6 if wait_time_history[a] else 0
        wt_text = font.render(f"{a} Wait: {aw:.2f} min", True, (255,255,255))
        screen.blit(wt_text, (margin, y_pos)); y_pos += lh
    
    if plan_route_button.update(mouse_pos, mouse_clicked):
        ui.toggle()
    plan_route_button.draw(screen)
    
    ui.update(mouse_pos, mouse_clicked)
    ui.draw(screen)
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
