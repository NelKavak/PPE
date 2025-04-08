from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, DirectionalLight, LVector3, LPoint3, Point3
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText
import random, numpy as np, heapq
from collections import deque

# --- Paramètres globaux et variables de simulation ---
total_wait_fixed = 0
total_wait_adaptive = 0
count_fixed = 0
count_adaptive = 0
exited_count = 0
MAX_TIME_IN_PARK = 36000

# Définition des attractions : (x, y, capacité, durée)
attractions = {
    "Space Mountain": (200, 200, 25, 1),
    "Big Thunder Mountain": (600, 200, 15, 2),
    "Pirates des Caraïbes": (200, 500, 20, 3),
    "Peter Pan’s Flight": (600, 500, 10, 3),
    "A": (300, 500, 20, 3),
    "B": (300, 100, 40, 3),
    "C": (600, 300, 20, 3),
    "D": (500, 150, 40, 3),
    "E": (700, 200, 20, 3),
    "F": (400, 200, 40, 3),
}
exit_gate = (400, 50)

queues = {name: deque() for name in attractions.keys()}
in_attraction = {name: deque() for name in attractions.keys()}
cycle_timer = {name: 0 for name in attractions.keys()}
wait_time_history = {name: deque(maxlen=55) for name in attractions.keys()}

visitor_spawn_timer = 0
total_time_elapsed = 0
spawn_curve_factor = 100
spawn_interval = 15

# Couleurs spécifiques pour chaque attraction
attraction_colors = {
    "Space Mountain": (1, 0, 0, 1),        # Rouge
    "Big Thunder Mountain": (0, 1, 0, 1),    # Vert
    "Pirates des Caraïbes": (0, 0, 1, 1),      # Bleu
    "Peter Pan’s Flight": (1, 1, 0, 1),        # Jaune
    "A": (1, 0, 1, 1),                         # Magenta
    "B": (0, 1, 1, 1),                         # Cyan
    "C": (0.5, 0.5, 0, 1),                     # Olive
    "D": (0, 0.5, 0.5, 1),                     # Teal
    "E": (0.5, 0, 0.5, 1),                     # Violet
    "F": (0.2, 0.8, 0.2, 1),                   # Vert clair
}

# --- Fonctions de simulation (logique identique à votre code) ---
def build_graph(last_attraction_for_adaptive=None):
    graph = {name: {} for name in attractions}
    for a1 in attractions:
        for a2 in attractions:
            if a1 != a2:
                pos1 = np.array(attractions[a1][:2])
                pos2 = np.array(attractions[a2][:2])
                distance = np.linalg.norm(pos1 - pos2)
                current_queue = len(queues[a2])
                capacity, duration = attractions[a2][2], attractions[a2][3]
                penalty = 1000 if last_attraction_for_adaptive and a2 == last_attraction_for_adaptive else 0
                cost = (distance / 10) + (current_queue / max(1, capacity)) * duration + penalty
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
    queue = []
    heapq.heappush(queue, (0, start, []))
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
    return best_path if best_path else ["Exit"]

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
    valid_destinations = [d for d in visitor["desires"] if d != visitor["last_attraction"]]
    if not valid_destinations:
        visitor["destination"] = "Exit"
        visitor["going_to_exit"] = True
        return
    graph = build_graph(visitor["last_attraction"])
    optimal_path = dijkstra(graph, visitor["last_attraction"], valid_destinations)
    if optimal_path and len(optimal_path) > 1:
        visitor["planned_route"] = optimal_path[1:]
        visitor["destination"] = visitor["planned_route"].pop(0)
        visitor["commit_to_destination"] = True
    else:
        visitor["destination"] = "Exit"
        visitor["going_to_exit"] = True

def generate_new_visitor(total_time_elapsed):
    attraction_weights = {
        "Space Mountain": 10,
        "Big Thunder Mountain": 9,
        "Pirates des Caraïbes": 4,
        "Peter Pan’s Flight": 7,
        "A": 5,
        "B": 8,
        "C": 3,
        "D": 2,
        "E": 6,
        "F": 9,
    }
    attractions_list = list(attraction_weights.keys())
    weights = list(attraction_weights.values())
    num_visits = random.randint(3, len(attractions_list) + 2)
    desires = random.choices(attractions_list, weights=weights, k=num_visits)
    fixed_path = random.random() < 0.8
    visitor = {
        "position": np.array([exit_gate[0], exit_gate[1], 0], dtype=np.float64),
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
        "fixed_path": fixed_path,
        "planned_route": [],
        "commit_to_destination": True,
        "start_time": total_time_elapsed,
        "stuck_timer": 0,
        "prev_position": np.array([exit_gate[0], exit_gate[1], 0], dtype=np.float64),
        "prev_destination": None,
        "counted_finished": False,
    }
    if not fixed_path:
        graph = build_graph()
        optimal_path = dijkstra(graph, "exit_gate", desires)
        if optimal_path and len(optimal_path) > 1:
            visitor["planned_route"] = optimal_path[1:]
            visitor["destination"] = visitor["planned_route"].pop(0)
        visitor["commit_to_destination"] = True
    return visitor

# --- Classe Simulation 3D avec Panda3D ---
class Simulation(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.disableMouse()
        self.camera.setPos(400, -800, 400)
        self.camera.lookAt(400, 300, 0)
        self.setBackgroundColor(0.1, 0.1, 0.1, 1)
        
        # Éclairage
        ambientLight = AmbientLight("ambientLight")
        ambientLight.setColor((0.3, 0.3, 0.3, 1))
        self.render.setLight(self.render.attachNewNode(ambientLight))
        
        directionalLight = DirectionalLight("directionalLight")
        directionalLight.setDirection(LVector3(-1, -1, -1))
        directionalLight.setColor((0.7, 0.7, 0.7, 1))
        self.render.setLight(self.render.attachNewNode(directionalLight))
        
        self.total_time_elapsed = 0
        self.visitor_spawn_timer = 0
        self.spawn_curve_factor = 100
        self.spawn_interval = 15
        
        # Variables de simulation globales
        self.attractions = attractions
        self.exit_gate = exit_gate
        self.queues = queues
        self.in_attraction = in_attraction
        self.cycle_timer = cycle_timer
        
        self.visitors = []
        
        self.total_wait_fixed = total_wait_fixed
        self.total_wait_adaptive = total_wait_adaptive
        self.count_fixed = count_fixed
        self.count_adaptive = count_adaptive
        self.exited_count = exited_count
        
        # Création des nœuds d'attractions
        self.attraction_nodes = {}
        self.attraction_texts = {}
        # On récupère les dimensions de la fenêtre pour le positionnement du texte
        win_width = self.win.getXSize()
        win_height = self.win.getYSize()
        for name, (x, y, cap, dur) in self.attractions.items():
            node = self.loader.loadModel("models/box")
            node.setScale(15)
            node.setPos(x, y, 0)
            col = attraction_colors.get(name, (1, 1, 1, 1))
            node.setColor(*col)
            node.reparentTo(self.render)
            self.attraction_nodes[name] = node
            # Affichage du nom en overlay : conversion simple des coordonnées
            # Ici, on fait un mapping de x dans [0,800] vers [-1,1] et de y dans [0,600] vers [1,-1]
            text_x = (x/ win_width)*2 - 1
            text_y = 1 - (y/ win_height)*2
            self.attraction_texts[name] = OnscreenText(text=name, pos=(text_x, text_y),
                                                       scale=0.07, fg=col, mayChange=False)
        
        # Création de la porte de sortie
        self.exit_node = self.loader.loadModel("models/smiley")
        self.exit_node.setScale(15)
        self.exit_node.setPos(self.exit_gate[0], self.exit_gate[1], 0)
        self.exit_node.setColor(0, 0, 1, 1)
        self.exit_node.reparentTo(self.render)
        
        # Dashboard
        self.dashboard = {}
        self.dashboard["visitors"] = OnscreenText(text="Visitors: 0", pos=(-1.3, 0.9), scale=0.05, fg=(1,1,1,1))
        self.dashboard["exited"] = OnscreenText(text="Exited: 0", pos=(-1.3, 0.8), scale=0.05, fg=(1,0,0,1))
        self.dashboard["avg_fixed"] = OnscreenText(text="Fixed Avg Wait: 0 min", pos=(-1.3, 0.7), scale=0.05, fg=(0,1,1,1))
        self.dashboard["avg_adaptive"] = OnscreenText(text="Adaptive Avg Wait: 0 min", pos=(-1.3, 0.6), scale=0.05, fg=(1,0.5,0,1))
        self.dashboard["spawn_curve"] = OnscreenText(text="Spawn Curve: 100", pos=(-1.3, 0.5), scale=0.05, fg=(1,1,1,1))
        self.dashboard["spawn_interval"] = OnscreenText(text="Spawn Interval: 15", pos=(-1.3, 0.4), scale=0.05, fg=(1,1,1,1))
        self.dashboard["spawn_prob"] = OnscreenText(text="Spawn Prob: 1.00", pos=(-1.3, 0.3), scale=0.05, fg=(1,1,1,1))
        
        self.taskMgr.add(self.update_simulation, "update_simulation")
    
    def update_dashboard(self):
        self.dashboard["visitors"].setText(f"Visitors: {len(self.visitors)}")
        self.dashboard["exited"].setText(f"Exited: {self.exited_count}")
        avg_fixed = (self.total_wait_fixed/self.count_fixed)/60 if self.count_fixed > 0 else 0
        avg_adaptive = (self.total_wait_adaptive/self.count_adaptive)/60 if self.count_adaptive > 0 else 0
        self.dashboard["avg_fixed"].setText(f"Fixed Avg Wait: {avg_fixed:.2f} min")
        self.dashboard["avg_adaptive"].setText(f"Adaptive Avg Wait: {avg_adaptive:.2f} min")
        self.dashboard["spawn_curve"].setText(f"Spawn Curve: {self.spawn_curve_factor}")
        self.dashboard["spawn_interval"].setText(f"Spawn Interval: {self.spawn_interval}")
        spawn_probability = max(0.1, min(1.0, 1 - np.exp(-self.total_time_elapsed / self.spawn_curve_factor)))
        self.dashboard["spawn_prob"].setText(f"Spawn Prob: {spawn_probability:.2f}")
    
    def update_simulation(self, task):
        dt = globalClock.getDt()
        self.total_time_elapsed += 1
        self.visitor_spawn_timer -= 1
        spawn_probability = max(0.1, min(1.0, 1 - np.exp(-self.total_time_elapsed / self.spawn_curve_factor)))
        
        if self.visitor_spawn_timer <= 0:
            num_visitors = random.randint(2, 5)
            for _ in range(num_visitors):
                if random.random() < spawn_probability:
                    self.visitors.append(generate_new_visitor(self.total_time_elapsed))
            self.visitor_spawn_timer = self.spawn_interval
        
        for visitor in self.visitors:
            if self.total_time_elapsed - visitor["start_time"] > MAX_TIME_IN_PARK:
                visitor["destination"] = "Exit"
                visitor["going_to_exit"] = True
                print(f"Forced exit: {id(visitor)}")
            
            dest_name = visitor["destination"]
            dest_pos = np.array(self.attractions.get(dest_name, self.exit_gate)[:2], dtype=np.float64)
            dest_pos = np.append(dest_pos, 0)
            direction = dest_pos - visitor["position"]
            distance = np.linalg.norm(direction)
            if distance > 5:
                visitor["position"] += (direction / distance) * visitor["speed"]
            else:
                if dest_name in self.attractions and not visitor["in_queue"]:
                    visitor["in_queue"] = True
                elif dest_name == "Exit":
                    visitor["finished"] = True
                    if not visitor.get("counted_finished", False):
                        visitor["counted_finished"] = True
                        self.exited_count += 1
                        print(f"Visitor {id(visitor)} exited. Total exited: {self.exited_count}")
        
        self.camera.setPos(400, -800, 400)
        self.camera.lookAt(400, 300, 0)
        
        self.update_dashboard()
        
        # Mise à jour des nœuds visiteurs
        for visitor in self.visitors:
            if "node" not in visitor:
                node = self.loader.loadModel("models/smiley")
                node.setScale(5)
                node.reparentTo(self.render)
                visitor["node"] = node
            visitor["node"].setPos(visitor["position"][0], visitor["position"][1], visitor["position"][2])
            if visitor["going_to_exit"]:
                col = (1, 1, 0, 1)  # Jaune
            else:
                col = (1, 0, 0, 1) if visitor["fixed_path"] else (1, 0.5, 0, 1)
            visitor["node"].setColor(*col)
        
        self.visitors = [v for v in self.visitors if not v["finished"]]
        
        return Task.cont

if __name__ == "__main__":
    sim = Simulation()
    sim.run()
