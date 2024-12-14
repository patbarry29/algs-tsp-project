#%%
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from PyQt5.QtWidgets import (
	QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
	QLabel, QLineEdit, QComboBox, QSlider, QPushButton, QCheckBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QIcon

from pyvis.network import Network
import tsplib95

from utils.create_distance_matrix import create_distance_matrix
from utils.generate_tsp import generate_tsp

# from ant_colony.ant_colony import ant_colony
# from brute_force.brute_force import brute_force
# from branch_and_bound.branch_and_bound import branch_and_bound
# from genetic.genetic import genetic
# from greedy.greedy import greedy
# from lin_kernighan.lin_kernighan import lin_kernighan
#from branch_and_bound.reduction_matrix_edge_selection import solve?
from randomized.randomized import randomized
from dynamic_programming.dynamic_programming import dynamic_programming

#%%

class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()

		self.setWindowTitle("TSP Problem Solver")
		self.setGeometry(100, 100, 1000, 600)
		self.setWindowIcon(QIcon(f"{current_dir}\\icon_use.png"))

		# Main layout
		main_layout = QHBoxLayout()

		# Left panel (1/4)
		left_panel = QWidget()
		self.left_layout = QVBoxLayout()
		self.left_layout.setSpacing(10)
		self.left_layout.setAlignment(Qt.AlignVCenter)

		# Banner
		image_label = QLabel()
		image_label.setPixmap(QPixmap(f"{current_dir}\\banner.png").scaled(225, 225, Qt.KeepAspectRatio))
		self.left_layout.addWidget(image_label)

		# The Toggle
		self.toggle = QCheckBox("I'm uploading a problem!")
		self.toggle.stateChanged.connect(self.toggle_options)
		self.left_layout.addWidget(self.toggle)

		# Dropdown with known problems
		self.dropdown_problems = QComboBox()
		self.dropdown_problems.addItems([
			 "a280.tsp", "ali535.tsp","att48.tsp","att532.tsp","bayg29.tsp","bays29.tsp","berlin52.tsp","bier127.tsp","brazil58.tsp","brd14051.tsp","brg180.tsp","burma14.tsp", "ch130.tsp", "ch150.tsp", "d198.tsp", "d493.tsp", "d657.tsp", "d1291.tsp", 
    "d1655.tsp", "d2103.tsp", "d18512.tsp", "dantzig42.tsp", "dsj1000.tsp", 
    "eil51.tsp", "eil76.tsp", "eil101.tsp", "fl1400.tsp", "fl1577.tsp", 
    "fl3795.tsp", "fnl4461.tsp", "fri26.tsp", "gil262.tsp", "gr17.tsp", 
    "gr21.tsp", "gr24.tsp", "gr48.tsp", "gr96.tsp", "gr120.tsp", 
    "gr137.tsp", "gr202.tsp", "gr229.tsp", "gr431.tsp", "gr666.tsp", 
    "hk48.tsp", "kroA100.tsp", "kroA150.tsp", "kroA200.tsp", "kroB100.tsp", 
    "kroB150.tsp", "kroB200.tsp", "kroC100.tsp", "kroD100.tsp", "kroE100.tsp", 
    "lin105.tsp", "lin318.tsp", "nrw1379.tsp", "p654.tsp", "pa561.tsp", 
    "pa561.tsp", "pcb442.tsp", "pcb1173.tsp", "pcb3038.tsp", "pla7397.tsp", 
    "pla33810.tsp", "pla85900.tsp", "pr76.tsp", "pr107.tsp", "pr124.tsp", 
    "pr136.tsp", "pr144.tsp", "pr152.tsp", "pr226.tsp", "pr264.tsp", 
    "pr299.tsp", "pr439.tsp", "pr1002.tsp", "pr2392.tsp", "rat99.tsp", 
    "rat195.tsp", "rat575.tsp", "rat783.tsp", "rd100.tsp", "rd400.tsp", 
    "rl1304.tsp", "rl1323.tsp", "rl1889.tsp", "rl5915.tsp", "rl5934.tsp", 
    "rl11849.tsp", "si175.tsp", "si535.tsp", "si1032.tsp", "st70.tsp", 
    "swiss42.tsp", "ts225.tsp", "tsp225.tsp", "u159.tsp", "u574.tsp", 
    "u724.tsp", "u1060.tsp", "u1432.tsp", "u1817.tsp", "u2152.tsp", 
    "u2319.tsp", "ulysses16.tsp", "ulysses22.tsp", "usa13509.tsp", 
    "vm1084.tsp", "vm1748.tsp"
    "br17.atsp", "ft53.atsp", "ft70.atsp", "ftv33.atsp", "ftv35.atsp", 
    "ftv38.atsp", "ftv44.atsp", "ftv47.atsp", "ftv55.atsp", "ftv64.atsp", 
    "ftv70.atsp", "ftv170.atsp", "kro124p.atsp", "p43.atsp", "rbg358.atsp", 
    "rbg403.atsp", "rbg443.atsp", "ry48p.atsp",
]
)
		self.dropdown_problems_label = QLabel("Problem Name")

		# Integer input
		self.int_input = QLineEdit()
		self.int_input.setPlaceholderText("20? 45?")
		self.int_input_label = QLabel("How many cities will you visit?")
		self.left_layout.addWidget(self.int_input_label)
		self.left_layout.addWidget(self.int_input)

		# Slider
		self.slider = QSlider(Qt.Horizontal)
		self.slider.setMinimum(1)
		self.slider.setMaximum(100)
		self.slider.setValue(50)
		self.slider_label = QLabel("How connected is your map?")
		self.slider.valueChanged.connect(lambda: self.slider_label.setText(f"{self.slider.value()}% Connected"))
		self.left_layout.addWidget(self.slider_label)
		self.left_layout.addWidget(self.slider)

		# Dropdown
		self.dropdown = QComboBox()
		self.dropdown.addItems([
			"Ant Colony",
			"Branch and Bound",
			"Branch and Bound - Matrix Reduction",
			"Brute Force",
			"Dynamic Programming",
			"Genetic Approach",
			"Greedy",
			"Lin-Kernighan",
			"Randomized"
			])
		self.dropdown_label = QLabel("Which algorithm do you want to try?")
		self.left_layout.addWidget(self.dropdown_label)
		self.left_layout.addWidget(self.dropdown)

		# Button
		self.run_button = QPushButton("Let's Go!")
		self.run_button.clicked.connect(self.on_run_button_clicked)
		self.left_layout.addWidget(self.run_button)

		# Add left layout to the left panel
		left_panel.setLayout(self.left_layout)
		left_panel.setMaximumWidth(250)
		main_layout.addWidget(left_panel)

		# Right panel (3/4) - Graph
		self.web_view = QWebEngineView()
		self.update_graph()
		self.web_view.setStyleSheet("border-radius: 20px;")
		main_layout.addWidget(self.web_view)
		
		# Set main layout
		container = QWidget()
		container.setLayout(main_layout)
		self.setCentralWidget(container)

	def toggle_options(self, state):

		for i in reversed(range(self.left_layout.count())):
			widget = self.left_layout.itemAt(i).widget()
			if widget:
				self.left_layout.removeWidget(widget)
				widget.setParent(None)

		if state == 2:
			image_label = QLabel()
			image_label.setPixmap(QPixmap(f"{current_dir}\\banner.png").scaled(225, 225, Qt.KeepAspectRatio))
			self.left_layout.addWidget(image_label)
			self.left_layout.addWidget(self.toggle)
			self.left_layout.addWidget(self.dropdown_problems_label)
			self.left_layout.addWidget(self.dropdown_problems)
			self.left_layout.addWidget(self.dropdown_label)
			self.left_layout.addWidget(self.dropdown)
			self.left_layout.addWidget(self.run_button)

		else:
			image_label = QLabel()
			image_label.setPixmap(QPixmap(f"{current_dir}\\banner.png").scaled(225, 225, Qt.KeepAspectRatio))
			self.left_layout.addWidget(image_label)
			self.left_layout.addWidget(self.toggle)
			self.left_layout.addWidget(self.int_input_label)
			self.left_layout.addWidget(self.int_input)
			self.left_layout.addWidget(self.slider)
			self.left_layout.addWidget(self.slider_label)
			self.left_layout.addWidget(self.dropdown_label)
			self.left_layout.addWidget(self.dropdown)
			self.left_layout.addWidget(self.run_button)

	def update_graph(self):

		"""Makes a placeholder graph and loads it in the WebEngineView."""

		net = Network(
			height='550px', width="100%",
			bgcolor="#ffffff", cdn_resources='local')
		net.add_node(1, label="Welcome to the TSP Problem Solver!", size=2)
		net.write_html(f"{current_dir}\\graph.html")
		self.web_view.load(QUrl.fromLocalFile(f"{current_dir}\\graph.html"))

	def on_run_button_clicked(self):

		"""Handles the button click, collects input values, and updates the graph."""

		algos = {
			# "Ant Colony": ant_colony,
			# "Branch and Bound": branch_and_bound,
			# "Branch and Bound - Matrix Reduction",
			# "Brute Force": brute_force,
			"Dynamic Programming": dynamic_programming,
			# "Genetic Approach": genetic,
			# "Greedy": greedy,
			# "Lin-Kernighan": lin_Kernighan,
			"Randomized": randomized
			}

		if self.toggle.isChecked():
			if ".atsp" in self.dropdown_problems.currentText():
				folder = "ALL_atsp"
			else: 
				folder = "ALL_tsp"
			problem = tsplib95.load(f"{parent_dir}\\data\\{folder}\\{self.dropdown_problems.currentText()}")
		else:
			# Generate problem
			n_cities = int(self.int_input.text())
			sparsity = self.slider.value()
			cities = generate_tsp(n=n_cities,sparsity=sparsity) #these are coodinates
			problem = tsplib95.load(f"{parent_dir}\\data\\random\\tsp\\random_tsp.tsp")

		# Run Algorithm
		algorithm = self.dropdown.currentText()
		dist_matrix = create_distance_matrix(problem)
		n_cities = len(dist_matrix)
		algo_solve = algos[algorithm](dist_matrix)
		route = algo_solve[0]
		cost = algo_solve[1]

		net = Network(
		height='550px', width="100%",
		bgcolor="#ffffff", cdn_resources='local')

		# Add nodes to the network with custom properties
		for n in range(1, n_cities+1):
			net.add_node(
				n, label = str(n),
				  size=10, shape="circle", color="#050447",
				  font=dict(color="#ccccdd", align="center"))

		pairs_route = [(route[-1], route[0])]
		for i in range(len(route)-1):
			pairs_route.append((route[i], route[i+1]))

		all_pairs = [(route[i], route[j]) for i in range(len(route)) for j in range(len(route)) if i != j]

		# Add route to the network with custom magnitude and color
		for pair in all_pairs:
			source = pair[0]
			target = pair[1]
			length = dist_matrix[source-1, target-1] # the matrix is 0 indexed

			if length > 0:
				if pair in pairs_route:
					edge_color = '#0770ff'
				else: 
					edge_color = '#ccccdd' #Gray

				net.add_edge(int(source), int(target), length=length, color=edge_color)

		net.set_options("""
			var options = {
				"physics": {
					"enabled": false,
				"stabilization": {
						"enabled": true,
						"iterations": 50
					},
				"repulsion": {
						"centralGravity": 0.9,
						"springLength": 150,
						"nodeDistance": 200
					}
				}
			}""")
		net.write_html(f"{current_dir}\\graph.html")

		self.left_layout.addWidget(QLabel(f"🚗 The cost of this path is {cost}"))
		self.web_view.load(QUrl.fromLocalFile(f"{current_dir}\\graph.html"))

# Run the application
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())