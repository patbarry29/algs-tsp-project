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
import numpy as np

from utils.create_distance_matrix import create_distance_matrix
from utils.generate_tsp import generate_tsp
from opt_solutions import opt_solutions

from ant_colony.ant_colony import ant_colony
from brute_force.brute_force import brute_force
from branch_and_bound.branch_and_bound import branch_and_bound
from branch_and_bound.reduction_matrix_edge_selection import branch_and_bound as matrix_reduction
from dynamic_programming.dynamic_programming import dynamic_programming
from genetic.genetic import genetic
from greedy.greedy import greedy
from lin_kernighan.lin_kernighan import lin_kernighan
from mst.mst import mst
from randomized.randomized import randomized

#%%

class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()

		self.setWindowTitle("TSP Problem Solver")
		self.setGeometry(100, 100, 1000, 600)
		icon_path = os.path.join(current_dir, "icon_use.png")
		self.setWindowIcon(QIcon(icon_path))

		# Main layout
		main_layout = QHBoxLayout()

		# Left panel (1/4)
		left_panel = QWidget()
		self.left_layout = QVBoxLayout()
		self.left_layout.setSpacing(10)
		self.left_layout.setAlignment(Qt.AlignVCenter)

		# Banner
		image_label = QLabel()
		image_label_path = os.path.join(current_dir, "banner.png")
		image_label.setPixmap(QPixmap(image_label_path).scaled(225, 225, Qt.KeepAspectRatio))
		self.left_layout.addWidget(image_label)

		# The Toggle
		self.toggle = QCheckBox("I'm uploading a problem!")
		self.toggle.stateChanged.connect(self.toggle_options)
		self.left_layout.addWidget(self.toggle)

		# Dropdown with known problems
		self.dropdown_problems = QComboBox()
		self.dropdown_problems.addItems(list(opt_solutions.keys()))
		self.dropdown_problems_label = QLabel("Problem Name")

		# Integer input
		self.int_input = QLineEdit()
		self.int_input.setPlaceholderText("20? 45? More?")
		self.int_input_label = QLabel("How many cities will you visit?")
		self.left_layout.addWidget(self.int_input_label)
		self.left_layout.addWidget(self.int_input)

		# Slider
		self.slider = QSlider(Qt.Horizontal)
		self.slider.setMinimum(1)
		self.slider.setMaximum(100)
		self.slider.setValue(80)
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
			"Markov Chain Monte Carlo",
			"Minimum Spanning Tree",
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
			image_label_path = os.path.join(current_dir, "banner.png")
			image_label.setPixmap(QPixmap(image_label_path).scaled(225, 225, Qt.KeepAspectRatio))
			self.left_layout.addWidget(image_label)
			self.left_layout.addWidget(self.toggle)
			self.left_layout.addWidget(self.dropdown_problems_label)
			self.left_layout.addWidget(self.dropdown_problems)
			self.left_layout.addWidget(self.dropdown_label)
			self.left_layout.addWidget(self.dropdown)
			self.left_layout.addWidget(self.run_button)

		else:
			image_label = QLabel()
			image_label_path = os.path.join(current_dir, "banner.png")
			image_label.setPixmap(QPixmap(image_label_path).scaled(225, 225, Qt.KeepAspectRatio))
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

		"""
		Makes a placeholder graph and loads it in the app.
		"""
		
		net = Network(
			height='550px', width="100%",
			bgcolor="#ffffff", cdn_resources='local')
		net.add_node(1, label="Welcome to the TSP Problem Solver!", size=2)
		html_path = os.path.join(current_dir, "graph.html")
		net.write_html(html_path)
		self.web_view.load(QUrl.fromLocalFile(html_path))

	def on_run_button_clicked(self):

		"""
		Handles the button click, updates the graph.
		"""

		algos = {
			"Ant Colony": ant_colony,
			"Branch and Bound": branch_and_bound,
			"Branch and Bound - Matrix Reduction": matrix_reduction,
			"Brute Force": brute_force,
			"Dynamic Programming": dynamic_programming,
			"Genetic Approach": genetic,
			"Greedy": greedy,
			"Lin-Kernighan": lin_kernighan,
			"Markov Chain Monte Carlo": randomized, 
			"Minimum Spanning Tree" : mst
			}

		if self.toggle.isChecked():
			problem_name = self.dropdown_problems.currentText()
			if ".atsp" in problem_name:
				folder = "ALL_atsp"
			else: 
				folder = "ALL_tsp"
			problem_path = os.path.join(parent_dir, 'data', folder, problem_name)
			problem = tsplib95.load(problem_path)
			sparsity_check = True
			#if it's a known problem get the opt too
			opt_cost = opt_solutions[problem_name][0]
			opt_route = opt_solutions[problem_name][1]
			optional_line = f", and the optimum cost is {opt_cost}"
		else:
			# Generate problem
			n_cities = int(self.int_input.text())
			sparsity = self.slider.value() / 100
			cities = generate_tsp(n=n_cities, sparsity=sparsity) #these are coodinates
			problem_path = os.path.join(current_dir, 'data', 'random', 'tsp', 'random_tsp.tsp')
			problem = tsplib95.load(problem_path)
			sparsity_check = False
			optional_line = ""
			opt_route = []

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
				  size=8, shape="circle", color="#050447",
				  font=dict(color="#ccccdd", align="center"))

		if route[-1] ==  route[0]:
			pairs_route = []
		else:
			pairs_route = [(route[-1], route[0])]
		for i in range(len(route)-1):
			pairs_route.append((route[i], route[i+1]))

		if len(opt_route) > 0:
			pairs_opt_route = [(opt_route[-1], opt_route[0])]
			for i in range(len(opt_route)-1):
				pairs_opt_route.append((opt_route[i], opt_route[i+1]))
		else:
			pairs_opt_route = []

		all_pairs = [(route[i], route[j]) for i in range(len(route)) for j in range(len(route)) if i != j]

		# Add route to the network with custom magnitude and color
		for pair in all_pairs:
			source = pair[0]
			target = pair[1]
			length = dist_matrix[source-1, target-1] # the matrix is 0 indexed

			if sparsity_check:
				if (length > 0) and (length < 1000000000):
					if pair in pairs_opt_route:
						edge_color = "#33d11c" #green
					elif pair in pairs_route:
						edge_color = '#0770ff' #blue
					else: 
						edge_color = '#e7e8e7' #Gray

					net.add_edge(int(source), int(target), length=length, color=edge_color)
			else:
				if (length > 0):
					
					if pair in pairs_opt_route:
						edge_color = "#33d11c" #green
					elif pair in pairs_route:
						edge_color = '#0770ff' #blue
					else: 
						edge_color = '#e7e8e7' #Gray

					net.add_edge(int(source), int(target), length=length, color=edge_color)

		net.set_options(f"""
		var options = {{
		"physics": {{
			"enabled": false,
			"stabilization": {{
				"enabled": true,
				"iterations": 50
			}},
			"repulsion": {{
				"centralGravity": 0.9,
				"springLength": 150,
				"nodeDistance": 200
			}}
		}}
		}}""")

		html_path = os.path.join(current_dir, "graph.html")
		net.write_html(html_path)

		#Inject custom JavaScript
		with open(html_path, "r") as file:
			html = file.read()

		# Add the custom JavaScript before the closing </body> tag
		custom_js = f"""
			<script>
				document.getElementById('mynetwork').insertAdjacentHTML('afterbegin', `
			<div style="
				position: absolute;
				top: 10px;
				left: 50%;
				transform: translateX(-50%);
				background-color: white;
				padding: 10px 20px;
				font-size: 16px;
				font-family: Arial, sans-serif;
				border: 1px solid black;
				border-radius: 5px;
			">
				<center>The cost of this path is {cost}{optional_line}</center>
			</div>
			`);
			</script>
		"""
		html = html.replace("</body>", f"{custom_js}</body>")

		# Save the modified HTML
		with open(html_path, "w") as file:
			file.write(html)

		self.web_view.load(QUrl.fromLocalFile(html_path))

# Run the application
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())