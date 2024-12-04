#%%
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from PyQt5.QtWidgets import (
	QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
	QLabel, QLineEdit, QComboBox, QSlider, QPushButton
)
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QIcon

import plotly.graph_objects as go
import networkx as nx
import tsplib95

from utils.create_distance_matrix import create_distance_matrix
from utils.generate_tsp import generate_tsp
from randomised.randomised import randomised
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
		left_layout = QVBoxLayout()
		left_layout.setSpacing(10)
		left_layout.setAlignment(Qt.AlignVCenter)

		# Banner
		image_label = QLabel()
		image_label.setPixmap(QPixmap(f"{current_dir}\\banner.png").scaled(225, 225, Qt.KeepAspectRatio))
		left_layout.addWidget(image_label)

		# Integer input
		self.int_input = QLineEdit()
		self.int_input.setPlaceholderText("20? 45?")
		left_layout.addWidget(QLabel("How many cities will you visit?"))
		left_layout.addWidget(self.int_input)

		# Slider
		self.slider = QSlider(Qt.Horizontal)
		self.slider.setMinimum(1)
		self.slider.setMaximum(100)
		self.slider.setValue(50)
		self.slider_label = QLabel("How connected is your map?")
		self.slider.valueChanged.connect(lambda: self.slider_label.setText(f"{self.slider.value()}% Connected"))
		left_layout.addWidget(self.slider_label)
		left_layout.addWidget(self.slider)

		# Dropdown
		self.dropdown = QComboBox()
		self.dropdown.addItems([
			"Ant Colony",
			"Branch and Bound",
			"Brute Force",
			"Dynamic Programming",
			"Genetic Approach",
			"Greedy",
			"Randomised"
			])
		left_layout.addWidget(QLabel("Which algorithm do you want to try?"))
		left_layout.addWidget(self.dropdown)

		# Button
		self.run_button = QPushButton("Let's Go!")
		self.run_button.clicked.connect(self.on_run_button_clicked)
		left_layout.addWidget(self.run_button)

		# Add left layout to the left panel
		left_panel.setLayout(left_layout)
		left_panel.setMaximumWidth(250)
		main_layout.addWidget(left_panel)

		# Right panel (3/4) - Plotly graph
		self.web_view = QWebEngineView()
		self.update_graph()
		self.web_view.setStyleSheet("border-radius: 20px;")
		main_layout.addWidget(self.web_view)
		

		# Set main layout
		container = QWidget()
		container.setLayout(main_layout)
		self.setCentralWidget(container)

	def update_graph(self):

		"""Updates the Plotly graph and loads it in the WebEngineView."""

		fig = go.Figure()
		fig.add_annotation(
			text="Welcome to the Traveling Salesman Problem Solver!",
			showarrow=False,
			font=dict(size=24, color="#290a59"),
			xref="paper", yref="paper",
			x=0.5, y=0.5,
			align="center"
		)

		# Remove axes, gridlines, and background for a clean look
		fig.update_layout(
			xaxis_visible=False,
			yaxis_visible=False,
			plot_bgcolor="white",
			paper_bgcolor="white",
			margin=dict(t=0, b=0, l=0, r=0)  # Minimize margins
		)

		# Save the graph as an HTML file and load it into the QWebEngineView
		fig.write_html(f"{current_dir}\\graph.html")
		self.web_view.load(QUrl.fromLocalFile(f"{current_dir}\\graph.html"))

	def on_run_button_clicked(self):

		"""Handles the button click, collects input values, and updates the graph."""

		algos = {
			"Randomised": randomised,
			"Dynamic Programming": dynamic_programming
			}

		try:
			n_cities = int(self.int_input.text())
			algorithm = self.dropdown.currentText()
			sparsity = self.slider.value()

			# Generate problem and Run Algorithm
			cities = generate_tsp(n=n_cities,sparsity=sparsity) #these are coodinates
			problem = tsplib95.load(f"{parent_dir}\\data\\random\\tsp\\random_tsp.tsp")
			dist_matrix = create_distance_matrix(problem)

			algo_solve = algos[algorithm](dist_matrix)
			route = algo_solve[0]
			cost = algo_solve[1]

			# Create a graph object
			G = nx.Graph()
			#cities = create_distance_matrix(tsplib95.load(r'burma14.tsp'))

			# Add edges with weights from the matrix
			num_nodes = len(dist_matrix)
			for i in range(num_nodes):
				for j in range(num_nodes):
					if dist_matrix[i, j] > 0:
						G.add_edge(i + 1, j + 1, weight=dist_matrix[i, j])

			# Sse the generated problem if they're coordinates, else Initialize random positions if needed 
			if cities.shape[1] == 2:
				pos = cities.copy()
			else:
				pos = nx.spring_layout(G)

			# Plot
			node_x = []
			node_y = []
			for node in G.nodes():
				x, y = pos[node-1]
				node_x.append(x)
				node_y.append(y)

			edge_x = []
			edge_y = []
			for edge in G.edges():
				x0, y0 = pos[edge[0]-1]
				x1, y1 = pos[edge[1]-1]
				edge_x.append(x0)
				edge_x.append(x1)
				edge_x.append(None)
				edge_y.append(y0)
				edge_y.append(y1)
				edge_y.append(None)

			edges_route = [(route[-1], route[0])]
			for i in range(len(route)-1):
				edges_route.append((int(route[i]), int(route[i+1])))

			edge_x_path = []
			edge_y_path = []
			for edge in edges_route:
				x0, y0 = pos[edge[0]-1]
				x1, y1 = pos[edge[1]-1]
				edge_x_path.append(x0)
				edge_x_path.append(x1)
				edge_x_path.append(None)
				edge_y_path.append(y0)
				edge_y_path.append(y1)
				edge_y_path.append(None)

			edge_trace = go.Scatter(
				x=edge_x, y=edge_y,
				line=dict(width=0.5, color='#888'),
				hoverinfo='none',
				mode='lines')

			path_trace = go.Scatter(
				x=edge_x_path, y=edge_y_path,
				line=dict(width=1.5, color='#0770ff'),
				hoverinfo='none',
				mode='lines'
				)

			node_trace = go.Scatter(
				x=node_x, y=node_y,
				mode='markers',
				hoverinfo='text',
				marker=dict(
					size=15, line_width=0, color="#290a59")
					)

			fig = go.Figure(
					data=[edge_trace, path_trace, node_trace],
					layout=go.Layout(
					hovermode='closest',
					margin=dict(b=10,l=5,r=5,t=10),
					annotations=[dict(
					text=f"Cost of the Route: {cost}",
					showarrow=False,
					xref="paper", yref="paper",
					x=0.005, y=-0.002)],
					xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
					yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
							))

			fig.update_layout(
				height=550, width=700, showlegend=False, 
				plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
				)
			fig.write_html(f"{current_dir}\\graph.html")

			self.web_view.load(QUrl.fromLocalFile(f"{current_dir}\\graph.html"))

		except ValueError:
			print("Please enter a valid integer!")

# Run the application
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())