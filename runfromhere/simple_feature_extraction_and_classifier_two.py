from random import randint

from anomalous_vertices_detection.randomforest.myrandomforest import *

from anomalous_vertices_detection.feature_controller import *
from anomalous_vertices_detection.datasets.twitter import load_data
from anomalous_vertices_detection.configs.predefined_features_sets import fast_link_features, fast_vertex_features
import pandas as pd
import numpy as np

# labels = {"neg": "Real", "pos": "Fake"}

labels = {"neg": 1, "pos":0}

my_graph, dataset_config = load_data(labels_map=labels, limit=50000)
print(my_graph.number_of_vertices)
print(len(my_graph.edges))
edges_output_path = "../output/" + dataset_config.name + "_edges.csv"
vertices_output_path = "../output/" + dataset_config.name + "_vertices.csv"
# print(len(my_graph.vertices))
print("Please wait, Features are being extracted!");
features = FeatureController(my_graph)

# Edge feature extraction
# features.extract_features_to_file(my_graph.edges[2:500], fast_link_features[my_graph.is_directed], edges_output_path)
# print("Edge feature Saved to "+vertices_output_path)

# Vertex feature extraction

no_of_vertices = 5000

x = randint(2, my_graph.number_of_vertices-no_of_vertices-2)
features.extract_features_to_file(my_graph.vertices[x:x+no_of_vertices], fast_vertex_features[my_graph.is_directed], vertices_output_path)
print("Vertices feature Saved to "+vertices_output_path)

# print(my_graph.vertices[2:100])
# print('Drawing of Graph: It might take a while. Hold on.')
# my_graph.draw_graph(my_graph.vertices[2:10])

#calling randomforest
randomforestcalling(vertices_output_path = vertices_output_path)

