import os
from anomalous_vertices_detection.configs.config import DATA_DIR
from anomalous_vertices_detection.configs.graph_config import GraphConfig
from anomalous_vertices_detection.graphs.graph_factory import GraphFactory


def load_data(dataset_file_name="twitter.csv.gz", labels_file_name="twitter_fake_ids.csv", labels_map=None, limit=50000):
    data_path = os.path.join(DATA_DIR, dataset_file_name)

    labels_path = os.path.join(DATA_DIR, labels_file_name)

    #  setting features of twitter graph
    twitter_config = GraphConfig("twitter", data_path,
                                 is_directed=True, labels_path=labels_path,
                                 graph_type="regular", vertex_min_edge_number=10, vertex_max_edge_number=50000)

    #  create and return graph
    return GraphFactory().factory(twitter_config, labels=labels_map, limit=limit), twitter_config
