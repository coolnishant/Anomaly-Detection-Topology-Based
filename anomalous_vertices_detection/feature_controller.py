from anomalous_vertices_detection.configs.config import *
from anomalous_vertices_detection.feature_extractor import FeatureExtractor
from anomalous_vertices_detection.utils.utils import dict_writer, delete_file_content
import types
from tqdm import tqdm
from six import iteritems


class FeatureController(object):
    """
    This class control the extraction of features from FeatureController.
    """

    def __init__(self, graph_obj):
        self._graph = graph_obj
        self._fe = FeatureExtractor(self._graph)
        self._temp_result = {}
        self._labels = []

    def set_label(self, v, u=None):
        """Set the edge/vertex label attribute in temp_result.
        If v and u are provided it sets the label of the edge
        between v and u. If only v are provided then it sets
        the label of the vertex v.

        Parameters
        ----------
        v : string
            Vertex
        u : string, optional
            Vertex
        """
        if u is None:
            self._temp_result["vertex_label"] = self._fe.get_node_label(v)
        else:
            self._temp_result["edge_label"] = self._fe.get_edge_label(v, u)

    def set_edge_weight(self, u, v):
        if self._graph.has_weight:
            self._temp_result["edge_weight"] = self._graph.get_edge_weight(u, v)

    def set_feature(self, feature_name, feature_func, *args):
        """ Execute and a feature extraction function and set its
         value in temp_result dictionary.

        Parameters
        ----------
        feature_name : string
            The name that should be given to the extracted feature.
            ""
        feature_func : string
            The function that should be executed to extract the feature.
            FeautureExtractor function name to calculate it
        args : tuple
            The feature extraction function arguments.
            (vertex id or two vertices id)
        """
        # here this function calculates all the data set
        self._temp_result[feature_name] = getattr(self._fe, feature_func)(*args)

    def init_entry(self, v, u=None):
        """ Initializing entry for an edge feature extraction.
         Set the edge source destination and label.

        Parameters
        ----------
        v : string
            Vertex
        u : string
            Vertex
        """
        #this line is changed
        # self._temp_result["vertex_label"] = self._fe.get_node_label(v)
        if u:
            self._temp_result = {"src": v, "dst": u}
        else:
            self._temp_result = {"src": v}
        #  this line is changed
        # self.set_label(v, u)

        # self.set_edge_weight(vertex1, vertex2)

    def extract_all_features(self, features_list, vertex1, vertex2=None):
        """Extract all features from predefined feature list for given vertices or edges.
        If two vertices are given then it will extract vertex and edge features.

        If only one vertex the just vertex features will be extracted.

        Parameters
        ----------
        features_list : dict
            Dictionary that contains the feature that should be extracted and with which function.
        vertex1 : string
            Vertex
        vertex2 : string, optional
            Vertex
        """
        self.extract_single_entry(features_list, "vertex_v", vertex1)
        if vertex2:
            self.extract_single_entry(features_list, "link", vertex1, vertex2)
            self.extract_single_entry(features_list, "vertex_u", vertex2)

    def extract_single_entry(self, features_list, features_type, *args):
        """Extract all features for a single vertex/edge.

        Parameters
        ----------
        features_list : dict
            Dict that contains the features that should be extracted by type.
        features_type : string
            The feature type that should be extracted
        args : tuple
            The feature extraction function arguments.
            (vertex id or two vertices id)
        """
        if features_type in features_list:
            for name, feature in iteritems(features_list[features_type]):
                self.set_feature(name, feature, *args)

    def features_generator(self, features_dict, item_iter):
        """ Generator that yield the extracted feature from the edge_iter.

        Parameters
        ----------
        features_dict : dict
            A dictionary that contains the features that should be extracted.
        item_iter : iterator
            Iterator over a list of vertices/edges that features should be extracted for them.
        """
        # for i in item_iter:
        #     print(i)
        #     print(item_iter)
        for count, item in enumerate(item_iter):
            if type(item) is str:
                item = (item,)
            # print(item)
            self.init_entry(*item[:2])  # we get the edge labeled (src, dst, vertex label, edge label)
            if len(item) > 1 and self._graph.has_edge(item[0], item[1]):
                self.extract_features_for_existing_edge(features_dict, item)
                # print('here')
            else:
                self.extract_all_features(features_dict, *(item[:2]))
                # print('here')
            yield self._temp_result

    def extract_features_for_existing_edge(self, features_dict, item):
        """
        Parameters
        ----------
        features_dict : dict
            A dictionary that contains the features that should be extracted.
        item : tuple
            A tuple contains an edge (v,u).
        """
        curr_edge = (item[0], item[1], self._graph.edge(*item))
        self._graph.remove_edge(item[0], item[1])
        #  after removing edge extract feature
        self.extract_all_features(features_dict, item[0], item[1])
        # print(item[0]+' '+item[1])
        #
        self._graph.add_edge(*curr_edge)
        #  reset all neighbor
        self._graph.get_neighbors.reset()

    @staticmethod
    def save_progress(array, temp_path):
        """ Save the progress od the feature extraction.

        Parameters
        ----------
        temp_path
        array : list[dict]
        """
        dict_writer(array, temp_path, "a+")

    def extract_features_to_file(self, data_iter, features_dict, output_path):
        """ Extract features from the graph and saves them to file.

        Parameters
        ----------
        features_dict
        output_path
        data_iter : iterator
            An iterable objects that contains the vertices/edges that their features should be extracted
        max_items_num : int, optional
            Maximal number of features that should be extracted, used to show progress information.
        """
        delete_file_content(output_path)
        features_array = []

        # for i in data_iter:
        #     print(i)
        # data_iter = data_iter[1:]

        for entry in self.features_generator(features_dict, data_iter):
            features_array.append(entry)
            # print(entry)
            # print("\n")
            if len(features_array) == save_progress_interval:
                self.save_progress(features_array, output_path)
                features_array = []
        if features_array:
            self.save_progress(features_array, output_path)
