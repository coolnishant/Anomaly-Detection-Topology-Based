from anomalous_vertices_detection.graphs.nxgraph import NxGraph



class GraphFactory(object):
    def factory(self, graph_config, labels=None, limit=50000):
           return self.make_graph(graph_config.data_path, graph_config.is_directed, graph_config.labels_path,
                                   max_num_of_edges=limit, start_line=graph_config.first_line,
                                   pos_label=labels["pos"],
                                   neg_label=labels["neg"], delimiter=graph_config.delimiter)

    def make_graph(self, graph_path, is_directed=False, labels_path=None, pos_label=None,
                   neg_label=None, start_line=0, max_num_of_edges=10000,delimiter=','):
        """
            Loads graph into specified package.
            Parameters
            ----------
            blacklist_path
            delimiter
            graph_path : string

            is_directed : boolean, optional (default=False)
               Hold true if the graph is directed otherwise false.

            labels_path : string or None, optional (default=False)
               The path of the node labels file.

            package : string(Networkx, GraphLab or GraphTool), optional (default="Networkx")
               The name of the package to should be used to load the graph.

            pos_label : string or None, optional (default=None)
               The positive label.

            neg_label : string or None, optional (default=None)
               The negative label.

            start_line : integer, optional (default=0)
               The number of the first line in the file to be read.

            max_num_of_edges : integer, optional (default=10000000)
               The maximal number of edges that should be loaded.

            weight_field : string

            Returns
            -------
            g : AbstractGraph
                A graph object with the randomly generated nodes.

        """
        graph = NxGraph(is_directed)
        if labels_path:
            print("Loading labels...")
            graph.load_labels(labels_path)
        graph.map_labels(positive=pos_label, negative=neg_label)
        print("Loading graph...")
        graph.load_graph(graph_path, start_line=start_line, limit=max_num_of_edges, delimiter=delimiter)
        print("Data loaded.")
        # print("Drawing Graph")
        # graph.draw_graph()
        return graph


