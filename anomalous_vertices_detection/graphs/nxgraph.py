import community
import networkx as nx
from networkx.classes.reportviews import OutEdgeView, EdgeView, NodeView
import matplotlib.pyplot as plt
from anomalous_vertices_detection.graphs import AbstractGraph
from anomalous_vertices_detection.utils.utils import *
from itertools import islice


class NxEdgeView(EdgeView):
    def __getitem__(self, key):
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            return islice(self.__iter__(), key.start, key.stop, key.step)
        elif isinstance(key, int):
            return next(islice(self.__iter__(), key, key+1))


class NxOutEdgeView(OutEdgeView):
    def __getitem__(self, key):
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            return islice(self.__iter__(), key.start, key.stop, key.step)
        elif isinstance(key, int):
            return next(islice(self.__iter__(), key, key+1))


class NxNodeView(NodeView):
    def __getitem__(self, key):
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            return islice(self.__iter__(), key.start, key.stop, key.step)
        elif isinstance(key, int):
            return next(islice(self.__iter__(), key, key+1))

class NxGraph(AbstractGraph):
    # __slots__ = []

    def __init__(self, is_directed=False, weight_field=None, graph_obj=None):
        """ Initialize a graph

        Parameters
        ----------
        is_directed: bool, optional (default=False)
            True if the graph direted otherwise False.
        weight_field: string, optional (default=None)
            The name of the weight attribute if exist.
        graph_obj: Graph (NetworkX), optional (default=[])
            A NetworkX graph. If graph_obj=None (default) an empty
            NetworkX graph is created.
        Examples
        --------
        >>> G = NxGraph()
        >>> G = NxGraph(True, "weight")
        >>> e = [(1,2),(2,3),(3,4)]
        >>> nx_graph = nx.Graph(e) #NetworkX Graph
        >>> G = NxGraph(graph_obj=nx_graph)
        """
        super(NxGraph, self).__init__(weight_field)
        if graph_obj:
            self._graph = graph_obj.copy()
        else:
            if is_directed is False:
                self._graph = nx.Graph()
            else:
                self._graph = nx.DiGraph()

    def add_node(self, vertex, attr_dict=None):
        self._graph.add_node(vertex, **attr_dict)

    def add_edge(self, vertex1, vertex2, edge_atrr=None):
        """ Adds a new edge to the graph
        Parameters
        ----------
        vertex1: string
            The name of the source vertex
        vertex2: string
            The name of the destination vertex
        edge_atrr: dict
            The attributes and the values of the edge

        Examples
        --------
        >>> g.add_edge("a","b")
        >>> g.add_edge("c","b",{"weight": 5})
        """
        vertex1, vertex2 = str(vertex1).strip(), str(vertex2).strip()
        self._graph.add_edge(vertex1, vertex2, attr=edge_atrr)

    def remove_edge(self, v, u):
        """Remove the edge between u and v.

        Parameters
        ----------
        v: string
            The source vertex.
        u: string
            The destination vertex.


        Examples
        --------
        >>> g.remove_edge("1","2")
        """
        self._graph.remove_edge(v, u)

    def get_vertex_edges(self, vertex, mode="out"):
        """ Return a list of edges.

        Parameters
        ----------
        vertex: string
            The node that its edge should be returned
        mode: string, optional (default=out)
            Equals to "out" to return outbound edges.
            Equals to "in" to return inbound edges.
            Equals to "all" to return all edges.

        Returns
        -------
        edge_list: list
            Edges that are adjacent to vertex.

        Examples
        --------
        >>> g.get_vertex_edges("1", "out")
        """
        if mode == "out":
            return self._graph.edges(vertex)
        if mode == "in":
            return self._graph.in_edges(vertex)
        if mode == "all":
            return self._graph.edges(vertex) + self._graph.in_edges(vertex)

    def get_edge_label(self, vertex1, vertex2):
        """ Return the label of the edge.

        Parameters
        ----------
        vertex1: string
            The source vertex.
        vertex2: string
            The destination vertex.

        Returns
        -------
        string: The edge label.

        Examples
        --------
        >>> g.get_edge_label("1", "2")
        """
        if self.has_edge(vertex1, vertex2):
            if "edge_label" in self._graph[vertex1][vertex2]:
                return self._graph[vertex1][vertex2]['edge_label']
        return self.negative_label

    def get_edge_weight(self, u, v):
        """ Return the edge weight.

        Parameters
        ----------
        v: string
            The source vertex.
        u: string
            The destination vertex.

        Returns
        -------
        int: The edge weight value

        Examples
        --------
        >>> g.get_edge_weight("1", "2")
        """
        return self.edge(u, v)[self._weight_field]

    def get_node_attributes(self, u):
        """ Return the node attributes.

        Parameters
        ----------
        u: string
            vertex.

        Returns
        -------
        dict : The node attribute dict

        Examples
        --------
        >>> g.get_node_attributes("1")
        """
        return self._graph.node[u]

    def edge(self, u, v):
        """ Return the edge
        Parameters
        ----------
        v: string
            The source vertex.
        u: string
            The destination vertex.

        Returns
        -------
        edge: The edge

        Examples
        --------
        >>> g.edge("1", "2")
        """
        return self._graph[u][v]

    @property
    def vertices(self):
        """ Return all the vertices in the graph.

        Returns
        -------
        list: List of all the vertices.

        Examples
        --------
        >>> g.vertices
        """
        return NxNodeView(self._graph)

    @property
    def vertices_iter(self):
        """Return an iterator over the vertices.

        Returns
        -------
        iterator: An iterator over all the vertices.

        Examples
        --------
        >>> g.vertices_iter
        """
        return self._graph.nodes_iter()

    @property
    def edges(self):
        """Return all the edges in the graph.

        Returns
        -------
        list: list of all edges

        Examples
        --------
        >>> g.edges
        """
        if self.is_directed:
            return NxOutEdgeView(self._graph)
        else:
            return NxEdgeView(self._graph)
        # return self._graph.edges()

    @property
    def number_of_vertices(self):
        """Return the of the vertices in the graph.

        Returns
        -------
        int: The number of vertices in the graph.

        Examples
        --------
        >>> g.number_of_vertices()
        """
        return self._graph.number_of_nodes()

    @property
    def is_directed(self):
        """ Return True if graph is directed, False otherwise.

        Examples
        --------
        >>> g.is_directed
        """
        return self._graph.is_directed()


    # @property
    def pagerank(self):
        """Return the PageRank of the nodes in the graph.

        Returns
        -------
        pagerank : dictionary
            Dictionary of nodes with PageRank as value

        Examples
        --------
        >>> g.pagerank()
         """
        return nx.pagerank(self._graph, weight=self._weight_field)

    def katz(self):
        """Compute the Katz centrality for the nodes of the graph G.

        Returns
        -------
        nodes : dictionary
        Dictionary of nodes with Katz centrality as the value.
        """
        return nx.katz_centrality(self._graph, weight=self._weight_field)

    # @property
    def hits(self, max_iter=100, normalized=True):
        """Return HITS hubs and authorities values for nodes.
        max_iter : interger, optional
          Maximum number of iterations in power method.

        normalized : bool (default=True)

        Normalize results by the sum of all of the values.
        Returns
        -------
        (hubs,authorities) : two-tuple of dictionaries
            Two dictionaries keyed by node containing the hub and authority
            values.

        Examples
        --------
        >>>
        """
        return nx.hits(self._graph, max_iter=max_iter, normalized=normalized)

    # @property
    def eigenvector(self):
        """ Compute the eigenvector centrality for the graph G.

        Returns
        -------
        nodes : dictionary
            Dictionary of nodes with eigenvector centrality as the value.

        Examples
        --------
        >>>
        """
        return nx.eigenvector_centrality(self._graph, weight=self._weight_field)

    # @property
    def load_centrality(self):
        """Compute load centrality for nodes.

        Returns
        -------
        nodes : dictionary
            Dictionary of nodes with centrality as the value.

        Examples
        --------
        >>>
        """
        return nx.load_centrality(self._graph, weight=self._weight_field)

    # @property
    def communicability_centrality(self):
        """Return communicability centrality for each node in G.

        If is the graph is directed it will be converted to undirected.

        Returns
        -------
        nodes: dictionary
            Dictionary of nodes with communicability centrality as the value.

        Examples
        --------
        >>>
        """
        if self.is_directed:
            return nx.communicability_centrality(self._graph.to_undirected())
        else:
            return nx.communicability_centrality(self._graph)

    # @property
    def betweenness_centrality(self):
        """
        Parameters
        ----------

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return nx.betweenness_centrality(self._graph, weight=self._weight_field)

    def closeness(self):
        """
        Parameters
        ----------

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return nx.closeness_centrality(self._graph)

    def get_vertex_degree(self, vertex):
        """
        Parameters
        ----------
        vertex:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return float(self._graph.degree(vertex))

    def get_shortest_path_length(self, vertex1, vertex2):
        """
        Parameters
        ----------
        vertex1:
        vertex2:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        try:
            return nx.shortest_path_length(self._graph, source=vertex1, target=vertex2, weight=self._weight_field)
        except nx.NetworkXNoPath:
            return 0

    def get_shortest_path_length_with_limit(self, vertex1, vertex2, cutoff=None):
        """
        Parameters
        ----------
        vertex1:
        vertex2:
        cutoff:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        try:
            return nx.single_source_dijkstra(self._graph, source=vertex1, target=vertex2, cutoff=cutoff,
                                             weight=self._weight_field)
        except nx.NetworkXNoPath:
            return 0

    def get_vertex_in_degree(self, vertex):
        """
        Parameters
        ----------
        vertex:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        if self.is_directed:
            return float(self._graph.in_degree(vertex))
        return None

    def get_vertex_out_degree(self, vertex):
        """
        Parameters
        ----------
        vertex:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        if self.is_directed:
            return float(self._graph.out_degree(vertex))
        return float(self._graph.degree(vertex))

    def get_subgraph(self, vertices):
        """
        Parameters
        ----------
        vertices:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return NxGraph(self.is_directed, self._weight_field, self._graph.subgraph(vertices))

    def has_edge(self, node1, node2):
        """
        Parameters
        ----------
        node1:
        node2:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return self._graph.has_edge(node1, node2)

    @property
    def connected_components(self):
        """
        Parameters
        ----------

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return nx.connected_components(self._graph)

    @memoize
    def get_neighbors(self, node):
        """
        Parameters
        ----------
        node:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return list(self._graph.neighbors(node))

    def neighbors_iter(self, vertex):
        """
        Parameters
        ----------
        vertex:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return self._graph.neighbors(vertex)

    def get_followers(self, node):
        """
        Parameters
        ----------
        node:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        if self.is_directed:
            return self._graph.predecessors(node)
        else:
            return self.get_neighbors(node)

    def get_clustering_coefficient(self, node):
        """
        Parameters
        ----------
        node:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return nx.clustering(self._graph, node, weight=self._weight_field)

    def disjoint_communities(self):
        """
        Parameters
        ----------

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        if self.is_directed:
            partition = community.best_partition(self._graph.to_undirected())
        else:
            partition = community.best_partition(self._graph)
        return partition

    def average_neighbor_degree(self):
        """
        Parameters
        ----------

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return nx.average_neighbor_degree(self._graph)

    def degree_centrality(self):
        """
        Parameters
        ----------

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return nx.degree_centrality(self._graph)

    def in_degree_centrality(self):
        """
        Parameters
        ----------

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return nx.in_degree_centrality(self._graph)

    def out_degree_centrality(self):
        """
        Parameters
        ----------

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return nx.out_degree_centrality(self._graph)

    def get_scc_number(self, vertices):
        """
        Parameters
        ----------
        vertices:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        neighborhood_subgraph = self.get_neighborhoods_subgraph(vertices)
        return nx.number_strongly_connected_components(neighborhood_subgraph._graph)

    def get_scc_number_plus(self, vertices):
        """
        Parameters
        ----------
        vertices:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        neighborhood_subgraph = self.get_neighborhoods_subgraph_plus(vertices)
        return nx.number_strongly_connected_components(neighborhood_subgraph._graph)

    def get_wcc_number(self, vertices):
        """
        Parameters
        ----------
        vertices:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        neighborhood_subgraph = self.get_neighborhoods_subgraph(vertices)
        return nx.number_weakly_connected_components(neighborhood_subgraph._graph)

    def get_inner_subgraph_scc_number(self, node1, node2):
        """
        Parameters
        ----------
        node1:
        node2:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        inner_subgraph = self.get_inner_subgraph(node1, node2)
        return nx.number_strongly_connected_components(inner_subgraph._graph)

    def get_inner_subgraph_wcc_number(self, node1, node2):
        """
        Parameters
        ----------
        node1:
        node2:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        inner_subgraph = self.get_inner_subgraph(node1, node2)
        return nx.number_weakly_connected_components(inner_subgraph._graph)

    def nodes_number_of_cliques(self):
        """
        Parameters
        ----------

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        if self.is_directed:
            return nx.number_of_cliques(self._graph.to_undirected())
        else:
            return nx.number_of_cliques(self._graph)

    def get_adamic_adar_index(self, node1, node2):
        """
        Parameters
        ----------
        node1:
        node2:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        try:
            return nx.adamic_adar_index(self._graph, [(node1, node2)]).next()
        except:
            return 0, 0, 0

    def get_resource_allocation_index(self, node1, node2):
        """
        Parameters
        ----------
        node1:
        node2:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        return nx.resource_allocation_index(self._graph, [(node1, node2)]).next()

    def write_graph(self, output_path):
        """
        Parameters
        ----------
        output_path:

        Returns
        -------
        NxGraph: Graph object

        Examples
        --------
        >>>
        """
        nx.write_edgelist(self._graph, output_path, delimiter=',', data=False)

    def draw_graph(self,temp_vertices):
        """
       Parameters
       ----------
       None

       Returns
       -------
       Plots Graph

       Examples
       --------
       >>>
       """
        all_labels = {}
        # temp_vertices = self._graph.vertices[start:end]
        for i in temp_vertices:
            lab = "Real"
            if((self.get_node_label(i)) == '0'):
                lab = "Fake"


            all_labels[i]=str(i)+" is "+lab
        #     all_labels[i] = i
        G = self._graph
        nx.draw(G,labels=all_labels,with_labels=True)
        # nx.draw(G)
        plt.show(G)