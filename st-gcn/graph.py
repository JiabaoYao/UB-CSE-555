import numpy as np

"""
Graph class is used to construct the graph based on our 3D key points
The code is referenced using the original st-gcn git repo's graph.py script
with modificaiton based on our code
Reference
https://github.com/yysijie/st-gcn/blob/221c0e152054b8da593774c0d483e59befdb9061/net/utils/graph.py#L124
"""


class Graph():
    def __init__(self, hop_size, center=1):
        self.center = center  # center joint
        self.get_edge()

        self.hop_size = hop_size
        self.hop_dis = self.get_hop_distance(self.num_node, self.edge, hop_size=hop_size)

        self.A = self.get_adjacency()

    def __str__(self):
        return self.A

    def get_edge(self):
        self.num_node = 50
        self_link = [(i, i) for i in range(self.num_node)]

        # we have 50 joints
        neighbor_link = [
            (0,1),(1,3),(3,5),(0,2),
            (2,4),(1,7),(0,6),(7,6),
            (8,9),(9,10),(10,11),(11,12),
            (8,13),(13,14),(14,15),(15,16),
            (8,17),(17,18),(18,19),(19,20),
            (8,21),(21,22),(22,23),(23,24),
            (8,25),(25,26),(26,27),(27,28),
            (29,30),(30,31),(31,32),(32,33),
            (29,34),(34,35),(35,36),(36,37),
            (29,38),(38,39),(39,40),(40,41),
            (29,42),(42,43),(43,44),(44,45),
            (29,46),(46,47),(47,48),(48,49)

        ]

        self.edge = self_link + neighbor_link
        self.center = 1

    def get_hop_distance(self, num_node, edge, hop_size):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(hop_size + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(hop_size, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        DAD = np.dot(A, Dn)
        return DAD

    def get_adjacency(self):
        valid_hop = range(0, self.hop_size + 1, 1)

        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1

        normalize_adjacency = self.normalize_digraph(adjacency)

        A_root = np.zeros((self.num_node, self.num_node))
        A_centripetal = np.zeros((self.num_node, self.num_node))
        A_centrifugal = np.zeros((self.num_node, self.num_node))

        for i, j in self.edge:
            hop = self.hop_dis[j, i]
            if hop > self.hop_size:
                continue

            d_i = self.hop_dis[i, self.center]
            d_j = self.hop_dis[j, self.center]

            if d_i == d_j:
                A_root[j, i] = normalize_adjacency[j, i]
            elif d_j < d_i:
                A_centripetal[j, i] = normalize_adjacency[j, i]
            else:
                A_centrifugal[j, i] = normalize_adjacency[j, i]

        A = np.stack([A_root, A_centripetal, A_centrifugal])

        return A

    def get_adjacency_matrix(self):
        return self.A
