import numpy as np

class HandOnlyGraph:
    """
    Hand‑only ST‑GCN graph (42 joints: 0‑20 left, 21‑41 right).

    Partitions returned by `get_adjacency_matrix()`:
        0. Identity / self‑loops                 (42×42)
        1. Left‑hand sub‑graph (wrist link)      (42×42)
        2. Right‑hand sub‑graph (wrist link)     (42×42)

    Shape: (3, 42, 42)  → drop‑in replacement for the original 3‑partition ST‑GCN loader.
    """

    def __init__(self):
        self.left = list(range(0, 21))
        self.right = list(range(21, 42))
        self.num_node = 42
        self.edge = self._build_edges()
        self.A = self._get_adjacency()

    # ---------- public ----------
    def get_adjacency_matrix(self):
        return self.A

    # ---------- private ----------
    def _build_edges(self):
        hand = [
            (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),          # index
            (0, 9), (9, 10), (10, 11), (11, 12),     # middle
            (0, 13), (13, 14), (14, 15), (15, 16),   # ring
            (0, 17), (17, 18), (18, 19), (19, 20)    # pinky
        ]
        left = hand
        right = [(i + 21, j + 21) for i, j in hand]
        # cross‑hand wrist link (0 ↔ 21)
        cross = [(0, 21)]
        return left + right + cross

    @staticmethod
    def _normalize_digraph(A):
        Dl = A.sum(axis=0)
        Dn = np.zeros_like(A)
        Dn[np.where(Dl > 0), np.where(Dl > 0)] = Dl[Dl > 0] ** (-1)
        return A @ Dn

    def _get_adjacency(self):
        A_left  = np.zeros((self.num_node, self.num_node))
        A_right = np.zeros((self.num_node, self.num_node))

        for i, j in self.edge:
            if i < 21 and j < 21:          # both left
                A_left[i, j] = A_left[j, i] = 1
            elif i >= 21 and j >= 21:      # both right
                A_right[i, j] = A_right[j, i] = 1
            else:                          # cross‑wrist link
                A_left[i, j] = A_left[j, i] = 1
                A_right[i, j] = A_right[j, i] = 1

        I   = np.eye(self.num_node, dtype=np.float32)
        Al  = self._normalize_digraph(A_left)
        Ar  = self._normalize_digraph(A_right)

        return np.stack([I, Al, Ar]).astype(np.float32)
    

# Based on ST-GCN: https://github.com/yysijie/st-gcn
# Original work by Sijie Yan, Yuanjun Xiong, Dahua Lin (CVPR 2018)
