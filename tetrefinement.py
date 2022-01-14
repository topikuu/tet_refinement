# Tetrahedral mesh (local) refinement
# Dependencies: Numpy (for matrices), Scipy (for sparse matrices)
# License: 3-clause BSD

# This code was initially made on top of scikit-fem which has its implications
# on e.g. point and tetrahedra data ordering and some naming schemes.

import numpy as np
from scipy.sparse import csc_matrix


def refine(p, t, marked_t=None, marked_e=None):
    """
    :param p: numpy.ndarray 3 x np of mesh node coordinates
    :param t: numpy.ndarray 4 x nt of mesh tetrahedron node indices
    :param marked_t: Indices of t which should be split
    :param marked_e: numpy.ndarray 2 x ne of edges which should be split
    :return: (new_p, new_t, new_t_to_old_t)
    """
    if marked_e is not None and marked_t is not None:
        raise RuntimeError("Only one of marked_t or marked_e can be defined.")
    if marked_t is not None:
        if isinstance(marked_t, list):
            marked_t = np.array(marked_t, dtype=np.int64)
        if marked_t.shape == (0,):
            return (p, t, np.arange(t.shape[1]))
    if marked_e is not None:
        if marked_e.shape == (0,):
            return (p, t, np.arange(t.shape[1]))

    # 1. Build new points that bisect the marked edges
    # 2. Classify tetras into groups with topologically equivalent edge split
    # patterns.
    # 3. Remove troublesome patterns by splitting the tetras further.
    # 4. Reorder point indices to match within class and split the tetras so
    # that the split edges are respected.
    # 5. Fix inconsistencies between shared bi-split faces.

    nt = t.shape[1]
    nv = p.shape[1]

    if marked_t is not None:
        (e0, e1) = _get_longest_edges(p, t, marked_t)
    else:
        e0 = marked_e[0, :]
        e1 = marked_e[1, :]
    # Preallocate p. Maximal count happens when all tetras are separate and all
    # edges are split in which case each 4 nodes are turned into 10 nodes.
    p = np.hstack((p, np.zeros((3, int(np.ceil(1.5 * nv))))))

    # Part 1. Add points. First the longest edges of selected marked tetras are
    # split. p is mutated.
    (edge_to_midpoint, nv) = _new_midpoints(p, nv, e0, e1)

    # Part 2. The columns of splits_per_vert_in_tets classify split topologies.
    # Furthermore, t_m reflect the order of counts in splits_per_vert_in_tets.
    # Note that due to non-unique ordering and chirality, non-uniqueness still
    # occurs and thus some of the element additions require further re-ordering
    # in adder functions.
    (t_m, marked, splits_per_vert_in_tets) =\
        _get_splits_per_vert_in_tets(t, nt, edge_to_midpoint)

    # Count new tetras and preallocate new tetrahedra array
    new_tet_count = 0
    for pattern_datum in _PatternData.all_splits:
        new_tet_count += pattern_datum.new_tet_count \
                         * np.sum(np.prod(splits_per_vert_in_tets == pattern_datum.pattern, axis=0))

    t = np.hstack((t, np.zeros((4, new_tet_count), dtype=np.int64)))
    new_to_old = np.full(t.shape[1], -1, dtype=np.int64)
    new_to_old[0:nt] = np.arange(nt)

    # Part 3. Search for problematic patterns. If such exist, split those tetras
    # by adding a center node.
    # "3-split Triple", "4-split Run" and "5-split" can each lead to topologies
    # that are impossible to fix without changing child tetras coming from
    # multiple parent tetras. Therefore these are split further by adding a
    # center node and thus splitting the original tetra into four new tetras
    # with some edges already split. These new tetras are always splittable
    # without similar issues.
    # 3-Triple turns into 2 x 2-pair + 2 x 0-split tetras,
    # 4-Run turns into 4 x 2-pair tetras and
    # 5-split turns into 2 x 2-pair + 2 x 3-loop tetras.
    tet_m_indices = np.array([], dtype=np.int64)
    for pattern_datum in _PatternData.difficult_splits:
        tet_m_indices = np.hstack(
            (tet_m_indices, (np.sum(splits_per_vert_in_tets == pattern_datum.pattern, axis=0) == 4).nonzero()[0]))

    t_i = marked[tet_m_indices]
    old_nt = nt
    (nv, nt) = _PatternData.ElementAdders.add_center_node(p, nv, t, nt, marked[tet_m_indices], t_m[:, tet_m_indices])
    if old_nt != nt:
        new_to_old[old_nt:nt] = new_to_old[np.tile(t_i, 3)]

    # New edges can never have splits so edge_to_midpoit can just be resized to
    # get empty references implicitly. At this point we are also done with
    # modifications to the new p.
    edge_to_midpoint.resize(nv, nv)
    p = p[:, :nv]

    # New tetras must still be split on the split edges so we have to
    # rebuild the tetra-to-split-pattern records.
    (t_m, marked, splits_per_vert_in_tets) = \
        _get_splits_per_vert_in_tets(t, nt, edge_to_midpoint)
    # Now all very problematic tetras have been converted into more easily split
    # ones and we can proceed with direct splits.

    # Part 4. Build the new tets one edge split pattern at a time. At this point
    # we don't care about inconsistencies at opposite facets but only
    # record alternative tet pairs which can fix those.
    alt_topos = _AlternativeTopologies(nt)

    for pattern_datum in _PatternData.simple_splits:
        pattern = pattern_datum.pattern
        adder_func = pattern_datum.splitter_func
        old_nt = nt
        index = (np.sum(splits_per_vert_in_tets == pattern, axis=0) == 4).nonzero()[0]
        t_i = marked[index]
        t_mi = t_m[:, index]
        nt = adder_func(t, old_nt, t_i, t_mi, edge_to_midpoint, alt_topos)

        if nt > old_nt:
            new_to_old[old_nt:nt] = new_to_old[np.tile(t_i, int((nt - old_nt) / t_i.shape[0]))]

    # Part 5. Fix inconsistent topologies:
    # 5.1. Sort bisplit facets lexicographically. That is, the point indices
    # of the parent facet define the order.
    # 5.2. Now bisplit facets contain element pairs and quartets with the same
    # first three elements (bisplit_facets[0:3, ..]). The pairs correspond
    # to bisplit facets which were on the surface of the mesh and these
    # can be safely ignored as the surface can not cause inconsistent
    # topology.
    # 5.3. Find the inconsistent facet quartets. These are the facet quartets
    # whose bisplit_facets[3:5, ...] are not the same. That is, two of the
    # facets have the last two elements in the opposite order than the other
    # two.
    # 5.4. Flip the edge on the first of the inconsistent facets, i.e.
    # overwrite tets with the alternate topology from alt_topos.alt_tets
    at = alt_topos

    # 5.1. Sort and remove the empty data
    order = np.lexsort(np.flip(at.bisplit_facets[:, :at.nbf], axis=0))
    at.bisplit_facets = at.bisplit_facets[:, order]

    # 5.2: Remove alternative tetras whose twice split face is on the surface.
    even_bfs = at.bisplit_facets[0:3, ::2]
    quartet_starts = (np.nonzero(np.sum(even_bfs[:, 0:-1] == even_bfs[:, 1:],
                                        axis=0) == 3)[0] * 2).reshape((-1, 1))
    good_bf_inds = np.hstack((quartet_starts, quartet_starts + 1,
                              quartet_starts + 2, quartet_starts + 3)).flatten()
    at.bisplit_facets = at.bisplit_facets[:, good_bf_inds]
    order = order[good_bf_inds]
    at.alt_tet_inds = at.alt_tet_inds[order]
    at.alt_tets = at.alt_tets[:, order]

    # 5.3: If bisplit_facets[4, 4*i] and bisplit_facets[4, 4*i + 2]
    # are equivalent, the facets match already. Otherwise they don't.
    quartets = np.arange(quartet_starts.shape[0]) * 4
    mismatches = at.bisplit_facets[4, quartets] != at.bisplit_facets[4, quartets + 2]
    quartets_to_flip = quartets[mismatches]

    # 5.4: Flip the tets:
    alt_tets_to_flip = np.hstack((quartets_to_flip, quartets_to_flip + 1))
    t[:, at.alt_tet_inds[alt_tets_to_flip]] = at.alt_tets[:, alt_tets_to_flip]

    return (p, t, new_to_old)


class _AlternativeTopologies:
    # These are used to fix the topology of elements whose splits can
    # cause geometrically coincident but partially disconnected
    # elements. These can occur when two edges of a facet are split as
    # the resulting 3 triangles are non-unique.

    # bisplit_faces contains the point indices of facets which have two
    # of their edges split. bisplit_facets[k, 0:3] are the original 3
    # points sorted to decreasing order, bisplit_facets[k, 3] is the
    # point with 3 edges to points in this facet and
    # bisplit_facets[k, 4] is the point with 4 edges in this facet.
    # alt_tet_inds[2k:2k+2] contains the indices of tets which form the
    # pair of tets which can be rotated in order to swap the facet
    # topology of bisplit facet k.
    # Overwriting t[:, alt_tet_inds[2k:2k+2]] = alt_topos[2k:2k+2]
    # results in rotating the topology of facet k.
    def __init__(self, nt):
        self.nbf = 0
        self.bisplit_facets = np.full((5, 2 * nt), -1, dtype=np.int64)
        self.alt_tet_inds = np.full(2 * nt, -1, dtype=np.int64)
        self.alt_tets = np.full((4, 2 * nt), -1, dtype=np.int64)


def _new_midpoints(p, nv, e0, e1):
    # Add points in between indices e0 and e1
    # Mutates p array.
    new_ps = .5 * (p[:, e0] + p[:, e1])
    n_new_ps = new_ps.shape[1]

    (new_p_locs, p_loc_index) = np.unique(new_ps, return_inverse=True, axis=1)
    n_new_p_locs = new_p_locs.shape[1]
    p[:, nv:(nv + n_new_p_locs)] = new_p_locs
    # +1 to values as zeros in sparse matrices are interpreted as missing
    # elements.
    data = nv + p_loc_index + 1
    data = np.hstack((data, data))
    coords = np.vstack((np.hstack((e0, e1)),
                        np.hstack((e1, e0))))
    [uniq_coords, data_index] = np.unique(coords, axis=1, return_index=True)
    edge_to_midpoint = csc_matrix((data[data_index], (uniq_coords[0, :], uniq_coords[1, :])), shape=(nv, nv))

    return (edge_to_midpoint, nv + n_new_p_locs)


def _count_splits_per_vertex(t_m, edge_to_midpoint):
    # t_m = vertex indices of marked tetras
    splits_per_vertex = np.zeros_like(t_m, dtype=np.int64)
    for r_i in range(4):
        for c_i in range(4):
            splits_per_vertex[r_i, :] += edge_to_midpoint[t_m[r_i, :], t_m[c_i, :]].A.flatten() != 0
    return splits_per_vertex

def _get_splits_per_vert_in_tets(t, nt, edge_to_midpoint):
    # Find tetras whose edges have been split
    split_edges_per_t = _get_split_edges_per_t(edge_to_midpoint, t, nt)
    marked = split_edges_per_t.nonzero()[0]

    # Count split edges per vertex and reorder vertices in tetras so that
    # the most split edges are connected to the last vertices.
    t_m = np.sort(t[:, marked], axis=0)
    splits_per_vert_in_tets = _count_splits_per_vertex(t_m, edge_to_midpoint)
    col_order = np.argsort(splits_per_vert_in_tets, axis=0)
    splits_per_vert_in_tets = np.take_along_axis(splits_per_vert_in_tets, col_order, axis=0)
    t_m = np.take_along_axis(t_m, col_order, axis=0)
    return (t_m, marked, splits_per_vert_in_tets)

def _get_split_edges_per_t(edge_to_midpoint, t, nt):
    return np.vstack((edge_to_midpoint[t[0, :nt], t[1, :nt]].A != 0,
                      edge_to_midpoint[t[0, :nt], t[2, :nt]].A != 0,
                      edge_to_midpoint[t[0, :nt], t[3, :nt]].A != 0,
                      edge_to_midpoint[t[1, :nt], t[2, :nt]].A != 0,
                      edge_to_midpoint[t[1, :nt], t[3, :nt]].A != 0,
                      edge_to_midpoint[t[2, :nt], t[3, :nt]].A != 0,)).sum(axis=0)

def _get_longest_edges(p, t, marked_t):
    # Get the end-point indices of the longest edge in t for each marked_t. If
    # two or more edges are of equal length, the edges are prioritized in the
    # descending order 01, 02, 03, 12, 13, 23

    t_m = t[:, marked_t]

    l01 = np.sqrt(np.sum((p[:, t_m[0, :]] - p[:, t_m[1, :]]) ** 2, axis=0))
    l12 = np.sqrt(np.sum((p[:, t_m[1, :]] - p[:, t_m[2, :]]) ** 2, axis=0))
    l02 = np.sqrt(np.sum((p[:, t_m[0, :]] - p[:, t_m[2, :]]) ** 2, axis=0))
    l03 = np.sqrt(np.sum((p[:, t_m[0, :]] - p[:, t_m[3, :]]) ** 2, axis=0))
    l13 = np.sqrt(np.sum((p[:, t_m[1, :]] - p[:, t_m[3, :]]) ** 2, axis=0))
    l23 = np.sqrt(np.sum((p[:, t_m[2, :]] - p[:, t_m[3, :]]) ** 2, axis=0))

    # Indices where (0, 1) is the longest etc.
    ix01 = ((l01 >= l02)
            * (l01 >= l03)
            * (l01 >= l12)
            * (l01 >= l13)
            * (l01 >= l23))
    ix02 = ((l02 > l01)
            * (l02 >= l03)
            * (l02 >= l12)
            * (l02 >= l13)
            * (l02 >= l23))
    ix03 = ((l03 > l01)
            * (l03 > l02)
            * (l03 >= l12)
            * (l03 >= l13)
            * (l03 >= l23))
    ix12 = ((l12 > l01)
            * (l12 > l02)
            * (l12 > l03)
            * (l12 >= l13)
            * (l12 >= l23))
    ix13 = ((l13 > l01)
            * (l13 > l02)
            * (l13 > l03)
            * (l13 > l12)
            * (l13 >= l23))
    ix23 = ((l23 > l01)
            * (l23 > l12)
            * (l23 > l02)
            * (l23 > l03)
            * (l23 > l13))

    # Collect edges
    e0_rows = 0 * (ix01 + ix02 + ix03) + 1 * (ix12 + ix13) + 2 * ix23
    e1_rows = 1 * ix01 + 2 * (ix02 + ix12) + 3 * (ix03 + ix13 + ix23)
    e0 = np.take_along_axis(t_m, indices=e0_rows.reshape(1, -1), axis=0).flatten()
    e1 = np.take_along_axis(t_m, indices=e1_rows.reshape(1, -1), axis=0).flatten()

    return (e0, e1)


class _PatternData():
    class PatternDatum():
        def __init__(self, pattern_array, new_tet_count, splitter_func):
            self.pattern = pattern_array
            self.new_tet_count = new_tet_count
            self.splitter_func = splitter_func

    class ElementAdders():
        @staticmethod
        def get_midpoints(e0, e1, edge_to_midpoint):
            # Returns the p indices at the center of e0 and e1 lists, and the
            # number of active points in the p array
            e_new = edge_to_midpoint[e0, e1].A.flatten()
            return e_new - 1

        @staticmethod
        def add_elements(t, nt, p_is, tet_mixes, t_i):
            n_class = t_i.shape[0]
            i0s = tet_mixes[:, 0]
            i1s = tet_mixes[:, 1]
            i2s = tet_mixes[:, 2]
            i3s = tet_mixes[:, 3]
            # Add the tets. First bunch goes to old elements, the remaining to
            # the end of list.
            t[:, t_i] = np.vstack((p_is[i0s[0]], p_is[i1s[0]], p_is[i2s[0]], p_is[i3s[0]]))
            for mix_i in range(1, tet_mixes.shape[0]):
                t[:, (nt + (mix_i - 1) * n_class):(nt + mix_i * n_class)] = \
                    np.vstack((p_is[i0s[mix_i]], p_is[i1s[mix_i]],
                               p_is[i2s[mix_i]], p_is[i3s[mix_i]]))

        @staticmethod
        def add_alternates(alt_topos, p_is, t_i, nt, alt_facet_orders,
                           alt_mixes, alt_relative_ind):
            n_class = t_i.shape[0]
            i0s = alt_mixes[:, 0]
            i1s = alt_mixes[:, 1]
            i2s = alt_mixes[:, 2]
            i3s = alt_mixes[:, 3]

            for (mix_i, rel_i) in enumerate(alt_relative_ind):
                nbf = alt_topos.nbf
                if rel_i == 0:
                    alt_topos.alt_tet_inds[nbf:nbf + n_class] = t_i
                else:
                    alt_topos.alt_tet_inds[nbf:nbf + n_class] = nt + np.arange(((rel_i - 1) * n_class),
                                                                               (rel_i * n_class))

                alt_topos.alt_tets[:, nbf:nbf + n_class] = \
                    np.vstack((p_is[i0s[mix_i]], p_is[i1s[mix_i]],
                               p_is[i2s[mix_i]], p_is[i3s[mix_i]]))
                f_i = mix_i // 2
                ordered_facets = np.vstack((np.sort(np.vstack([p_is[alt_facet_orders[f_i, 0]],
                                                               p_is[alt_facet_orders[f_i, 1]],
                                                               p_is[alt_facet_orders[f_i, 2]]]), axis=0),
                                            p_is[alt_facet_orders[f_i, 3]],
                                            p_is[alt_facet_orders[f_i, 4]]))
                alt_topos.bisplit_facets[:, nbf:nbf + n_class] = ordered_facets
                alt_topos.nbf = nbf + n_class

        @staticmethod
        def squish_indices(t_m, indices):
            # Remove the indices[i]:th element from column i in t_m.
            # Assumes indices.shape[1] == t_m.shape[1], np.min(indices) == 0
            # and np.max(indices) <= t_m.shape[0]
            keep_mask = np.ones_like(t_m, dtype='bool')
            keep_mask[(indices.flatten(), np.arange(0, t_m.shape[1]))] = False
            return t_m.T[keep_mask.T].reshape((-1, t_m.shape[0] - 1)).T

        @staticmethod
        def add_center_node(p, nv, t, nt, t_i, t_m):
            # Mutates: p, t
            # Returns: (new nv, new nt)
            # t_i = indices to t which correspond to the elements of
            # to be split.
            if t_i.size == 0:
                return (nv, nt)
            p_i0 = t_m[0, :]
            p_i1 = t_m[1, :]
            p_i2 = t_m[2, :]
            p_i3 = t_m[3, :]
            p_i4 = np.arange(nv, nv + t_i.shape[0])
            # Add median points of tetras
            p[:, p_i4] = np.mean(p[:, t_m], axis=1)

            mix = np.array([[0, 1, 2, 4],
                            [0, 1, 3, 4],
                            [0, 2, 3, 4],
                            [1, 2, 3, 4]])
            p_is = [p_i0, p_i1, p_i2, p_i3, p_i4]
            _PatternData.ElementAdders.add_elements(t, nt, p_is, mix, t_i)
            return (nv + t_i.shape[0],
                    nt + (mix.shape[0] - 1) * t_i.shape[0])

        @staticmethod
        def one_split(t, nt, t_i, t_m, edge_to_midpoint, alt_topos):
            if t_m.size == 0:
                return nt
            # Mutates: t, t_m, alt_topos
            # Edge splits per vert: [0, 0, 1, 1]
            # Swizzle vertex indices to match the assumed topology (see docs)
            p_i0 = t_m[3, :]
            p_i1 = t_m[2, :]
            p_i2 = t_m[1, :]
            p_i3 = t_m[0, :]
            p_i4 = _PatternData.ElementAdders.get_midpoints(p_i0, p_i1, edge_to_midpoint)
            p_is = [p_i0, p_i1, p_i2, p_i3, p_i4]

            # rows of mix refer to orderings of p_i*s for tetras
            mix = np.array([[0, 2, 3, 4],
                            [1, 2, 3, 4]])
            _PatternData.ElementAdders.add_elements(t, nt, p_is, mix, t_i)
            return nt + (mix.shape[0] - 1) * t_i.shape[0]

        @staticmethod
        def two_split_pair(t, nt, t_i, t_m, edge_to_midpoint, alt_topos):
            if t_m.size == 0:
                return nt
            # Edge splits per vert: [0, 1, 1, 2]
            p_i0 = t_m[3, :]
            p_i1 = t_m[2, :]
            p_i2 = t_m[1, :]
            p_i3 = t_m[0, :]
            p_i4 = _PatternData.ElementAdders.get_midpoints(p_i0, p_i1, edge_to_midpoint)
            p_i5 = _PatternData.ElementAdders.get_midpoints(p_i0, p_i2, edge_to_midpoint)
            p_is = [p_i0, p_i1, p_i2, p_i3, p_i4, p_i5]
            mix = np.array([[0, 3, 4, 5],
                            [1, 2, 3, 4],
                            [2, 3, 4, 5]])

            _PatternData.ElementAdders.add_elements(t, nt, p_is, mix, t_i)

            alt_mix = np.array([[1, 2, 3, 5],
                                [1, 3, 4, 5]])
            alt_replacement_indices = [1, 2]
            evil_facets = np.array([[0, 1, 2, 5, 4]])
            _PatternData.ElementAdders.add_alternates(alt_topos, p_is,
                                                     t_i, nt, evil_facets, alt_mix,
                                                     alt_replacement_indices)
            return nt + (mix.shape[0] - 1) * t_i.shape[0]

        @staticmethod
        def two_split_opposite(t, nt, t_i, t_m, edge_to_midpoint, alt_topos):
            if t_m.size == 0:
                return nt
            # Edge splits per vert: [1, 1, 1, 1]
            p_i0 = t_m[3, :]
            # Find the vertices connected to p_i0s by trying the remaining
            # three and then filtering for the remaining two vertices

            # Remove last row so that np.take_along_axis can be used
            # Edge splits per vert: [1, 1, 1], of which two form a pair and
            # third is connected to p_i013
            t_m = t_m[0:3, :]
            pair_row_indices = -np.ones((1, p_i0.shape[0]), dtype=np.int64)
            for row in range(3):
                matches = (edge_to_midpoint[p_i0, t_m[row, :]] != 0)
                pair_row_indices[matches] = row

            p_i1 = np.take_along_axis(t_m, pair_row_indices, axis=0)
            # Edge splits per vert: [1, 1], and they form a pair
            t_m = _PatternData.ElementAdders.squish_indices(t_m, pair_row_indices)

            p_i2 = t_m[0, :]
            p_i3 = t_m[1, :]
            p_i4 = _PatternData.ElementAdders.get_midpoints(p_i0, p_i1, edge_to_midpoint)
            p_i5 = _PatternData.ElementAdders.get_midpoints(p_i2, p_i3, edge_to_midpoint)
            p_is = [p_i0, p_i1, p_i2, p_i3, p_i4, p_i5]

            mix = np.array([[0, 2, 4, 5],
                            [0, 3, 4, 5],
                            [1, 2, 4, 5],
                            [1, 3, 4, 5]])
            _PatternData.ElementAdders.add_elements(t, nt, p_is, mix, t_i)
            return nt + (mix.shape[0] - 1) * t_i.shape[0]

        @staticmethod
        def three_split_loop(t, nt, t_i, t_m, edge_to_midpoint, alt_topos):
            if t_m.size == 0:
                return nt
            # Edge splits per vert: [0, 2, 2, 2]
            p_i0 = t_m[3, :]
            p_i1 = t_m[2, :]
            p_i2 = t_m[1, :]
            p_i3 = t_m[0, :]
            p_i4 = _PatternData.ElementAdders.get_midpoints(p_i0, p_i1, edge_to_midpoint)
            p_i5 = _PatternData.ElementAdders.get_midpoints(p_i0, p_i2, edge_to_midpoint)
            p_i6 = _PatternData.ElementAdders.get_midpoints(p_i1, p_i2, edge_to_midpoint)
            p_is = [p_i0, p_i1, p_i2, p_i3, p_i4, p_i5, p_i6]

            mix = np.array([[0, 3, 4, 5],
                            [1, 3, 4, 6],
                            [3, 4, 5, 6],
                            [2, 3, 5, 6]])
            _PatternData.ElementAdders.add_elements(t, nt, p_is, mix, t_i)
            return nt + (mix.shape[0] - 1) * t_i.shape[0]

        @staticmethod
        def three_split_chain(t, nt, t_i, t_m, edge_to_midpoint, alt_topos):
            if t_m.size == 0:
                return nt
            # Edge splits per vert: [1, 1, 2, 2]
            p_i2 = t_m[0, :]
            # Edge splits per vert: [1, 2, 2] and one of the last two is
            # connected to p_i0
            t_m = t_m[1:, :]
            pair_row_indices = -np.ones((1, p_i2.shape[0]), dtype=np.int64)
            for row in range(1, 3):
                matches = (edge_to_midpoint[p_i2, t_m[row, :]] != 0)
                pair_row_indices[matches] = row

            p_i0 = np.take_along_axis(t_m, pair_row_indices, axis=0)
            # Edge splits per vert: [1, 2] and the second element is connected
            # to p_i1
            t_m = _PatternData.ElementAdders.squish_indices(t_m, pair_row_indices)
            p_i3 = t_m[0, :]
            p_i1 = t_m[1, :]
            p_i4 = _PatternData.ElementAdders.get_midpoints(p_i0, p_i1, edge_to_midpoint)
            p_i5 = _PatternData.ElementAdders.get_midpoints(p_i0, p_i2, edge_to_midpoint)
            p_i6 = _PatternData.ElementAdders.get_midpoints(p_i1, p_i3, edge_to_midpoint)
            p_is = [p_i0, p_i1, p_i2, p_i3, p_i4, p_i5, p_i6]

            mix = np.array([[0, 3, 4, 5],
                            [3, 4, 5, 6],
                            [1, 2, 5, 6],
                            [1, 4, 5, 6],
                            [2, 3, 5, 6]])
            _PatternData.ElementAdders.add_elements(t, nt, p_is, mix, t_i)

            alt_mix = np.array([[0, 4, 5, 6],
                                [0, 3, 5, 6],
                                [1, 2, 4, 6],
                                [2, 4, 5, 6]])
            alt_replacement_indices = [0, 1, 2, 3]
            evil_facets = np.array([[0, 1, 3, 6, 4],
                                    [0, 1, 2, 4, 5]])
            _PatternData.ElementAdders.add_alternates(alt_topos, p_is,
                                                     t_i, nt, evil_facets, alt_mix,
                                                     alt_replacement_indices)
            return nt + (mix.shape[0] - 1) * t_i.shape[0]

        @staticmethod
        def four_split_dangler(t, nt, t_i, t_m, edge_to_midpoint, alt_topos):
            if t_m.size == 0:
                return nt
            # Edge splits per vert: [1, 2, 2, 3]
            p_i0 = t_m[3, :]
            p_i1 = t_m[2, :]
            p_i2 = t_m[1, :]
            p_i3 = t_m[0, :]
            p_i4 = _PatternData.ElementAdders.get_midpoints(p_i0, p_i1, edge_to_midpoint)
            p_i5 = _PatternData.ElementAdders.get_midpoints(p_i0, p_i2, edge_to_midpoint)
            p_i6 = _PatternData.ElementAdders.get_midpoints(p_i0, p_i3, edge_to_midpoint)
            p_i7 = _PatternData.ElementAdders.get_midpoints(p_i1, p_i2, edge_to_midpoint)
            p_is = [p_i0, p_i1, p_i2, p_i3, p_i4, p_i5, p_i6, p_i7]

            mix = np.array([[0, 4, 5, 6],
                            [4, 5, 6, 7],
                            [1, 4, 6, 7],
                            [1, 3, 6, 7],
                            [2, 3, 6, 7],
                            [2, 5, 6, 7]])
            _PatternData.ElementAdders.add_elements(t, nt, p_is, mix, t_i)

            alt_mix = np.array([[3, 4, 6, 7],
                                [1, 3, 4, 7],
                                [2, 3, 5, 7],
                                [3, 5, 6, 7]])
            alt_replacement_indices = [2, 3, 4, 5]
            evil_facets = np.array([[0, 1, 3, 4, 6],
                                    [0, 2, 3, 5, 6]])
            _PatternData.ElementAdders.add_alternates(alt_topos, p_is,
                                                     t_i, nt, evil_facets, alt_mix,
                                                     alt_replacement_indices)
            return nt + (mix.shape[0] - 1) * t_i.shape[0]

        @staticmethod
        def six_split(t, nt, t_i, t_m, edge_to_midpoint, alt_topos):
            if t_m.size == 0:
                return nt
            p_i0 = t_m[3, :]
            p_i1 = t_m[2, :]
            p_i2 = t_m[1, :]
            p_i3 = t_m[0, :]
            p_i4 = _PatternData.ElementAdders.get_midpoints(p_i0, p_i1, edge_to_midpoint)
            p_i5 = _PatternData.ElementAdders.get_midpoints(p_i0, p_i2, edge_to_midpoint)
            p_i6 = _PatternData.ElementAdders.get_midpoints(p_i0, p_i3, edge_to_midpoint)
            p_i7 = _PatternData.ElementAdders.get_midpoints(p_i1, p_i2, edge_to_midpoint)
            p_i8 = _PatternData.ElementAdders.get_midpoints(p_i1, p_i3, edge_to_midpoint)
            p_i9 = _PatternData.ElementAdders.get_midpoints(p_i2, p_i3, edge_to_midpoint)
            p_is = [p_i0, p_i1, p_i2, p_i3, p_i4,
                    p_i5, p_i6, p_i7, p_i8, p_i9]

            mix = np.array([[0, 4, 5, 6],
                            [3, 6, 8, 9],
                            [2, 5, 7, 9],
                            [1, 4, 7, 8],
                            [4, 9, 6, 8],
                            [4, 9, 6, 5],
                            [4, 9, 5, 7],
                            [4, 9, 7, 8]])
            _PatternData.ElementAdders.add_elements(t, nt, p_is, mix, t_i)
            return nt + (mix.shape[0] - 1) * t_i.shape[0]

    one_split = PatternDatum(np.array([[0], [0], [1], [1]]), 1, ElementAdders.one_split)
    two_split_pair = PatternDatum(np.array([[0], [1], [1], [2]]), 2, ElementAdders.two_split_pair)
    two_split_opposite = PatternDatum(np.array([[1], [1], [1], [1]]), 3, ElementAdders.two_split_opposite)
    three_split_triple = PatternDatum(np.array([[1], [1], [1], [3]]), 9, None)
    three_split_loop = PatternDatum(np.array([[0], [2], [2], [2]]), 3, ElementAdders.three_split_loop)
    three_split_chain = PatternDatum(np.array([[1], [1], [2], [2]]), 4, ElementAdders.three_split_chain)
    four_split_dangler = PatternDatum(np.array([[1], [2], [2], [3]]), 5, ElementAdders.four_split_dangler)
    four_split_run = PatternDatum(np.array([[2], [2], [2], [2]]), 11, None)
    five_split = PatternDatum(np.array([[2], [2], [3], [3]]), 13, None)
    six_split = PatternDatum(np.array([[3], [3], [3], [3]]), 7, ElementAdders.six_split)

    all_splits = [one_split, two_split_pair, two_split_opposite,
                  three_split_triple, three_split_loop, three_split_chain,
                  four_split_dangler, four_split_run, five_split, six_split]
    difficult_splits = [three_split_triple, four_split_run, five_split]
    simple_splits = [one_split, two_split_pair, two_split_opposite,
                     three_split_loop, three_split_chain, four_split_dangler,
                     six_split]
