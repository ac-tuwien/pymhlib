from unittest import TestCase, main
"""Basic functionality tests.

Note that several of the heuristics would need to run for a longer time in order to get a more reasonable solution,
nor are all the applied algorithms really meaningful for the considered problem.
"""

from pymhlib.demos.common import run_optimization, data_dir, add_general_arguments_and_parse_settings
from pymhlib.settings import get_settings_parser, settings, seed_random_generators
from pymhlib.demos.maxsat import MAXSATInstance, MAXSATSolution
from pymhlib.demos.tsp import TSPInstance, TSPSolution
from pymhlib.demos.graph_coloring import GCInstance, GCSolution
from pymhlib.demos.misp import MISPInstance, MISPSolution
from pymhlib.demos.mkp import MKPInstance, MKPSolution
from pymhlib.demos.qap import QAPInstance, QAPSolution
from pymhlib.demos.vertex_cover import VertexCoverInstance, VertexCoverSolution

parser = get_settings_parser()
add_general_arguments_and_parse_settings()


class TestAll(TestCase):

    def test_maxsat_gvns(self):
        seed_random_generators(42)
        settings.inst_file = data_dir + "maxsat-adv1.cnf"
        settings.alg = 'gvns'
        settings.mh_titer = 100
        solution = run_optimization('MAXSAT', MAXSATInstance, MAXSATSolution, embedded=True)
        self.assertEqual(solution.obj(), 769)

    def test_tsp_sa(self):
        seed_random_generators(42)
        settings.inst_file = data_dir + "xqf131.tsp"
        settings.alg = 'sa'
        settings.mh_titer = 50000
        solution = run_optimization('TSP', TSPInstance, TSPSolution, embedded=True)
        self.assertEqual(solution.obj(), 2592)

    def test_tsp_ssga(self):
        seed_random_generators(42)
        settings.inst_file = data_dir + "xqf131.tsp"
        settings.alg = 'ssga'
        settings.mh_titer = 500
        solution = run_optimization('TSP', TSPInstance, TSPSolution, embedded=True)
        self.assertEqual(solution.obj(), 1376)

    def test_graph_coloring_gvns(self):
        seed_random_generators(42)
        settings.inst_file = data_dir + "fpsol2.i.1.col"
        settings.alg = 'gvns'
        settings.mh_titer = 500
        solution = run_optimization('Graph Coloring', GCInstance, GCSolution, embedded=True)
        self.assertEqual(solution.obj(), 1634)

    def test_misp_pbig(self):
        seed_random_generators(42)
        settings.inst_file = data_dir + "frb40-19-1.mis"
        settings.alg = 'pbig'
        settings.mh_titer = 500
        solution = run_optimization('MISP', MISPInstance, MISPSolution, embedded=True)
        self.assertEqual(solution.obj(), 32)

    def test_mkp_gvns(self):
        seed_random_generators(42)
        settings.inst_file = data_dir + "mknapcb5-01.txt"
        settings.alg = 'gvns'
        settings.mh_titer = 70
        solution = run_optimization('MKP', MKPInstance, MKPSolution, embedded=True)
        self.assertEqual(solution.obj(), 55610)

    def test_qap_gvns(self):
        seed_random_generators(42)
        settings.inst_file = data_dir + 'bur26a.dat'
        settings.alg = 'gvns'
        settings.mh_titer = 1000
        solution = run_optimization('QAP', QAPInstance, QAPSolution, embedded=True)
        self.assertEqual(solution.obj(), 5426670)

    def test_vertex_cover_gvns(self):
        seed_random_generators(42)
        settings.inst_file = data_dir + "frb40-19-1.mis"
        settings.alg = 'gvns'
        settings.mh_titer = 100
        solution = run_optimization('Vertex Cover', VertexCoverInstance, VertexCoverSolution, embedded=True)
        self.assertEqual(solution.obj(), 726)

    def test_maxsat_alns(self):
        seed_random_generators(42)
        settings.inst_file = data_dir + "maxsat-adv1.cnf"
        settings.alg = 'alns'
        settings.mh_titer = 600
        solution = run_optimization('MAXSAT', MAXSATInstance, MAXSATSolution, embedded=True)
        self.assertEqual(solution.obj(), 727)

    def test_maxsat_par_alns(self):
        seed_random_generators(42)
        settings.inst_file = data_dir + "maxsat-adv1.cnf"
        settings.alg = 'par_alns'
        settings.mh_titer = 600
        solution = run_optimization('MAXSAT', MAXSATInstance, MAXSATSolution, embedded=True)
        # self.assertEqual(solution.obj(), 728)  # non-deterministic result


if __name__ == '__main__':
    main()
