import unittest

from pymhlib.permutation_solution import PermutationSolution, cycle_crossover, partially_matched_crossover


class TestSolution(PermutationSolution):
    """Solution to a QAP instance.

    Attributes
        - inst: associated QAPInstance
        - x: integer vector representing a permutation
    """

    def copy(self):
        sol = TestSolution(len(self.x), init=False)
        sol.copy_from(self)
        return sol

    def copy_from(self, other: 'TestSolution'):
        super().copy_from(other)

    def calc_objective(self):
        return 0

    def change(self, values):
        self.x = values

    def construct(self, par, result):
        """Scheduler method that constructs a new solution.

        Here we just call initialize.
        """
        del result
        self.initialize(par)


class CycleCrossoverTestCase(unittest.TestCase):

    def test_no_change(self):
        a = TestSolution(7)
        b = TestSolution(7)

        a.change([1, 4, 6, 2, 0, 3, 5])
        b.change([6, 1, 4, 2, 5, 0, 3])

        cycle_crossover(a, b)

        a_new = [1, 4, 6, 2, 0, 3, 5]
        b_new = [6, 1, 4, 2, 5, 0, 3]

        for i in range(0, 7):
            self.assertEqual(a.x[i], a_new[i])
            self.assertEqual(b.x[i], b_new[i])

    def test_change(self):
        a = TestSolution(7)
        b = TestSolution(7)

        a.change([1, 4, 6, 2, 0, 3, 5])
        b.change([6, 1, 4, 0, 2, 5, 3])

        cycle_crossover(a, b)

        a_new = [1, 4, 6, 0, 2, 3, 5]
        b_new = [6, 1, 4, 2, 0, 5, 3]

        for i in range(0, 7):
            self.assertEqual(a.x[i], a_new[i])
            self.assertEqual(b.x[i], b_new[i])


class PartialMatchedCrossoverTestCast(unittest.TestCase):

    def test_general(self):
        a = TestSolution(10)
        b = TestSolution(10)

        a.change([8, 4, 7, 3, 6, 2, 5, 1, 9, 0])
        b.change([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        child_a = partially_matched_crossover(a, b, range(3, 8))

        expect_a = [0, 7, 4, 3, 6, 2, 5, 1, 8, 9]

        for i in range(0, 9):
            self.assertEqual(child_a.x[i], expect_a[i])


if __name__ == '__main__':
    unittest.main()
