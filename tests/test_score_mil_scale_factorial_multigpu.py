import unittest

from score_mil_scale_factorial_multigpu import partition_task_pairs


class TestTaskPartition(unittest.TestCase):
    def test_each_pseudo_seed_pair_is_assigned_once_and_kept_intact(self):
        pseudo_unknowns = ("ASC-US", "LISL")
        seeds = (11, 12, 13)

        assignments = partition_task_pairs(pseudo_unknowns, seeds, ("cuda:0", "cuda:1"))

        self.assertEqual(len(assignments), 2)
        assigned_pairs = [pair for worker in assignments for pair in worker["tasks"]]
        expected_pairs = [(pseudo, seed) for pseudo in pseudo_unknowns for seed in seeds]
        self.assertCountEqual(assigned_pairs, expected_pairs)
        self.assertEqual(len(assignments[0]["tasks"]), 3)
        self.assertEqual(len(assignments[1]["tasks"]), 3)

    def test_duplicate_device_assignment_is_rejected(self):
        with self.assertRaises(ValueError):
            partition_task_pairs(("ASC-US",), (11, 12), ("cuda:0", "cuda:0"))

    def test_assignment_is_deterministic(self):
        arguments = (("ASC-US", "LISL"), (11, 12, 13), ("cuda:0", "cuda:1"))
        self.assertEqual(partition_task_pairs(*arguments), partition_task_pairs(*arguments))


if __name__ == "__main__":
    unittest.main()
