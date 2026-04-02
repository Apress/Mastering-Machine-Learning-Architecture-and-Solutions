import unittest  # Using the unittest framework for testing

# (Data quality check functions from previous examples would be here)

class TestDataQuality(unittest.TestCase):
    def test_missing_values(self):
        data = [1, 2, None, 4, None]
        self.assertEqual(check_missing_values(data), 2)

    def test_data_range(self):
        data = [10, 50, 150, 20, None]
        self.assertEqual(check_data_range(data, 0, 100), 1)

    def test_data_types(self):
        data = [1, 2, "abc", 4, None]
        self.assertEqual(check_data_types(data, int), 1)

    # Add more test cases as needed...

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) # To run the tests within the interactive environment
