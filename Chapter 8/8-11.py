class TestMLPipeline(unittest.TestCase):

    def setUp(self):
        """Set up the objects for testing."""
        self.loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.trainer = ModelTrainer()
        # Load data once for all tests.
        self.X, self.y = self.loader.load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)

    def test_data_loader(self):
        """Test that data loader returns correct shapes."""
        self.assertEqual(self.X.shape[0], 150)
        self.assertEqual(self.X.shape[1], 4)

    def test_preprocessor(self):
        """Test that preprocessor correctly scales the data."""
        X_train_scaled = self.preprocessor.fit_transform(self.X_train)
        # Mean of scaled training data should be close to 0.
        self.assertTrue(np.allclose(np.mean(X_train_scaled, axis=0), np.zeros(4), atol=1e-1))

    def test_model_trainer(self):
        """Test that the model can be trained and achieves reasonable accuracy."""
        X_train_scaled = self.preprocessor.fit_transform(self.X_train)
        X_test_scaled = self.preprocessor.transform(self.X_test)
        model = self.trainer.train(X_train_scaled, self.y_train)
        accuracy = self.trainer.evaluate(X_test_scaled, self.y_test)
        self.assertGreaterEqual(accuracy, 0.7)


# Running the unit tests
def run_unit_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMLPipeline)
    unittest.TextTestRunner(verbosity=2).run(suite)


# Execute unit tests
run_unit_tests()
