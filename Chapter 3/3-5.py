import hpbandster.core.nameserver as ns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import HyperBand
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import make_classification
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)

# 1. Configuration Space
config_space = CS.ConfigurationSpace()
n_estimators = CSH.UniformIntegerHyperparameter("n_estimators", lower=10, upper=200, default_value=100)
max_depth = CSH.UniformIntegerHyperparameter("max_depth", lower=3, upper=10, default_value=5)
config_space.add(n_estimators)
config_space.add(max_depth)

# 2. Worker Class (Simplified - No Multiprocessing)
class MyWorker(Worker):  # Inherit from Worker
    def compute(self, config, budget, **kwargs):
        rf = RandomForestClassifier(n_estimators=config["n_estimators"],
                                   max_depth=config["max_depth"],
                                   random_state=42)
        score = cross_val_score(rf, X_train, y_train, cv=3, scoring='accuracy').mean()
        return ({'accuracy': score},)  # Must return a tuple

# 3. Sample Data
X, y = make_classification(n_samples=100, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Run ID
run_id = str(uuid.uuid4())

# 5. Initialize NameServer (Simplified - In-Process)
NS = ns.NameServer(host='127.0.0.1', port=None, run_id=run_id)
NS.start()  # Start the nameserver

# 6. Initialize and Run Worker (Simplified - No separate process)
w = MyWorker(nameserver='127.0.0.1', nameserver_port=NS.port, id='Worker_0', run_id=run_id)

# 7. Initialize Hyperband
hb = HyperBand(configspace=config_space,
               eta=3,
               min_budget=1,
               max_budget=81,
               nameserver='127.0.0.1',
               nameserver_port=NS.port,
               run_id=run_id)

# 8. Run Hyperband
hb.run(10)

# 9. Get Best Configuration
id, best_config = hb.get_incumbent()
print(f"Best Configuration: {best_config}")

# 10. Shutdown
hb.shutdown(shutdown_workers=True)
NS.shutdown()

# 11. Evaluate the best model
rf_best = RandomForestClassifier(n_estimators=best_config['n_estimators'],
                                max_depth=best_config['max_depth'],
                                random_state=42)
rf_best.fit(X_train, y_train)
test_accuracy = rf_best.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")
