class Agent:
    def __init__(self, name):
        self.name = name

    def communicate(self, message):
        print(f"{self.name} received message: {message}")


# Simulate communication between agents
agent_a = Agent("Agent A")
agent_b = Agent("Agent B")

agent_a.communicate("Initiate cooperative behavior.")
agent_b.communicate("Acknowledged. Executing coordinated action.")
