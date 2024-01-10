from model import ElectionSystem

# ElectionSystem(num_voters, num_candidates, width,  height)
model = ElectionSystem(self, 100, 4, 100, 100)
for i in range(100):
    model.step()

# server.launch()