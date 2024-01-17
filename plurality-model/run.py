from model import ElectionSystem

# winner should be 7
opinions1 = [
    (0, 3),
    (1, 3),
    (1, 4),
    (2, 4),
    (3, 2),
    (3, 1),
    (4, 1),
    (0, 4),
    (4, 0)
]

# winner should be 8
opinions2 = [
    (0, 3),
    (1, 3),
    (1, 4),
    (3, 0),
    (3, 2),
    (3, 1),
    (4, 1),
    (0, 4),
    (4, 0)
]

# ElectionSystem(num_voters, num_candidates, width,  height)
model = ElectionSystem(7, 2, x_max=5, y_max=5, opinions=opinions2)
for i in range(5):
    model.step()

# server.launch()