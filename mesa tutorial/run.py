from money_model import MoneyModel

# run model with 100 agents in a 10x10 grid for 100 steps
model = MoneyModel(100, 10, 10)
for i in range(100):
    model.step()