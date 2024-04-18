from agents import *
from settings import settings

model = BankModel(settings)
model.create_world()
model.run(100)
print(model.cb.__dict__)

