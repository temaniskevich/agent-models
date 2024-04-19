from agents import *
from settings import settings
import matplotlib.pyplot as plt

model = BankModel(settings)
model.create_world()
model.run(400)
print(model.cb.__dict__)
print(model.banks[2].credits)
#print(model.banks[7].deposits)

plt.plot(model.banks[2].cash_history)
plt.plot(model.banks[7].cash_history)
plt.plot(model.banks[15].cash_history)
#plt.plot(model.cb.cash_history)
#plt.plot(model.banks[10].cash_history)
#plt.plot(model.banks[11].cash_history)
plt.show()
