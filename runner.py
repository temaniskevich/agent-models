from agents import *
from settings import settings
import matplotlib.pyplot as plt

def simulation(steps):
    model = BankModel(settings)
    model.create_world()
    model.run(steps)
    return model.system_liquidity_history


model = BankModel(settings)
model.create_world()
model.run(800)
#print(model.cb.__dict__)
#print(model.banks[2].credits)
#plt.plot(model.banks[2].cash_history)
#plt.plot(model.banks[7].cash_history)
#plt.plot(model.banks[15].cash_history)
#plt.plot(model.cb.cash_history)
#plt.plot(model.system_liquidity_history)
#plt.plot(model.banks[10].cash_history)

fig1, ax1 = plt.subplots()
# Создаем первое полотно
ax1.plot(model.banks[2].cash_history)
ax1.plot(model.banks[0].cash_history)
ax1.set_title('Ликвидности 2-х банков')
# Создаем второе полотно
fig2, ax2 = plt.subplots()
ax2.plot(model.system_liquidity_history, color = 'green')
ax2.set_title('Ликвидность сектора')
plt.show()



