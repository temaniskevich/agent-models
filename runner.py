from agents import *
from settings import settings
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import pandas as pd

style.use('ggplot')
def simulation(steps):
    model = BankModel(settings)
    model.create_world()
    model.run(steps)
    return model.system_liquidity_history


model = BankModel(settings)
model.create_world(cb_cash=0)
model.run(50)

y = [cash_list[-1] for cash_list in model.banks_dict.values()]
x = model.banks_dict.keys()
data = {'Bank': x,
        'Cash': y}
data = pd.DataFrame(data)
data_sorted = data.sort_values(by='Cash', ascending=False)


#1 Ликвидность системы
fig1, axs = plt.subplots(1,2)
axs[0].plot(model.banks[2].cash_history, label='Банк 1')
axs[0].plot(model.banks[3].cash_history, label='Банк 2')
axs[0].set_title('Ликвидности банков')
axs[0].set_xlabel('Шаг модели')
axs[0].set_ylabel('Рубли')
axs[0].legend()
axs[1].plot(model.system_liquidity_history, color='g')
axs[1].set_title('Ликвидность сектора')
axs[1].set_xlabel('Шаг модели')

#2 Кредиты и депозиты в модели
plt.figure(2)
plt.plot(model.system_deposits_history, label='Депозиты')
plt.plot(model.system_credits_history, label='Кредиты')
plt.legend()
plt.ylabel('Рубли')
plt.xlabel('Шаг модели')
plt.title('Депозиты и кредиты в системе')

#3 Динамика рынка МБК
plt.figure(3)
plt.plot(model.system_mbk_credits, color='royalblue', ls='--')
plt.xlabel('Шаг модели')
plt.ylabel('Количество кредитов')
plt.title('Рынок МБК')

#4 Показатели ЦБ
fig1, axs = plt.subplots(1,2)
# # Создаем первое полотно
axs[1].plot(model.cb.cash_history)
axs[1].set_xlabel('Шаг модели')
axs[1].set_title('Резервы ЦБ')
axs[1].set_ylabel('Рубли')
axs[0].plot(model.cb_credits_history, color='royalblue', ls='dotted')
axs[0].set_ylabel('Кол-во кредитов')
axs[0].set_xlabel('Шаг модели')
axs[0].set_title('Кредиты ЦБ')


#5 Индекс ХХИ
plt.figure(5)
plt.plot(model.hhi_history, ls='--')
plt.xlabel('Шаг модели')
plt.title('Индекс Херфиндаля-Хиршмана')

#6 Распределение банков
plt.figure(6, figsize=(15,7))
sns.barplot(data = data_sorted, x = 'Cash', y = 'Bank', palette='magma', hue= 'Bank', legend=False)
plt.xticks(fontsize=8)
plt.xlabel('Банки')
plt.ylabel('Ликвидность')
plt.title('Распределение ликвидности банков')

# 7 Просто ликвидность системы
plt.figure(7)
plt.plot(model.system_liquidity_history, color='g')
plt.title('Ликвидность сектора')
plt.xlabel('Шаг модели')
plt.ylabel('Рубли')

# 8 Просто 3 банка
plt.figure(8)
plt.plot(model.banks[0].cash_history, label='Банк 1')
plt.plot(model.banks[10].cash_history, label='Банк 2')
plt.plot(model.banks[16].cash_history, label='Банк 3')
plt.title('Ликвидность отдельных банков')
plt.xlabel('Шаг модели')
plt.ylabel('Рубли')

# axs[1,0].plot(model.banks[2].reliability_history)
# axs[1,0].plot(model.banks[3].reliability_history)
# axs[1,1].plot(model.system_mbk_credits)
plt.show()

#plt.plot(model.system_deposits_history)
#plt.plot(model.system_credits_history)

