from settings import settings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from agents import HistoryList, Flow, Bank, BankModel


# liquid_distr = [0.45, 0.14, 0.08, 0.06, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
# banks = {}
#
# for i in range(len(settings['liquid_distribution'])):
#     banks[i] = 10e6 * settings['liquid_distribution'][i]
# print(f'Bank Sector: {banks}')
#
# system_liquid = sum(banks.values())
#
# print(f'System Liquidity = {sum(banks.values())}')
#
# deposit_1 = np.random.randint(settings['deposit_volume_bound'][0], settings['deposit_volume_bound'][1])
# #print(f' DEPOSIT = {deposit_1}')
#
# i = 0
# deposits = []
# credits = []
# while i < settings['num_steps']:  # количество шагов модели
#     i += 1
#     deposit_supply = []
#     credit_supply = []
#     for _ in range(np.random.randint(settings["deposit_amount_bound"][0], settings["deposit_amount_bound"][1])):  #генерируем депозиты
#         deposit = np.random.randint(settings['deposit_volume_bound'][0], settings['deposit_volume_bound'][1])
#         deposit_supply.append(deposit)  # создаем массив из сгенерированных депозитов
#
#     deposits.append(deposit_supply)  # сохраняем пришедшие депозиты в массив
#
#     for _ in range(np.random.randint(settings["credit_amount_bound"][0], settings["credit_amount_bound"][1])):
#         credit = np.random.randint(settings['credit_volume_bound'][0], settings['credit_volume_bound'][1])
#         credit_supply.append(credit)
#     credits.append(credit_supply)
#
#     for _ in range(len(deposit_supply)):
#         banks[np.random.randint(0, 20)] += deposit_supply.pop()
#
#     for _ in range(len(credit_supply)):
#         banks[np.random.randint(0, 20)] -= credit_supply.pop()
#
# print(deposits[7])
# print(credits[7])
#
# print(sorted(banks.values()))
# print('---------------------------------------------------------------------------------')
#
# deposit_list = HistoryList()
# [deposit_list.append(Flow('deposit')) for _ in range(7)]
#
# test_flow1 = Flow('deposit')
# print(test_flow1.__dict__) # -> {'flow_type': 'deposit', 'amount': None, 'volume': 285040, 'maturity': 183, 'days_to_pay': 30}
# test_flow2 = Flow('credit')
# print(test_flow2.__dict__)
#
# deposit_list.append(test_flow1)
# print(deposit_list.history_values)
# print('---------------------------------------------------------------------------------')
# pv = Flow('deposit')
# print(pv.__dict__)
#
# fv = pv.volume * (1 + pv.rate / (360/pv.payment_period)) ** (pv.maturity / pv.payment_period)
# print(fv)
# print('---------------------------------------------------------------------------------')
# # Проверка работы дельты
# test_bank = Bank('test_bank')
# test_bank.cash = 3000000
# test_bank.set_reliability()
# test_bank.set_delta()
# deposit = Flow('deposit')
# deposit.volume = 100000
# test_bank.deposit_apps.append(deposit)
# test_bank.deposit_apps[0].update_rate(test_bank.delta)
# #print(test_bank.deposit_apps)

model = BankModel(settings)
model.create_world()
model.run(200)
#print(model.banks[1].credits.values[0].flow_type)
# print(model.banks[19].cash_history)
# print([sublist[-1] for sublist in model.banks_dict.values()])
y = [cash_list[-1] for cash_list in model.banks_dict.values()]
x = model.banks_dict.keys()
data = {'Bank': x,
        'Cash': y}
data = pd.DataFrame(data)
data_sorted = data.sort_values(by='Cash', ascending=False)

print(x)

print(y)
# plt.figure(figsize=(15,7))
# sns.barplot(data = data_sorted, x = data_sorted['Bank'], y = data_sorted['Cash'], hue = data_sorted['Bank'])
# plt.xticks(fontsize=8)
# plt.show()

print(data_sorted)

#print(model.banks[0].deposits)
print(model.cb.credits.history_values['cb'])
print(sum([credit.volume for credit in model.banks[0].credits]))