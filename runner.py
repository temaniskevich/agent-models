import random
from settings import settings
import numpy as np
from agents import Bank, Flow, HistoryList

# 1 - создание 20 банков

banks_list = [Bank(f'Bank_{i}') for i in range(1, 21)]

# 2 - создание ликвидности у банков согласно распределению 'real'
for bank in range(len(banks_list)):
    banks_list[bank].cash = np.array(settings["liquid_distribution"])[bank] * 10e6
    banks_list[bank].cash_history.append(banks_list[bank].cash)
    banks_list[bank].set_reliability()
    banks_list[bank].set_delta()

def run(n_steps, day=0):
    system_deposits = []
    system_credits = []
    day += 1
    for _ in range(n_steps):
        # 4 - генерация Потоков
        deposit_supply = []
        credit_supply = []

        for _ in range(np.random.randint(settings["deposit_amount_bound"][0],
                                         settings["deposit_amount_bound"][1])):  # генерируем депозиты
            deposit = Flow('deposit')
            deposit_supply.append(deposit)  # создаем массив из сгенерированных депозитов

        system_deposits.append(deposit_supply)  # записываем сгенерированные депозиты в историю

        for _ in range(np.random.randint(settings["credit_amount_bound"][0], settings["credit_amount_bound"][1])):
            credit = Flow('credit')
            credit_supply.append(credit)

        system_credits.append(credit_supply)  # записываем сгенерированные кредиты в историю

        # 5 - Распределение заявок на депозиты и кредиты по банкам
        for _ in range(len(deposit_supply)):
            selected_bank = np.random.choice(range(20))  # выбираем рандомный банк
            banks_list[selected_bank].deposit_apps.append(
                random.choice(deposit_supply))  # отправляем ему заявку на депозит

        for _ in range(len(credit_supply)):
            selected_bank = np.random.choice(range(20))  # выбираем рандомный банк
            banks_list[selected_bank].credit_apps.append(
                random.choice(credit_supply))  # записываем в потенциальные заявки рандомный кредит

        # 6 - Прием потоков и назначение ставок - включить в процесс валидации
        for bank in banks_list:
            bank.validate()
            bank.solve()
            bank.restart()



    # 8 - Валидация системы

    # 9 - сохранение истории ликвидности системы и по банкам отдельно в файлы
    # 10 - отрисовка графиков


run(n_steps=10)

print(banks_list[1].__dict__)
print('---------------------------------------------------------------------------------------------------------------')
print(f'Кол-во заявок на депозиты: {len(banks_list[1].deposit_apps)}, кол-во депозитов: {len(banks_list[1].deposits)}')
print(f'Кол-во заявок на кредиты: {len(banks_list[1].credit_apps)}, кол-во кредитов: {len(banks_list[1].credits)}')
print(banks_list[1].cash_history)