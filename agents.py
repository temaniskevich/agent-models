import numpy as np
import random
from settings import settings
from copy import deepcopy

class HistoryList:
    def __init__(self, values=None, histories=None, history_values=None):
        self.values = values if values is not None else []
        self.histories = histories if histories is not None else {
            'mbk': [], 'cb': [], 'deposit': [], 'credit': []}
        self.history_values = history_values if history_values is not None else {
            'mbk': [], 'cb': [], 'deposit': [], 'credit': []}

    def append(self, item):
        self.values.append(item)
        self.history_values[item.flow_type].append(item.volume)

    def extend(self, other):
        for item in other:
            self.values.append(item)
            self.history_values[item.flow_type].append(item.volume)

    def update_history_removed(self, item):
        self.history_values[item.flow_type].append(-item.volume)

    def update_history(self):
        for key in self.history_values.keys():
            if self.histories[key]:
                self.histories[key].append(
                    self.histories[key][-1] + sum(self.history_values[key]))
            else:
                self.histories[key].append(sum(self.history_values[key]))

            self.history_values[key] = []

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        for item in self.values:
            yield item

    def _string_representation(self):
        return f'{self.values}'

    def __repr__(self):
        return self._string_representation()

    def __str__(self):
        return self._string_representation()


class Flow:
    def __init__(self, flow_type, volume=None, rate = settings['cb_rate']):
        self.flow_type = flow_type
        self.volume = volume
        self.rate = rate

        # Депозит или Кредит
        if (flow_type == 'deposit' or 'credit') and volume is None:
            self.volume = round(np.random.uniform(settings[f'{flow_type}_volume_bound'][0],
                                                settings[f'{flow_type}_volume_bound'][1]))
            self.maturity = random.choice(
                settings[f'{flow_type}_maturity'])
            self.payment_period = random.choice(settings['payment_period'])
            self.days_to_pay = self.payment_period

        elif (flow_type == 'mbk' or 'cb'):
            self.maturity = random.choice(
                settings[f'{flow_type}_maturity'])
            self.payment_period = random.choice(settings['payment_period'])
            self.days_to_pay = self.payment_period

        else:
            raise ValueError(f'Unimplemented type of {flow_type} was given')

    def update_rate(self, delta):
        """
        Функция, которая назначает ставку потоку в зависимости от того,
        в какой банк попал данный поток для розничных кредитов и депозитов.
        На рынке МБК и в транзакциях с ЦБ действует ключевая ставка
        """
        if self.flow_type == 'deposit':
            self.rate -= delta
        elif self.flow_type == 'credit':
            self.rate += delta

        assert self.rate > 0.01


    def _string_representation(self):
        return f'{self.flow_type.title()} {round(self.rate * 100)}%'
       # return f'{self.flow_type.title()} размером {self.volume} рублей, {round(self.rate * 100)}%'
       # return f'{self.flow_type.title()} (тип: {self.flow_type}) сроком погашения {self.maturity} дней, размером {self.volume} рублей, по ставке {self.rate * 100}%, с периодом выплат {self.payment_period} дней'

    def __repr__(self):
        return self._string_representation()

    def __str__(self):
        return self._string_representation()


class Bank:
    def __init__(self, name):
        self.name = name
        self.cash = 0
        self.risk_tolerance = np.random.uniform(0,0.1)
        self.reserves_to_cb = 0

        self.deposits = HistoryList()
        self.credits = HistoryList()

        self.deposit_apps = []
        self.credit_apps = []

        self.cash_history = []
        self.delta_history = []
        self.reliability_history = []

        self.delta = 0
        self.fixed_costs = settings['bank_fixed_costs']
        self.operating_costs = settings['bank_operating_costs']

    def set_reliability(self):
        self.reliability = self.cash / 2e6
        self.reliability_history.append(self.reliability)

    def set_delta(self):
        if self.reliability >= 1:
            delta = 0.03
        elif self.reliability > 0.75:
            delta = 0.02
        elif self.reliability > 0.6:
            delta = 0.01
        else:
            delta = 0

        self.delta = delta
        self.delta_history.append(self.delta)

    def validate(self):
        # 1. Принять все депозиты, назначить им ставки
        for deposit in self.deposit_apps:
            deposit.update_rate(self.delta)
            self.deposits.append(deposit)
            self.reserves_to_cb += settings['cb_reserve_rate'] * deposit.volume

        # Принять кредит, если есть кэш на него
        free_cash = deepcopy(self.cash)
        # Массив для подвержденных заявок (тк он отличается от массива всех заявок)
        approved = []
        for credit in self.credit_apps:
            credit.update_rate(self.delta)
            if free_cash >= credit.volume:  # почему-то не работает
                self.credits.append(credit)
                approved.append(credit)
                free_cash -= credit.volume
            elif free_cash < credit.volume:
                prob = np.random.uniform(0, 0.3)  # генерируем случайную "рисковость" актива
                if prob <= self.risk_tolerance:  # если его рисковость устраивает банк - выдает кредит
                    self.credits.append(credit)
                    approved.append(credit)
                    free_cash -= credit.volume

        # 2. Посчитать текущие обязательства

        # Кредиты, которые выдаст по заявкам
        credits_to_give = sum(credit.volume for credit in approved)

        # Проценты по депозитам, которые нужно выплатить сегодня
        deposit_coupon_to_return = sum(
           deposit.volume * deposit.rate/(360/deposit.payment_period) for deposit in self.deposits if deposit.days_to_pay == 0)

        # Разделить депозиты на те, что возвращаем сегодня и на те, что пока что остаются
        self.deposits_to_return = []
        deposits_to_stay = []

        for deposit in self.deposits:
            # Если депозит истекает сегодня
            if deposit.maturity == 0:
                self.deposits_to_return.append(deposit)
                self.deposits.update_history_removed(deposit)
            # Если депозит не истекает сегодня
            else:
                # Уменьшаем срок до истечения на 1.
                deposit.maturity -= 1
                # Уменьшаем количество дней до следующей выплаты, а если сейчас ноль, то ставим новый период
                deposit.days_to_pay = deposit.payment_period if deposit.days_to_pay == 0 else deposit.days_to_pay - 1
                deposits_to_stay.append(deposit)


        self.deposits = HistoryList(
                values=deposits_to_stay, histories=self.deposits.histories, history_values=self.deposits.history_values)

        deposits_volume_to_return = sum(deposit.volume for deposit in self.deposits_to_return)

        # Сумма всех обязательств банка на сегодня
        self.current_obligations = (deposits_volume_to_return + deposit_coupon_to_return + credits_to_give) \
                                    #+ self.reserves_to_cb)



        # 3. Посчитать текущие притоки
        # Депозиты, которые поступят на счет банка
        deposits_to_get = sum(deposit.volume for deposit in self.deposit_apps)

        # Проценты по кредитам, которые должны прийти сегодня
        credit_coupon_to_get = sum(
           credit.volume * credit.rate/(360/credit.payment_period) for credit in self.credits if credit.days_to_pay == 0)

        # Разделяем кредиты на те, которые нам будут возвращать сегодня и те, которые останутся
        self.credits_to_get = []
        credits_to_stay = []
        for credit in self.credits:
            # Если истекает сегодня
            if credit.maturity == 0:
                self.credits_to_get.append(credit)
                self.credits.update_history_removed(credit)
            # Если кредит не истекает сегодня
            else:
                credit.maturity -= 1
                credit.days_to_pay = credit.payment_period if credit.days_to_pay == 0 else credit.days_to_pay - 1
                credits_to_stay.append(credit)

        self.credits = HistoryList(
            values=credits_to_stay, histories=self.credits.histories, history_values=self.credits.history_values)

        # Сумма кредитов, которые банку вернут сегодня
        credits_volume_to_get = sum(credit.volume for credit in self.credits_to_get)

        # Сумма всех притоков денег банку за день
        self.current_inflows = credits_volume_to_get + credit_coupon_to_get + deposits_to_get






    def solve(self):

        # Пересчет издержек в зависимости от объема операционной деятельности

        self.cash += self.current_inflows
        self.cash -= self.current_obligations
        operating_costs = self.cash * self.operating_costs
        self.cash -= self.fixed_costs + operating_costs

        if self.cash >= 0:
            self.solved = True
        else:
            self.solved = False

        #assert self.solved == True

        if not self.solved:  # если не может покрыть долги сам - идет на МБК
            self.loan_amount = - self.cash + np.random.randint(1e5, 1e6)

    def restart(self):
        # Обновим историю в кредитах и депозитах
        self.deposits.update_history()
        self.credits.update_history()

        # Обнуляем состояния переменных
        self.deposit_apps = []
        self.credit_apps = []

        #assert self.solved, 'Bank must be solved at the end of day'
        self.current_obligations = 0
        self.current_inflows = 0
        self.reserves_to_cb = 0

        #assert self.loan_amount == 0, f'Loan amount must be zero during clearing. Current loan amount is {self.loan_amount}'

        self.deposits_to_return = []
        self.credits_to_get = []

        self.cash_history.append(self.cash)

        self.set_reliability()
        self.set_delta()




    def get_cash(self):
        print(f"Net Assets of bank '{self.name}' = {self.cash}")

    def _string_representation(self):
        return self.name

    def __repr__(self):
        return self._string_representation()

    def __str__(self):
        return self._string_representation()


class Market:

    def __init__(self, settings_instance):
        self.settings_instance = settings_instance
        self.settings = self.settings_instance.__dict__

        self.settings_instance.bank_params = list(zip(
            self.settings['simulation_starting_cash'],
            self.settings['simulation_starting_deposits'],
            self.settings['simulation_starting_credits'],

        ))
        # Инициализируем банки и цб
        self.banks = [Bank(i, settings_instance)
                      for i in range(self.settings['banks_number'])]

        all_cash = np.array([bank.cash for bank in self.banks])
        self.init_cash_proportions = all_cash / np.sum(all_cash)

        self.n_deposits = list(np.round(self.init_cash_proportions *
                                        self.settings['deposits_number']).astype(int))
        self.n_credits = list(np.round(self.init_cash_proportions *
                                       self.settings['credits_number']).astype(int))

        self.n_business_credits = list(np.round(self.init_cash_proportions *
                                                self.settings['business_credits_number']).astype(int))

        # Инициализируем стартовые МБК-кредиты/депозиты
        for _ in range(np.random.choice(range(*self.settings['flow_mbk_init_number_bounds']))):
            depositor, creditor = np.random.choice(
                self.banks, 2, replace=False)

            flow_amount = np.random.uniform(
                *self.settings['flow_mbk_init_amount_bounds'])
            flow = Flow('flow', 'mbk', self.settings, flow_amount)

            depositor.credits.append(flow)
            creditor.deposits.append(deepcopy(flow))

        self.cb = CentralBank(settings_instance)
        self.solved_banks = []
        self.unsolved_banks = []
        self.global_step = 0

    def __len__(self):
        return self.settings['banks_number']

    def day_route(self, step):

        if self.settings['applications_distribution'] == 'proportional':
            for bank in self.banks:
                for _ in range(self.n_deposits[bank.id]):
                    bank.potential_deposits.append(
                        Flow('deposit', self.settings))

                for _ in range(self.n_credits[bank.id]):
                    bank.potential_credits.append(
                        Flow('credit', 'standard', self.settings))

                for _ in range(self.n_business_credits[bank.id]):
                    bank.potential_credits.append(
                        Flow('credit', 'business', self.settings))
        elif self.settings['applications_distribution'] == 'uniform':
            # Генерируем потенциальные заявки на депозиты/кредиты и распределяем их по банкам
            for _ in range(self.settings['deposits_number']):
                selected_bank = np.random.choice(
                    range(self.settings['banks_number']))
                self.banks[selected_bank].potential_deposits.append(
                    Flow('deposit', 'standard', self.settings))

            for _ in range(self.settings['credits_number']):
                selected_bank = np.random.choice(
                    range(self.settings['banks_number']))
                self.banks[selected_bank].potential_credits.append(
                    Flow('credit', 'standard', self.settings))

            for _ in range(self.settings['business_credits_number']):
                selected_bank = np.random.choice(
                    range(self.settings['banks_number']))
                self.banks[selected_bank].potential_credits.append(
                    Flow('credit', 'business', self.settings))
        else:
            raise ValueError('Unsopported type of applications distribution')

        # Проверяем, может ли каждый банк расплатиться и записываем в соответствующие списки
        [bank.check_solvency() for bank in self.banks]
        self.cb.check_solvency()

        [self.solved_banks.append(bank) if bank.solved else self.unsolved_banks.append(
            bank) for bank in self.banks]

        # Незакрывшиеся банки отправляем на МБК
        indeces = list(range(len(self.solved_banks)))

        for bank in self.unsolved_banks:
            # Для каждого банка пытаемся найти кредитора среди всех остальных
            # Один кредитор может выдать больше одного кредита
            np.random.shuffle(indeces)
            result = False
            for idx in indeces:
                result = self.banks_commitment(bank, self.solved_banks[idx])
                if result:
                    break
            # Если не закредитовался на МБК, санация ЦБ
            if not result:
                self.central_bank_rescue(self.cb, bank)
                bank.solved = True

        # Производим расчёты
        [bank.solve() for bank in self.banks]

        # Обнуляем состояние банка
        [bank.clear() for bank in self.banks]

        # Обнуляем состояние цб
        self.cb.clear()

        # Обнуляем состояние рынка
        self.clear()

    def clear(self):
        self.unsolved_banks = []
        self.solved_banks = []

    # Функция запуска симуляции
    def run(self, model_number=0, disable=True, shocks={}):
        for step in tqdm(range(self.settings['num_steps']), leave=False, disable=disable,
                         desc=f'Model №{model_number}'):
            if step in shocks:
                self.settings_instance.impose_shocks(shocks[step])

            self.settings_instance.refresh_countable()
            self.settings = self.settings_instance.__dict__
            self.day_route(step)

    # Функция посторной инициализации
    def re_init(self):
        self.__init__()

    # Попытка соглашения о кредите между банками
    def banks_commitment(self, borrower, creditor):

        creditor_ability = (
                                   creditor.cash - creditor.current_obligations) * self.settings['loan_to_give_share']

        # Проверяем, может ли потенциальный кредитор выдать кредит на всю сумму текущего долга
        if creditor_ability >= borrower.loan_amount:
            loan_flow = Flow('flow', 'mbk', self.settings,
                             borrower.loan_amount)

            # В случае успеха добавляем поток как кредит одной стороне и как депозит другой
            borrower.deposits.append(loan_flow)
            creditor.credits.append(deepcopy(loan_flow))
            # self.mbk.append()

            borrower.cash += borrower.loan_amount
            creditor.cash -= borrower.loan_amount

            # Помечаем банк как разрешённый и возвращаем маркер успеха
            borrower.solved = True
            borrower.loan_amount = 0

            return True
        # Если у кредитора недостаточно денег, чтобы покрыть весь объём, то он отдаёт максимум, который может
        else:

            loan_flow = Flow('flow', 'mbk', self.settings, creditor_ability)

            borrower.deposits.append(loan_flow)
            creditor.credits.append(deepcopy(loan_flow))

            borrower.cash += creditor_ability
            creditor.cash -= creditor_ability

            # Уменьшаем текущий долг и возвращаем маркер незакрытого банка
            borrower.loan_amount -= creditor_ability
            return False

    # Санация центральным банком
    def central_bank_rescue(self, central_bank, bank):
        # Центральный банк возмещает под небольшой процент долг банка
        loan_amount = bank.loan_amount
        loan_flow = Flow('flow', 'cb', self.settings, loan_amount)
        central_bank.credits.append(loan_flow)
        bank.deposits.append(deepcopy(loan_flow))

        bank.cash += loan_amount
        bank.loan_amount = 0
        return True

    def refresh_settings_instance(self):
        self.settings = self.settings_instance.__dict__

    def _string_representation(self):
        return f'Instance of market. Can generate deposits and credits. Contains banks.'

    def __str__(self):
        return self._string_representation()

    def __repr__(self):
        return self._string_representation()

class BankModel:
    def __init__(self, start_settings):

        #self.start_settings = start_settings
        self.settings = start_settings

        # В модели есть Банки и Центральный Банк
        self.banks = [Bank(f'Bank_{id}') for id in range(1, 21)]
        self.cb = Bank('Central_Bank')
        self.solved_banks = []
        self.unsolved_banks = []

        # Массивы всей модели для отрисовки графиков
        self.system_liquidity_history = []
        self.system_mbk_credits = []
        self.cb_credits_history = []
        self.hhi_history = []
        self.banks_dict = {bank.name: bank.cash_history for bank in self.banks}
        self.system_credits_history = []
        self.system_deposits_history = []


    def create_world(self, cb_cash=1e12):
        """
        Создает мир - распределяет ликвидность между банками и назначает стартовый кэш ЦБ
        :param cb_cash: Объем стартовых резервов Центробанка
        :return: Создает стартовое состояние мира
        """
        self.cb.cash += cb_cash
        self.cb.cash_history.append(self.cb.cash)

        # Добавляем банкам ликвидность согласно стартовым настройкам
        for bank in range(len(self.banks)):
            self.banks[bank].cash = np.array(settings["liquid_distribution"])[bank] * 1e10
            self.banks[bank].cash_history.append(self.banks[bank].cash)
            self.banks[bank].set_reliability()
            self.banks[bank].set_delta()

        self.system_liquidity_history.append(sum([bank.cash for bank in self.banks]))

    def hhi_index(self, system_liquidity):
        return sum([(((bank.cash/system_liquidity)*100)**2) for bank in self.banks])

    def run(self, n_steps):
        """
        Запускает симуляцию, которая длится n_steps дней
        :param n_steps: Количество шагов модели
        :return: Данные модели
        """
        self.system_deposits = []
        self.system_credits = []

        for _ in range(n_steps):

            # 4 - генерация Потоков
            deposit_supply = []
            credit_supply = []

            for _ in range(np.random.randint(self.settings["deposit_amount_bound"][0],
                                             self.settings["deposit_amount_bound"][1])):  # генерируем депозиты
                deposit = Flow('deposit')
                deposit_supply.append(deposit)  # создаем массив из сгенерированных депозитов

            self.system_deposits.append(deepcopy(deposit_supply))  # записываем сгенерированные депозиты в историю

            for _ in range(np.random.randint(self.settings["credit_amount_bound"][0], self.settings["credit_amount_bound"][1])):
                credit = Flow('credit')
                credit_supply.append(credit)

            self.system_credits.append(deepcopy(credit_supply))  # записываем сгенерированные кредиты в историю

            # 5 - Распределение заявок на депозиты и кредиты по банкам
            for deposit in deposit_supply:
                selected_bank = np.random.choice(range(20))
                self.banks[selected_bank].deposit_apps.append(deposit)  # отправляем ему заявку на депозит

            for credit in credit_supply:
                selected_bank = np.random.choice(range(20))  # выбираем рандомный банк
                self.banks[selected_bank].credit_apps.append(credit)  # записываем в потенциальные заявки рандомный кредит

            # 6 - Прием потоков и назначение ставок - включить в процесс валидации
            for bank in self.banks:
                bank.validate()
                self.cb.cash += bank.reserves_to_cb
                bank.solve()
                if bank.solved:
                    self.solved_banks.append(bank)
                else:
                    self.unsolved_banks.append(bank)

            self.cb.validate()
            self.cb.solve()

            # Незакрывшиеся банки отправляем на МБК
            indeces = list(range(len(self.solved_banks)))
            for bank in self.unsolved_banks:
                # Для каждого банка пытаемся найти кредитора среди всех остальных
                # Один кредитор может выдать больше одного кредита

                np.random.shuffle(indeces)
                result = False
                for idx in indeces:
                    result = banks_commitment(borrower=bank, creditor=self.solved_banks[idx])
                    if result:
                        break
                # Если не закредитовался на МБК, санация ЦБ
                if not result:
                    central_bank_rescue(self.cb, bank)
                    bank.solved = True

            [bank.restart() for bank in self.banks]
            self.solved_banks = []
            self.unsolved_banks = []

            system_liquidity = sum([bank.cash for bank in self.banks])
            self.system_liquidity_history.append(system_liquidity + self.cb.cash)

            mbk_count = 0
            cb_count = 0
            #deposit_count = 0
            for bank in self.banks:
                for credit in bank.credits.values:
                    if credit.flow_type == 'mbk':
                        mbk_count += 1

            for credit in self.cb.credits.values:
                if credit.flow_type == 'cb':
                    cb_count += 1

                #     elif credit.flow_type == 'credit':
                #         credit_count += 1
                # for deposit in bank.deposits.values:
                #     if deposit.flow_type == 'deposit':
                #         deposit_count += 1

            self.system_mbk_credits.append(mbk_count)
            self.system_credits_history.append(sum([credit.volume for credit in bank.credits for bank in self.banks]))
            self.system_deposits_history.append(sum([deposit.volume for deposit in bank.deposits for bank in self.banks]))
            self.cb.cash_history.append(self.cb.cash)
            self.cb_credits_history.append(cb_count)

            self.hhi_history.append(self.hhi_index(system_liquidity))
            self.banks_dict = {bank.name: bank.cash_history for bank in self.banks}



def banks_commitment(borrower, creditor):

    creditor_ability = creditor.cash

    # Проверяем, может ли потенциальный кредитор выдать кредит на всю сумму текущего долга
    if creditor_ability >= borrower.loan_amount:
        loan_flow = Flow('mbk', borrower.loan_amount)

        # В случае успеха добавляем поток как кредит одной стороне и как депозит другой
        borrower.deposits.append(loan_flow)
        creditor.credits.append(deepcopy(loan_flow))
        # self.mbk.append()

        borrower.cash += borrower.loan_amount
        creditor.cash -= borrower.loan_amount

        # Помечаем банк как разрешённый и возвращаем маркер успеха
        borrower.solved = True
        borrower.loan_amount = 0

        return True
    # Если у кредитора недостаточно денег, чтобы покрыть весь объём, то он отдаёт максимум, который может
    else:

        loan_flow = Flow('mbk', creditor_ability)

        borrower.deposits.append(loan_flow)
        creditor.credits.append(deepcopy(loan_flow))

        borrower.cash += creditor_ability
        creditor.cash -= creditor_ability

        # Уменьшаем текущий долг и возвращаем маркер незакрытого банка
        borrower.loan_amount -= creditor_ability
        return False

# Санация центральным банком
def central_bank_rescue(central_bank, bank):
    # Центральный банк возмещает под небольшой процент долг банка
    loan_amount = bank.loan_amount
    loan_flow = Flow('cb', loan_amount)
    central_bank.credits.append(loan_flow)
    bank.deposits.append(deepcopy(loan_flow))

    bank.cash += loan_amount
    central_bank.cash -= loan_amount
    bank.loan_amount = 0
    return True


