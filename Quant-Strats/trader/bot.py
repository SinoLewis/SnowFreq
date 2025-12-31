class TraderBot:
    def __init__(self, broker, cash=1e6):
        self.broker = broker
        self.cash = cash
        self.position = 0
        self.equity = []

    def step(self, price, target_pos):
        delta = target_pos - self.position
        exec_price, cost = self.broker.execute(price, delta)
        self.cash -= delta * exec_price + cost
        self.position += delta
        self.equity.append(self.cash + self.position * price)
