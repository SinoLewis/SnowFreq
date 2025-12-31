class PaperBroker:
    def __init__(self, fee=0.0003):
        self.fee = fee

    def execute(self, price, size):
        return price, abs(size) * self.fee
