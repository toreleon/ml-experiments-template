"""
Custom your own metric in here !
"""
from sklearn.metrics import f1_score


class Metric:
    def __init__(self, name, function):
        self.name = name
        self.function = function

    def __call__(self, output, target):
        return self.function(output, target)


class F1Score(Metric):
    def __init__(self, average="binary"):
        super().__init__(name="f1-score", function=self.f1_score)

        self.average = average

    def f1_score(self, output, target):
        pred = output.argmax(dim=1)
        return f1_score(target.cpu(), pred.cpu(), average=self.average)


class Accuracy(Metric):
    def __init__(self):
        super().__init__(name="accuracy", function=self.accuracy)

    def accuracy(self, output, target):
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return correct / len(target)

    def __call__(self, output, target):
        return self.function(output, target)
