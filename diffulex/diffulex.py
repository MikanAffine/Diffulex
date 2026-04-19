from diffulex.engine.engine import DiffulexEngine


class Diffulex:
    def __new__(cls, model, **kwargs):
        return DiffulexEngine(model, **kwargs)
