from importlib import import_module


def get_core_model():
    core = import_module("health_risk_model.core_model")
    return core

