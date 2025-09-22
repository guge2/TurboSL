models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(name, config):
    model = models[name](config)
    return model


from . import neus, geometry, albedo, ambient, neus_wz_camproblur, blur_wz_recenter
