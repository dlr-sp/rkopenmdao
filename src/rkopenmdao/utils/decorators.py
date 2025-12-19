class indexed_static:
    def __init__(self, func):
        self.func = func
        self.cache_name = f"_{func.__name__}_cache"

    def __get__(self, obj, cls):
        if not hasattr(cls, self.cache_name):
            setattr(cls, self.cache_name, self.func())
        return getattr(cls, self.cache_name)
