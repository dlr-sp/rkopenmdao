class indexed_static:
    """
    Descriptor that turns a function into a lazily-evaluated, cached
    class-level attribute.

    The wrapped function is called only once (on first access), and its
    return value is stored on the class. Subsequent accesses return the
    cached value instead of calling the function again.

    This behaves similarly to a static property with memoization.
    """

    def __init__(self, func):
        self.func = func
        # Name of the attribute used to store the cached value on the class
        self.cache_name = f"_{func.__name__}_cache"

    def __get__(self, obj, cls):
        # If the cached value does not yet exist on the class,
        # compute it and store it
        if not hasattr(cls, self.cache_name):
            setattr(cls, self.cache_name, self.func())

        # Return the cached value
        return getattr(cls, self.cache_name)
