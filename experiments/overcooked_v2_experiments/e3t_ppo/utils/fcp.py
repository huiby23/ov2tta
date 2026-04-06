class FCPWrapperPolicy:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError("E3T-PPO does not support FCP populations.")
