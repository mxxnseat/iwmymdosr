def weighted_mse(k = 20):
    def fn(y_hat, y):
        weight = 1 + k * y

        return (weight*(y_hat-y)**2).mean()
    
    return fn
