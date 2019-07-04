from functools import wraps

def colored(msg, color):
    end = "\033[0m"
    if color == "ok":
        pre = "\033[92m"
    elif color == "fail":
        pre = "\033[91m"
    else:
        raise ValueError("Color option not recognized")
    return pre + msg + end

def test_method(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        msg = f"Running {f.__name__}".ljust(50) + "..."
        print(msg, end="")
        try:
            res = f(*args, **kwargs)
            print(colored("success", "ok"))
            return res
        except Exception as e:
            print(colored("failure", "fail"))
            raise e
    return wrapped



class BaseTester:
    def __init__(self):
        pass
    def run_tests(self):
        my_test_methods = [m for m in dir(self) if callable(getattr(self, m)) and m.startswith("test_")]
        for mname in my_test_methods:
            m = getattr(self, mname)
            test_method(m)()
