from functools import wraps
import sys

def colored(msg, color):
    end = "\033[0m"
    if color == "ok":
        pre = "\033[92m"
    elif color == "fail":
        pre = "\033[91m"
    else:
        raise ValueError("Color option not recognized")
    return pre + msg + end

def test_method(f, completions):
    @wraps(f)
    def wrapped(*args, **kwargs):
        msg = f"Running {f.__name__}".ljust(50) + "..."
        print(msg, end="")
        sys.stdout.flush()
        try:
            res = f(*args, **kwargs)
            if res is not None and res.lower() == "skipped":
                print("SKIPPED")
            else:
                print(colored("success", "ok"))
            for compl in completions:
                compl()
            return res
        except Exception as e:
            print(colored("failure", "fail"))
            for compl in completions:
                compl()
            raise e
    return wrapped

def iter_test_method(f, completions, it, maxit):
    @wraps(f)
    def wrapped(*args, **kwargs):
        max_digits = len(str(maxit))
        msg = f"Running {f.__name__}".ljust(50) + "Iteration: {}/{}".format(str(it).zfill(max_digits), maxit) + "..."
        if it == maxit:
            print(msg, end="")
        else:
            print(msg, end="\r")
        sys.stdout.flush()
        try:
            res = f(*args, **kwargs)
            if res is not None and res.lower() == "skipped":
                print(msg + "SKIPPED")
            elif it == maxit:
                print(colored("success", "ok"))
            for compl in completions:
                compl()
            return res
        except Exception as e:
            print(msg, end="")
            print(colored("failure", "fail"))
            for compl in completions:
                compl()
            raise e
    return wrapped



class BaseTester:
    def __init__(self):
        pass
    def run_tests(self, number=None, multi=False, its=0):
        cleanups = [m for m in dir(self) if callable(getattr(self, m)) and m.startswith("cleanup_")]
        my_test_methods = [m for m in dir(self) if callable(getattr(self, m)) and m.startswith("test_")]
        if multi:
            if number is not None:
                my_test_methods = [m for m in my_test_methods if number in m]
            for mname in my_test_methods:
                m = getattr(self, mname)
                for it in range(its):
                    res = iter_test_method(m, cleanups, it+1, its)()
                    if res == "skipped":
                        break
            exit()

        if number is not None:
            my_test_methods = [m for m in my_test_methods if "_" + str(number) + "_" in m]
        completions = [getattr(self, m) for m in cleanups]
        for mname in my_test_methods:
            m = getattr(self, mname)
            test_method(m, completions)()
