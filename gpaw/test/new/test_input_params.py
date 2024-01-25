from gpaw.new.input_parameters import parameter_functions


def test_order():
    assert list(parameter_functions) == sorted(parameter_functions)
