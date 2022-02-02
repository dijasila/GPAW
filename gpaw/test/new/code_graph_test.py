import pytest
from gpaw.doctools.codegraph import make_figures


@pytest.mark.ci
@pytest.mark.serial
def test_code_graph():
    pytest.importorskip('graphviz')
    boxes = make_figures(render=False)
    assert boxes == 56
