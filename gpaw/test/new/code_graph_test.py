import pytest
from gpaw.doctools.codegraph import main


@pytest.mark.ci
@pytest.mark.serial
def test_code_graph(monkeypatch):
    monkeypatch.setattr('graphviz.Digraph.render',
                        lambda self, name, format: None)
    pytest.importorskip('graphviz')
    main()
