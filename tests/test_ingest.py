from bs4 import BeautifulSoup

from ingest import build_page_metadata


def test_build_page_metadata_uses_slug_for_generic_plugin_titles():
    soup = BeautifulSoup("<html><body><h1>Pipeline</h1></body></html>", "html.parser")

    metadata = build_page_metadata(
        "https://plugins.jenkins.io/workflow-aggregator/",
        "jenkins_plugin",
        soup,
    )

    assert metadata["plugin_id"] == "workflow-aggregator"
    assert metadata["plugin_name"] == "Workflow Aggregator"
    assert "workflow-aggregator" in metadata["plugin_aliases"]
