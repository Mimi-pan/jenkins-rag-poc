from langchain_core.documents import Document

from rag_core import (
    FALLBACK,
    build_response_instructions,
    evaluate_retrieval,
    detect_workflow_mode,
    has_question_support,
    should_force_fallback,
    strip_inline_sources,
    tokenize_text,
)


class StubVectorStore:
    def __init__(self, results):
        self._results = results

    def similarity_search_with_score(self, question, k=4):
        return self._results[:k]


def test_tokenize_text_keeps_plugin_style_terms():
    tokens = tokenize_text("Git plugin uses workflow-aggregator and kubernetes_plugin.")
    assert "workflow-aggregator" in tokens
    assert "kubernetes_plugin" in tokens


def test_has_question_support_accepts_supported_query():
    results = [
        (
            Document(
                page_content="The Git plugin provides Git support for Jenkins jobs and pipelines.",
                metadata={"title": "Git plugin", "plugin_name": "Git plugin"},
            ),
            0.42,
        ),
    ]
    assert has_question_support("What does the Git plugin provide?", results) is True


def test_has_question_support_rejects_unsupported_query():
    results = [
        (
            Document(
                page_content="Jenkins monitoring exposes health and metrics endpoints.",
                metadata={"title": "Monitoring"},
            ),
            0.33,
        ),
    ]
    assert has_question_support("How do I configure AWS Lambda with Terraform?", results) is False


def test_strip_inline_sources_removes_source_footer():
    answer = "Use a Jenkinsfile for pipelines.\n\nSources: https://www.jenkins.io/doc/book/pipeline/"
    assert strip_inline_sources(answer) == "Use a Jenkinsfile for pipelines."


def test_should_force_fallback_for_unsupported_patterns():
    answer = "Based on the provided context, there is no specific answer."
    assert should_force_fallback(answer) is True


def test_fallback_constant_is_stable():
    assert FALLBACK == "I could not find this in the Jenkins documentation."


def test_detect_workflow_mode_for_pipeline_questions():
    assert detect_workflow_mode("How do I set up a Jenkins pipeline?") == "pipeline"


def test_build_response_instructions_for_troubleshooting_questions():
    instructions = build_response_instructions("How do I debug a failed Jenkins build?")
    assert "numbered troubleshooting steps" in instructions.lower()


def test_evaluate_retrieval_marks_supported_results():
    vectorstore = StubVectorStore([
        (
            Document(
                page_content="The Git plugin provides Git support for Jenkins jobs and pipelines.",
                metadata={"source": "https://plugins.jenkins.io/git/", "plugin_name": "Git plugin"},
            ),
            0.42,
        ),
    ])

    decision = evaluate_retrieval("What does the Git plugin provide?", vectorstore)

    assert decision.supported is True
    assert decision.should_fallback is False
    assert decision.sources == ["https://plugins.jenkins.io/git/"]


def test_evaluate_retrieval_falls_back_when_no_results():
    decision = evaluate_retrieval("What is a Jenkinsfile?", StubVectorStore([]))

    assert decision.results == []
    assert decision.should_fallback is True
