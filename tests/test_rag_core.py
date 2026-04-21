from langchain_core.documents import Document

import demo_env
from rag_core import (
    FALLBACK,
    build_response_instructions,
    evaluate_retrieval,
    detect_workflow_mode,
    has_question_support,
    is_unsupported_decision_question,
    retrieve,
    renumber_numbered_lines,
    should_force_fallback,
    strip_inline_sources,
    tokenize_text,
)


def test_configure_openmp_sets_demo_default(monkeypatch):
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)

    demo_env.configure_openmp()

    assert demo_env.os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"


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


def test_is_unsupported_decision_question_detects_subjective_requests():
    assert is_unsupported_decision_question(
        "Which Jenkins plugin should I choose for the fastest Kubernetes autoscaling?"
    ) is True


def test_has_question_support_rejects_subjective_recommendation_queries():
    results = [
        (
            Document(
                page_content="The Kubernetes plugin allocates Jenkins agents in Kubernetes pods.",
                metadata={"title": "Kubernetes plugin", "plugin_name": "Kubernetes plugin"},
            ),
            0.28,
        ),
    ]
    assert has_question_support(
        "Which Jenkins plugin should I choose for the fastest Kubernetes autoscaling?",
        results,
    ) is False


def test_strip_inline_sources_removes_source_footer():
    answer = "Use a Jenkinsfile for pipelines.\n\nSources: https://www.jenkins.io/doc/book/pipeline/"
    assert strip_inline_sources(answer) == "Use a Jenkinsfile for pipelines."


def test_strip_inline_sources_removes_singular_source_footer():
    answer = "Git plugin integrates Git repositories.\n\nSource: https://plugins.jenkins.io/git/"
    assert strip_inline_sources(answer) == "Git plugin integrates Git repositories."


def test_strip_inline_sources_removes_demo_noise():
    answer = (
        "Answer:\n\nUse a Jenkinsfile for pipelines.\n\n"
        "5. For more information, refer to the official Jenkins documentation: "
        "https://www.jenkins.io/doc/book/pipeline/jenkinsfile/"
    )

    assert strip_inline_sources(answer) == "Use a Jenkinsfile for pipelines."


def test_strip_inline_sources_removes_detailed_reference_line():
    answer = (
        "The Git plugin integrates Git repositories.\n\n"
        "For more detailed information, please refer to https://plugins.jenkins.io/git/."
    )

    assert strip_inline_sources(answer) == "The Git plugin integrates Git repositories."


def test_strip_inline_sources_removes_unavailable_inline_url():
    answer = (
        "Configure credentials in Jenkins by following the documentation "
        "(https://www.jenkins.io/doc/book/managing-credentials/)."
    )

    assert strip_inline_sources(
        answer,
        sources=["https://www.jenkins.io/doc/book/using/using-credentials/"],
    ) == "Configure credentials in Jenkins by following the documentation."


def test_strip_inline_sources_removes_echoed_step_instruction():
    answer = (
        "Answer with numbered steps for setting up or understanding a Jenkinsfile:\n\n"
        "1. A Jenkinsfile is a text file that contains the definition of a Jenkins Pipeline."
    )

    assert strip_inline_sources(answer).startswith("1. A Jenkinsfile is a text file")


def test_strip_inline_sources_removes_trailing_answer_label():
    answer = "The Git plugin integrates Git repositories.\n\nAnswer"

    assert strip_inline_sources(answer) == "The Git plugin integrates Git repositories."


def test_strip_inline_sources_removes_example_code_block():
    answer = """Use the parallel keyword inside a stage.

Here's an example from your provided context:

```groovy
pipeline {
    agent any
}
```"""

    assert strip_inline_sources(answer) == "Use the parallel keyword inside a stage."


def test_strip_inline_sources_removes_dangling_snippet_intro():
    answer = (
        "Use the parallel keyword inside a stage.\n\n"
        "Here's a snippet from the provided context that demonstrates running parallel stages:"
    )

    assert strip_inline_sources(answer) == "Use the parallel keyword inside a stage."


def test_strip_inline_sources_removes_dangling_numbered_example_intro():
    answer = """1. Configure the credential in Jenkins.

2. In your Jenkinsfile, set the environment variable for the credential you want to use:

3. The credentials helper expands username and password variables."""

    assert strip_inline_sources(answer) == (
        "1. Configure the credential in Jenkins.\n\n"
        "2. The credentials helper expands username and password variables."
    )


def test_renumber_numbered_lines_keeps_sequence():
    answer = "2. Second step\n\n4. Fourth step"

    assert renumber_numbered_lines(answer) == "1. Second step\n\n2. Fourth step"


def test_should_force_fallback_for_unsupported_patterns():
    answer = "Based on the provided context, there is no specific answer."
    assert should_force_fallback(answer) is True


def test_should_not_force_fallback_for_supported_caveats():
    answer = (
        "The Git plugin can apply tags to the Git repository in the workspace. "
        "However, it does not push the applied tag to another location."
    )

    assert should_force_fallback(answer) is False


def test_fallback_constant_is_stable():
    assert FALLBACK == "I could not find this in the Jenkins documentation."


def test_detect_workflow_mode_for_pipeline_questions():
    assert detect_workflow_mode("How do I set up a Jenkins pipeline?") == "pipeline"


def test_build_response_instructions_for_troubleshooting_questions():
    instructions = build_response_instructions("How do I debug a failed Jenkins build?")
    assert "up to 4 numbered troubleshooting steps" in instructions.lower()


def test_build_response_instructions_limits_pipeline_examples():
    instructions = build_response_instructions("How do I run parallel stages in Jenkins?")

    assert "up to 4 numbered steps" in instructions.lower()
    assert "do not include a full jenkinsfile example" in instructions.lower()


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


def test_retrieve_boosts_plugin_slug_queries_without_plugin_keyword():
    docs_result = (
        Document(
            page_content="Pipeline syntax explains declarative and scripted pipeline structure.",
            metadata={"source": "https://www.jenkins.io/doc/book/pipeline/", "source_type": "jenkins_docs"},
        ),
        0.20,
    )
    plugin_result = (
        Document(
            page_content="This plugin and its dependencies form a suite of plugins for Jenkins Pipeline.",
            metadata={
                "source": "https://plugins.jenkins.io/workflow-aggregator/",
                "source_type": "jenkins_plugin",
                "plugin_id": "workflow-aggregator",
                "plugin_name": "Pipeline",
            },
        ),
        0.25,
    )

    results = retrieve(StubVectorStore([docs_result, plugin_result]), "workflow-aggregator")

    assert results[0][0].metadata["source"] == "https://plugins.jenkins.io/workflow-aggregator/"


def test_retrieve_penalizes_generic_plugin_page_for_non_plugin_query():
    plugin_result = (
        Document(
            page_content="This plugin and its dependencies form a suite of plugins for Jenkins Pipeline.",
            metadata={
                "source": "https://plugins.jenkins.io/workflow-aggregator/",
                "source_type": "jenkins_plugin",
                "plugin_id": "workflow-aggregator",
                "plugin_name": "Pipeline",
            },
        ),
        0.10,
    )
    credentials_result = (
        Document(
            page_content="Handling credentials in Pipeline uses credential IDs and bindings.",
            metadata={
                "source": "https://www.jenkins.io/doc/book/using/using-credentials/",
                "source_type": "jenkins_docs",
            },
        ),
        0.12,
    )

    results = retrieve(
        StubVectorStore([plugin_result, credentials_result]),
        "How do I use credentials in Jenkins pipeline?",
    )

    assert results[0][0].metadata["source"] == "https://www.jenkins.io/doc/book/using/using-credentials/"


def test_retrieve_boosts_plugin_overview_for_provide_questions():
    config_result = (
        Document(
            page_content="This option configures polling and repository checkout behavior.",
            metadata={
                "source": "https://plugins.jenkins.io/git/",
                "source_type": "jenkins_plugin",
                "plugin_id": "git",
                "plugin_name": "Git",
            },
        ),
        0.10,
    )
    overview_result = (
        Document(
            page_content="Introduction The git plugin provides fundamental git operations for Jenkins projects.",
            metadata={
                "source": "https://plugins.jenkins.io/git/",
                "source_type": "jenkins_plugin",
                "plugin_id": "git",
                "plugin_name": "Git",
            },
        ),
        0.15,
    )

    results = retrieve(
        StubVectorStore([config_result, overview_result]),
        "What does the Git plugin provide in Jenkins?",
    )

    assert "plugin provides" in results[0][0].page_content.lower()
