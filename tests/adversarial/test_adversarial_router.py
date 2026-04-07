import pytest

import apollo

from . import helpers


@pytest.mark.adversarial_llm
@pytest.mark.adversarial_router
@pytest.mark.adversarial_safety
def test_adversarial_router_suite(
    adversarial_cases,
    adversarial_taxonomy,
    adversarial_context_presets,
    adversarial_baseline,
    adversarial_report_dir,
):
    if not apollo.GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY is required for LLM-backed adversarial router runs")

    cases = helpers.select_cases(adversarial_cases, "llm_router")
    results = [helpers.run_router_case(case, adversarial_context_presets) for case in cases]
    summary = helpers.summarize_results("llm_router", cases, results, adversarial_taxonomy, adversarial_baseline)
    helpers.write_summary_reports(summary, adversarial_report_dir)
    helpers.assert_summary_is_acceptable(summary, fail_on_safety=True, fail_on_regression=True)
