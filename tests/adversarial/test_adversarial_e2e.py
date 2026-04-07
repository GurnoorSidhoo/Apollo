import pytest

import apollo

from . import helpers


@pytest.mark.adversarial_llm
@pytest.mark.adversarial_e2e
@pytest.mark.adversarial_safety
def test_adversarial_end_to_end_suite(
    adversarial_cases,
    adversarial_taxonomy,
    adversarial_context_presets,
    adversarial_baseline,
    adversarial_report_dir,
):
    if not apollo.GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY is required for mocked end-to-end adversarial runs")

    cases = helpers.select_cases(adversarial_cases, "llm_end_to_end")
    results = [helpers.run_end_to_end_case(case, adversarial_context_presets) for case in cases]
    summary = helpers.summarize_results("llm_end_to_end", cases, results, adversarial_taxonomy, adversarial_baseline)
    helpers.write_summary_reports(summary, adversarial_report_dir)
    helpers.assert_summary_is_acceptable(summary, fail_on_safety=True, fail_on_regression=True)
