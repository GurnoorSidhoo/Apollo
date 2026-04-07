import pytest

from . import helpers


@pytest.mark.adversarial_deterministic
def test_adversarial_deterministic_suite(
    adversarial_cases,
    adversarial_taxonomy,
    adversarial_context_presets,
    adversarial_baseline,
    adversarial_report_dir,
):
    cases = helpers.select_cases(adversarial_cases, "deterministic")
    results = [helpers.run_deterministic_case(case, adversarial_context_presets) for case in cases]
    summary = helpers.summarize_results("deterministic", cases, results, adversarial_taxonomy, adversarial_baseline)
    helpers.write_summary_reports(summary, adversarial_report_dir)
    helpers.assert_summary_is_acceptable(summary, fail_on_safety=False, fail_on_regression=True)
