import unittest
from unittest import mock

import pytest

import apollo


class ImmediateThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.daemon = daemon

    def start(self):
        if self.target:
            self.target(*self.args, **self.kwargs)


def _mock_command(action, *phrases):
    return {
        "action": action,
        "phrases": [phrase.lower() for phrase in phrases],
        "description": action.__name__,
    }


@pytest.fixture(name="route_commands")
def fixture_route_commands():
    return {
        "type_text": _mock_command(apollo.type_text, "type", "write", "dictate"),
        "show_help": _mock_command(apollo.show_help, "help", "what can you do", "list commands", "show commands"),
        "stop_listening": _mock_command(apollo.stop_listening, "stop listening", "go to sleep", "sleep", "pause"),
        "save_file": _mock_command(apollo.save_file, "save", "save file", "save this"),
        "close_window": _mock_command(apollo.close_window, "close window", "close this", "close it"),
        "open_chrome": _mock_command(
            apollo.open_chrome,
            "open chrome",
            "launch chrome",
            "open google chrome",
            "open browser",
            "launch browser",
            "open my browser",
        ),
        "scroll_down": _mock_command(apollo.scroll_down, "scroll down", "page down"),
        "paste": _mock_command(apollo.paste, "paste", "paste that", "paste it"),
    }


CLASSIFY_ROUTE_CASES = [
    pytest.param("", None, 0.0, "", apollo.Route.UNKNOWN, 1, id="r01-empty"),
    pytest.param("type hello world", "type_text", 1.0, "hello world", apollo.Route.DIRECT, 2, id="r02-type-payload"),
    pytest.param("type and then paste", "type_text", 0.7, "and then paste", apollo.Route.WORKFLOW, 5, id="r03-type-multi-step"),
    pytest.param("help", "show_help", 1.0, "", apollo.Route.DIRECT, 3, id="r04-help"),
    pytest.param("stop listening", "stop_listening", 1.0, "", apollo.Route.DIRECT, 3, id="r05-stop"),
    pytest.param("don't save", "save_file", 1.0, "", apollo.Route.ROUTER, 4, id="r06-negation-save"),
    pytest.param("don't close it minimize it", "close_window", 0.6, "", apollo.Route.ROUTER, 4, id="r07-negation-close"),
    pytest.param("instead of chrome open safari", "open_chrome", 0.5, "", apollo.Route.ROUTER, 4, id="r08-negation-instead"),
    pytest.param("open chrome and go to github", "open_chrome", 1.0, "and go to github", apollo.Route.WORKFLOW, 5, id="r09-open-and-go"),
    pytest.param("open chrome and then search", "open_chrome", 1.0, "and then search", apollo.Route.WORKFLOW, 5, id="r10-open-and-then"),
    pytest.param("ask claude about the weather", None, 0.0, "", apollo.Route.WORKFLOW, 5, id="r11-ask-claude"),
    pytest.param("new chat", None, 0.0, "", apollo.Route.WORKFLOW, 5, id="r12-new-chat"),
    pytest.param("click on the submit button", None, 0.0, "", apollo.Route.WORKFLOW, 6, id="r13-click-submit"),
    pytest.param("click on settings", None, 0.0, "", apollo.Route.WORKFLOW, 6, id="r14-click-settings"),
    pytest.param("close chrome", None, 0.0, "", apollo.Route.WORKFLOW, 7, id="r15-close-chrome"),
    pytest.param("quit spotify", None, 0.0, "", apollo.Route.WORKFLOW, 7, id="r16-quit-spotify"),
    pytest.param("open slack", None, 0.0, "", apollo.Route.WORKFLOW, 8, id="r17-open-slack"),
    pytest.param("open discord", None, 0.0, "", apollo.Route.WORKFLOW, 8, id="r18-open-discord"),
    pytest.param("save", "save_file", 1.0, "", apollo.Route.DIRECT, 9, id="r19-save"),
    pytest.param("scroll down", "scroll_down", 1.0, "", apollo.Route.DIRECT, 9, id="r20-scroll-down"),
    pytest.param("paste", "paste", 1.0, "", apollo.Route.DIRECT, 9, id="r21-paste"),
    pytest.param("open chrome", "open_chrome", 1.0, "", apollo.Route.DIRECT, 9, id="r22-open-chrome"),
    pytest.param("please save", "save_file", 0.65, "", apollo.Route.DIRECT, 10, id="r23-please-save"),
    pytest.param("can you scroll down", "scroll_down", 0.55, "", apollo.Route.ROUTER, 11, id="r24-can-you-scroll"),
    pytest.param("um save the file please", "save_file", 0.45, "", apollo.Route.ROUTER, 11, id="r25-um-save-file"),
    pytest.param("um", None, 0.0, "", apollo.Route.UNKNOWN, 12, id="r26-um"),
    pytest.param("pizza", None, 0.0, "", apollo.Route.UNKNOWN, 12, id="r27-pizza"),
    pytest.param("try again", None, 0.0, "", apollo.Route.UNKNOWN, 12, id="r28-try-again"),
    pytest.param("do that again", None, 0.0, "", apollo.Route.ROUTER, 13, id="r29-do-that-again"),
    pytest.param("what time is it in london", None, 0.0, "", apollo.Route.ROUTER, 13, id="r30-what-time"),
    pytest.param("click on it", None, 0.0, "", apollo.Route.ROUTER, 13, id="r31-click-it"),
    pytest.param("click", None, 0.0, "", apollo.Route.UNKNOWN, 12, id="r32-click-bare"),
    pytest.param("ensure claude is using sonnet", None, 0.0, "", apollo.Route.WORKFLOW, 5, id="r33-ensure-sonnet"),
    pytest.param("open the app", None, 0.0, "", apollo.Route.ROUTER, 13, id="r34-open-the-app"),
]


@pytest.mark.parametrize(
    "transcript,command_key,confidence,extra,expected_route,expected_rule",
    CLASSIFY_ROUTE_CASES,
)
def test_classify_route_matrix(route_commands, transcript, command_key, confidence, extra, expected_route, expected_rule):
    command = route_commands.get(command_key) if command_key else None
    route = apollo.classify_route(transcript, command, confidence, extra)
    assert route == expected_route, f"expected rule {expected_rule}, got {route}"


@pytest.mark.parametrize(
    "transcript,command_key,extra,expected_reason",
    [
        pytest.param("click on settings", None, "", "click target: settings", id="click-target"),
        pytest.param("close chrome", None, "", "quit app: Google Chrome", id="quit-app"),
        pytest.param("open slack", None, "", "open app: Slack", id="simple-open"),
        pytest.param("open chrome and go to github", "open_chrome", "and go to github", "multi-step request", id="multi-step"),
    ],
)
def test_build_deterministic_workflow_reason(route_commands, transcript, command_key, extra, expected_reason):
    command = route_commands.get(command_key) if command_key else None
    reason = apollo._build_deterministic_workflow_reason(transcript, command, extra)
    assert reason == expected_reason


def test_classify_route_keeps_soft_prefixed_exact_matches_conservative(route_commands):
    route = apollo.classify_route("can you scroll down", route_commands["scroll_down"], 1.0, "")
    assert route == apollo.Route.ROUTER


class ApolloReliabilityTests(unittest.TestCase):
    def _inline_timeout(self, timeout_seconds, func, *args, **kwargs):
        return func(*args, **kwargs)

    def _run_structured(self, responses, func):
        calls = []
        iterator = iter(responses)

        def fake_candidate(**kwargs):
            calls.append(kwargs)
            value = next(iterator)
            if isinstance(value, Exception):
                raise value
            return value, "rest"

        with mock.patch.object(apollo, "GEMINI_API_KEY", "test-key"):
            with mock.patch.object(apollo, "run_with_timeout", side_effect=self._inline_timeout):
                with mock.patch.object(apollo, "gemini_generate_structured_candidate", side_effect=fake_candidate):
                    return func(), calls

    def test_repeated_wake_word_restarts_capture_without_appending(self):
        listener = apollo.AudioListener()
        listener._start_command_capture("call my mom", now=1.0, reason="test")

        with mock.patch.object(apollo, "match_command", return_value=(None, 0, "")):
            listener._handle_transcript("biggie type hello")

        self.assertTrue(listener.is_capturing_command)
        self.assertEqual(listener.command_buffer, "type hello")

    def test_unknown_planner_result_clears_capture_state(self):
        listener = apollo.AudioListener()
        listener._start_command_capture("mystery command", now=1.0, reason="test")

        with mock.patch.object(apollo.threading, "Thread", ImmediateThread):
            with mock.patch.object(apollo, "APOLLO_2STAGE_PLANNER", True):
                with mock.patch.object(apollo, "LLM_FALLBACK_ENABLED", True):
                    with mock.patch.object(apollo, "call_router", return_value={"action": "unknown", "reason": "not supported here"}):
                        with mock.patch.object(apollo, "play_sound"):
                            with mock.patch.object(apollo, "say"):
                                with mock.patch.object(apollo, "update_command_state"):
                                    with mock.patch.object(apollo, "debug_event"):
                                        listener._handle_utterance_end()

        self.assertFalse(listener.is_capturing_command)
        self.assertEqual(listener.command_buffer, "")

    def test_deepgram_variants_prefer_bare_by_default(self):
        listener = apollo.AudioListener()

        with mock.patch.object(apollo.audio, "DEEPGRAM_PREFER_RICH_VARIANTS", False):
            variants = listener._build_deepgram_connect_variants()

        self.assertEqual(variants[0]["name"], "bare")

    def test_deepgram_variants_can_prefer_rich_when_enabled(self):
        listener = apollo.AudioListener()

        with mock.patch.object(apollo.audio, "DEEPGRAM_PREFER_RICH_VARIANTS", True):
            variants = listener._build_deepgram_connect_variants()

        self.assertNotEqual(variants[0]["name"], "bare")
        self.assertIn("bare", [variant["name"] for variant in variants])

    def test_has_mediapipe_face_mesh_checks_classic_module_paths(self):
        with mock.patch.object(apollo.utils.importlib.util, "find_spec", side_effect=lambda name: object() if name == "mediapipe.solutions.face_mesh" else None):
            self.assertTrue(apollo.has_mediapipe_face_mesh())

        with mock.patch.object(apollo.utils.importlib.util, "find_spec", side_effect=lambda name: object() if name == "mediapipe.python.solutions.face_mesh" else None):
            self.assertTrue(apollo.has_mediapipe_face_mesh())

        with mock.patch.object(apollo.utils.importlib.util, "find_spec", return_value=None):
            self.assertFalse(apollo.has_mediapipe_face_mesh())

    def test_mid_sentence_biggie_does_not_trigger_wake_word(self):
        detected, trailing = apollo.detect_wake_word("yeah my biggie is stupid")
        self.assertFalse(detected)
        self.assertEqual(trailing, "")

    def test_scroll_down_matches_registered_command(self):
        command, confidence, extra = apollo.match_command("scroll down")
        self.assertIsNotNone(command)
        self.assertEqual(command["action"].__name__, "scroll_down")
        self.assertGreaterEqual(confidence, 0.75)
        self.assertEqual(extra, "")

    def test_validate_router_output_accepts_missing_reason_and_corrects_function(self):
        result = apollo.validate_router_output({
            "action": "command",
            "function": "save_fil",
        })

        self.assertEqual(result["action"], "command")
        self.assertEqual(result["function"], "save_file")
        self.assertEqual(result["reason"], "router reason missing")

    def test_validate_router_output_rejects_unknown_fields(self):
        with self.assertRaises(apollo.PlannerValidationError):
            apollo.validate_router_output({
                "action": "workflow",
                "reason": "needs many steps",
                "extra": "nope",
            })

    def test_validate_router_output_empty_function_becomes_unknown(self):
        result = apollo.validate_router_output({
            "action": "command",
            "function": "   ",
            "reason": "blank function field",
        })

        self.assertEqual(result["action"], "unknown")

    def test_validate_workflow_output_rejects_invalid_step_shape(self):
        with self.assertRaises(apollo.PlannerValidationError):
            apollo.validate_workflow_output({
                "description": "Pressing enter",
                "steps": [
                    {"type": "keypress", "reason": "press enter key"},
                ],
            })

    def test_validate_workflow_output_rejects_unknown_fields(self):
        with self.assertRaises(apollo.PlannerValidationError):
            apollo.validate_workflow_output({
                "description": "Talking",
                "steps": [
                    {"type": "say", "text": "hi", "reason": "say hello now", "extra": "nope"},
                ],
            })

    def test_validate_workflow_output_rejects_negative_wait(self):
        with self.assertRaises(apollo.PlannerValidationError):
            apollo.validate_workflow_output({
                "description": "Waiting",
                "steps": [
                    {"type": "wait", "seconds": -1, "reason": "bad wait value"},
                ],
            })

    def test_validate_workflow_output_truncates_overlong_workflow(self):
        result = apollo.validate_workflow_output({
            "description": "Doing many things",
            "steps": [
                {"type": "say", "text": f"step {index}", "reason": "announce step item"}
                for index in range(apollo.MAX_WORKFLOW_STEPS + 2)
            ],
        })

        self.assertEqual(len(result["steps"]), apollo.MAX_WORKFLOW_STEPS)
        self.assertEqual(result["steps"][-1]["type"], "say")
        self.assertEqual(result["steps"][-1]["text"], "I planned only the first few steps.")

    def test_call_router_accepts_valid_structured_json(self):
        result, calls = self._run_structured(
            ['{"action":"command","function":"save_file","reason":"save file"}'],
            lambda: apollo.call_router("save"),
        )

        self.assertEqual(result["action"], "command")
        self.assertEqual(result["function"], "save_file")
        self.assertEqual(len(calls), 1)

    def test_call_workflow_planner_accepts_valid_structured_json(self):
        result, calls = self._run_structured(
            ['{"description":"Clicking submit","steps":[{"type":"vision","task":"Click the submit button","reason":"click submit"}]}'],
            lambda: apollo.call_workflow_planner("click submit", "needs multiple steps"),
        )

        self.assertEqual(result["description"], "Clicking submit")
        self.assertEqual(result["steps"][0]["type"], "vision")
        self.assertEqual(len(calls), 1)

    def test_llm_interpret_command_eval_mode_prefers_eval_model(self):
        with mock.patch.dict("os.environ", {"APOLLO_EVAL_PLANNER_MODEL": "gemini-2.5-flash"}):
            with mock.patch.object(apollo, "gemini_generate_json", return_value='{"action":"unknown"}') as generate_json:
                apollo.llm_interpret_command("click submit", complex=True, eval_mode=True)

        self.assertEqual(
            generate_json.call_args.kwargs["preferred_models"],
            apollo.model_candidates("gemini-2.5-flash", apollo.GEMINI_PLANNER_MODEL, apollo.GEMINI_MODEL),
        )

    def test_llm_interpret_command_normal_complex_mode_keeps_production_model(self):
        with mock.patch.object(apollo, "gemini_generate_json", return_value='{"action":"unknown"}') as generate_json:
            apollo.llm_interpret_command("click submit", complex=True)

        self.assertEqual(
            generate_json.call_args.kwargs["preferred_models"],
            apollo.model_candidates(apollo.GEMINI_PLANNER_MODEL, apollo.GEMINI_MODEL, "gemini-2.5-flash"),
        )

    def test_resolve_eval_planner_model_falls_back_to_production_when_unset(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            model = apollo.resolve_eval_planner_model()

        self.assertEqual(model, apollo.GEMINI_PLANNER_MODEL)

    def test_replan_accepts_valid_structured_json(self):
        result, calls = self._run_structured(
            ['{"description":"Recovering search","steps":[{"type":"vision","task":"Click the search field in Spotify","reason":"resume search"}]}'],
            lambda: apollo.replan_workflow(
                "play god playlist",
                {"description": "Original", "steps": [{"type": "focus_app", "app": "Spotify", "reason": "focus spotify"}]},
                {"type": "vision", "task": "Click search", "reason": "resume search"},
                {"reason": "step_returned_false", "message": "Workflow step returned False"},
                [{"index": 1, "step": {"type": "focus_app", "app": "Spotify", "reason": "focus spotify"}, "result": "ok"}],
            ),
        )

        self.assertEqual(result["steps"][0]["type"], "vision")
        self.assertEqual(len(calls), 1)

    def test_execute_vision_task_accepts_valid_structured_json(self):
        image_bytes = b"fake"
        metadata = {
            "region_x": 10.0,
            "region_y": 20.0,
            "logical_width": 100.0,
            "logical_height": 50.0,
            "pixel_width": 200,
            "pixel_height": 100,
            "scale_x": 2.0,
            "scale_y": 2.0,
        }

        with mock.patch.object(apollo, "GEMINI_API_KEY", "test-key"):
            with mock.patch.object(apollo, "capture_screenshot", return_value=(image_bytes, metadata)):
                with mock.patch.object(apollo, "run_with_timeout", side_effect=self._inline_timeout):
                    with mock.patch.object(apollo, "gemini_generate_structured_candidate", return_value=('{"action":"click","x":20,"y":10,"description":"Click play"}', "rest")):
                        with mock.patch.object(apollo, "say"):
                            with mock.patch.object(apollo, "click_at") as click_at:
                                handled = apollo.execute_vision_task("Click play", "play music")

        self.assertTrue(handled)
        click_at.assert_called_once_with(20, 25)

    def test_structured_wrapper_retries_on_second_attempt(self):
        result, calls = self._run_structured(
            [
                '{"action":"command"',
                '{"action":"command","function":"save_file","reason":"save file"}',
            ],
            lambda: apollo.call_gemini_structured(
                system_instruction="router",
                user_text="save",
                response_json_schema=apollo.ROUTER_RESPONSE_JSON_SCHEMA,
                validator=apollo.validate_router_output,
                preferred_models=["primary", "fallback"],
                call_type="router",
                max_output_tokens=100,
                timeout_seconds=1,
            ),
        )

        self.assertEqual(result["function"], "save_file")
        self.assertEqual([call["model_name"] for call in calls], ["primary", "primary"])

    def test_structured_wrapper_uses_fallback_model_after_invalid_output(self):
        result, calls = self._run_structured(
            [
                '{"action":"command"',
                '{"action":"command"',
                '{"action":"command","function":"save_file","reason":"save file"}',
            ],
            lambda: apollo.call_gemini_structured(
                system_instruction="router",
                user_text="save",
                response_json_schema=apollo.ROUTER_RESPONSE_JSON_SCHEMA,
                validator=apollo.validate_router_output,
                preferred_models=["primary", "fallback"],
                call_type="router",
                max_output_tokens=100,
                timeout_seconds=1,
            ),
        )

        self.assertEqual(result["function"], "save_file")
        self.assertEqual([call["model_name"] for call in calls], ["primary", "primary", "fallback"])

    def test_structured_wrapper_rejects_partial_json(self):
        with mock.patch.object(apollo, "GEMINI_API_KEY", "test-key"):
            with mock.patch.object(apollo, "run_with_timeout", side_effect=self._inline_timeout):
                with mock.patch.object(
                    apollo,
                    "gemini_generate_structured_candidate",
                    side_effect=[
                        ('{"action":"command"', "rest"),
                        ('{"action":"command"', "rest"),
                        ('{"action":"command"', "rest"),
                    ],
                ):
                    with self.assertRaises(apollo.StructuredOutputError) as ctx:
                        apollo.call_gemini_structured(
                            system_instruction="router",
                            user_text="save",
                            response_json_schema=apollo.ROUTER_RESPONSE_JSON_SCHEMA,
                            validator=apollo.validate_router_output,
                            preferred_models=["primary", "fallback"],
                            call_type="router",
                            max_output_tokens=100,
                            timeout_seconds=1,
                        )

        self.assertEqual(ctx.exception.category, "truncated_output")

    def test_structured_wrapper_rejects_truncated_workflow_json(self):
        with mock.patch.object(apollo, "GEMINI_API_KEY", "test-key"):
            with mock.patch.object(apollo, "run_with_timeout", side_effect=self._inline_timeout):
                with mock.patch.object(
                    apollo,
                    "gemini_generate_structured_candidate",
                    side_effect=[
                        ('{"description":"Plan","steps":[{"type":"say"', "rest"),
                        ('{"description":"Plan","steps":[{"type":"say"', "rest"),
                        ('{"description":"Plan","steps":[{"type":"say"', "rest"),
                    ],
                ):
                    with self.assertRaises(apollo.StructuredOutputError) as ctx:
                        apollo.call_gemini_structured(
                            system_instruction="planner",
                            user_text="plan",
                            response_json_schema=apollo.WORKFLOW_PLANNER_RESPONSE_JSON_SCHEMA,
                            validator=apollo.validate_workflow_output,
                            preferred_models=["primary", "fallback"],
                            call_type="workflow_planner",
                            max_output_tokens=100,
                            timeout_seconds=1,
                        )

        self.assertEqual(ctx.exception.category, "truncated_output")

    def test_structured_wrapper_rejects_wrong_field_names(self):
        with mock.patch.object(apollo, "GEMINI_API_KEY", "test-key"):
            with mock.patch.object(apollo, "run_with_timeout", side_effect=self._inline_timeout):
                with mock.patch.object(
                    apollo,
                    "gemini_generate_structured_candidate",
                    side_effect=[
                        ('{"description":"Plan","steps":[{"type":"say","text":"hi","reasn":"oops"}]}', "rest"),
                        ('{"description":"Plan","steps":[{"type":"say","text":"hi","reasn":"oops"}]}', "rest"),
                        ('{"description":"Plan","steps":[{"type":"say","text":"hi","reasn":"oops"}]}', "rest"),
                    ],
                ):
                    with self.assertRaises(apollo.StructuredOutputError) as ctx:
                        apollo.call_gemini_structured(
                            system_instruction="planner",
                            user_text="plan",
                            response_json_schema=apollo.WORKFLOW_PLANNER_RESPONSE_JSON_SCHEMA,
                            validator=apollo.validate_workflow_output,
                            preferred_models=["primary", "fallback"],
                            call_type="workflow_planner",
                            max_output_tokens=100,
                            timeout_seconds=1,
                        )

        self.assertEqual(ctx.exception.category, "schema_mismatch")

    def test_structured_wrapper_rejects_missing_required_fields(self):
        with mock.patch.object(apollo, "GEMINI_API_KEY", "test-key"):
            with mock.patch.object(apollo, "run_with_timeout", side_effect=self._inline_timeout):
                with mock.patch.object(
                    apollo,
                    "gemini_generate_structured_candidate",
                    side_effect=[
                        ('{"description":"Plan"}', "rest"),
                        ('{"description":"Plan"}', "rest"),
                        ('{"description":"Plan"}', "rest"),
                    ],
                ):
                    with self.assertRaises(apollo.StructuredOutputError) as ctx:
                        apollo.call_gemini_structured(
                            system_instruction="planner",
                            user_text="plan",
                            response_json_schema=apollo.WORKFLOW_PLANNER_RESPONSE_JSON_SCHEMA,
                            validator=apollo.validate_workflow_output,
                            preferred_models=["primary", "fallback"],
                            call_type="workflow_planner",
                            max_output_tokens=100,
                            timeout_seconds=1,
                        )

        self.assertEqual(ctx.exception.category, "schema_mismatch")

    def test_structured_wrapper_rejects_unknown_fields(self):
        with mock.patch.object(apollo, "GEMINI_API_KEY", "test-key"):
            with mock.patch.object(apollo, "run_with_timeout", side_effect=self._inline_timeout):
                with mock.patch.object(
                    apollo,
                    "gemini_generate_structured_candidate",
                    side_effect=[
                        ('{"action":"workflow","reason":"many steps","extra":"nope"}', "rest"),
                        ('{"action":"workflow","reason":"many steps","extra":"nope"}', "rest"),
                        ('{"action":"workflow","reason":"many steps","extra":"nope"}', "rest"),
                    ],
                ):
                    with self.assertRaises(apollo.StructuredOutputError) as ctx:
                        apollo.call_gemini_structured(
                            system_instruction="router",
                            user_text="open chrome and github",
                            response_json_schema=apollo.ROUTER_RESPONSE_JSON_SCHEMA,
                            validator=apollo.validate_router_output,
                            preferred_models=["primary", "fallback"],
                            call_type="router",
                            max_output_tokens=100,
                            timeout_seconds=1,
                        )

        self.assertEqual(ctx.exception.category, "schema_mismatch")

    def test_structured_wrapper_rejects_invalid_enum_values(self):
        with mock.patch.object(apollo, "GEMINI_API_KEY", "test-key"):
            with mock.patch.object(apollo, "run_with_timeout", side_effect=self._inline_timeout):
                with mock.patch.object(
                    apollo,
                    "gemini_generate_structured_candidate",
                    side_effect=[
                        ('{"action":"teleport","description":"bad"}', "rest"),
                        ('{"action":"teleport","description":"bad"}', "rest"),
                        ('{"action":"teleport","description":"bad"}', "rest"),
                    ],
                ):
                    with self.assertRaises(apollo.StructuredOutputError) as ctx:
                        apollo.call_gemini_structured(
                            system_instruction="vision",
                            user_text="click",
                            response_json_schema=apollo.VISION_ACTION_RESPONSE_JSON_SCHEMA,
                            validator=apollo.validate_vision_action_output,
                            preferred_models=["primary", "fallback"],
                            call_type="vision",
                            max_output_tokens=100,
                            timeout_seconds=1,
                        )

        self.assertEqual(ctx.exception.category, "schema_mismatch")

    def test_structured_wrapper_rejects_empty_response(self):
        with mock.patch.object(apollo, "GEMINI_API_KEY", "test-key"):
            with mock.patch.object(apollo, "run_with_timeout", side_effect=self._inline_timeout):
                with mock.patch.object(
                    apollo,
                    "gemini_generate_structured_candidate",
                    side_effect=[
                        ("", "rest"),
                        ("", "rest"),
                        ("", "rest"),
                    ],
                ):
                    with self.assertRaises(apollo.StructuredOutputError) as ctx:
                        apollo.call_gemini_structured(
                            system_instruction="router",
                            user_text="save",
                            response_json_schema=apollo.ROUTER_RESPONSE_JSON_SCHEMA,
                            validator=apollo.validate_router_output,
                            preferred_models=["primary", "fallback"],
                            call_type="router",
                            max_output_tokens=100,
                            timeout_seconds=1,
                        )

        self.assertEqual(ctx.exception.category, "empty_response")

    def test_invalid_planner_output_fails_closed_without_executing(self):
        with mock.patch.object(apollo, "APOLLO_2STAGE_PLANNER", True):
            with mock.patch.object(apollo, "LLM_FALLBACK_ENABLED", True):
                with mock.patch.object(apollo, "call_router") as call_router:
                    with mock.patch.object(
                        apollo,
                        "call_workflow_planner",
                        side_effect=apollo.StructuredOutputError(
                            "schema_mismatch",
                            "workflow schema mismatch",
                            call_type="workflow_planner",
                        ),
                    ):
                        with mock.patch.object(apollo, "execute_workflow") as execute_workflow:
                            with mock.patch.object(apollo, "say"):
                                with mock.patch.object(apollo, "play_sound"):
                                    with mock.patch.object(apollo, "update_command_state"):
                                        with mock.patch.object(apollo, "debug_event"):
                                            handled = apollo.route_command("open chrome and go to github")

        self.assertFalse(handled)
        call_router.assert_not_called()
        execute_workflow.assert_not_called()

    def test_click_enter_is_not_routed_to_vision_click(self):
        self.assertEqual(apollo.extract_click_target_request("click enter"), "")

    def test_two_stage_route_prefilters_simple_save(self):
        with mock.patch.object(apollo, "APOLLO_2STAGE_PLANNER", True):
            with mock.patch.object(apollo, "execute_matched_command", return_value=True) as execute_match:
                with mock.patch.object(apollo, "call_router") as call_router:
                    handled = apollo.route_command("save")

        self.assertTrue(handled)
        execute_match.assert_called_once()
        call_router.assert_not_called()

    def test_two_stage_route_command_response_executes_registered_command(self):
        with mock.patch.object(apollo, "APOLLO_2STAGE_PLANNER", True):
            with mock.patch.object(apollo, "LLM_FALLBACK_ENABLED", True):
                with mock.patch.object(apollo, "match_command", return_value=(None, 0, "")):
                    with mock.patch.object(apollo, "call_router", return_value={
                        "action": "command",
                        "function": "save_file",
                        "reason": "direct single command",
                    }):
                        with mock.patch.object(apollo, "execute_registered_command", return_value=True) as execute_registered:
                            with mock.patch.object(apollo, "play_sound"):
                                with mock.patch.object(apollo, "update_command_state"):
                                    with mock.patch.object(apollo, "debug_event"):
                                        handled = apollo.route_command("please save the file")

        self.assertTrue(handled)
        execute_registered.assert_called_once_with("save_file", "please save the file", args=None, source="router")

    def test_two_stage_click_request_routes_to_workflow_planner(self):
        workflow = {
            "description": "Clicking the requested target",
            "steps": [
                {
                    "type": "vision",
                    "task": "Click on the UI element labeled submit on the current screen",
                    "reason": "locate and click submit",
                },
            ],
        }
        with mock.patch.object(apollo, "APOLLO_2STAGE_PLANNER", True):
            with mock.patch.object(apollo, "LLM_FALLBACK_ENABLED", True):
                with mock.patch.object(apollo, "call_router") as call_router:
                    with mock.patch.object(apollo, "call_workflow_planner", return_value=workflow) as planner:
                        with mock.patch.object(apollo, "execute_workflow", return_value=True) as execute_workflow:
                            with mock.patch.object(apollo, "play_sound"):
                                with mock.patch.object(apollo, "update_command_state"):
                                    with mock.patch.object(apollo, "debug_event"):
                                        handled = apollo.route_command("click on the submit button")

        self.assertTrue(handled)
        call_router.assert_not_called()
        planner.assert_called_once_with("click on the submit button", "click target: the submit button")
        execute_workflow.assert_called_once_with(workflow, "click on the submit button")

    def test_two_stage_close_chrome_routes_to_quit_workflow(self):
        workflow = {
            "description": "Quitting Google Chrome",
            "steps": [
                {"type": "quit_app", "app": "Google Chrome", "reason": "user asked to quit chrome"},
            ],
        }
        with mock.patch.object(apollo, "APOLLO_2STAGE_PLANNER", True):
            with mock.patch.object(apollo, "LLM_FALLBACK_ENABLED", True):
                with mock.patch.object(apollo, "call_router") as call_router:
                    with mock.patch.object(apollo, "call_workflow_planner", return_value=workflow) as planner:
                        with mock.patch.object(apollo, "execute_workflow", return_value=True) as execute_workflow:
                            with mock.patch.object(apollo, "play_sound"):
                                with mock.patch.object(apollo, "update_command_state"):
                                    with mock.patch.object(apollo, "debug_event"):
                                        handled = apollo.route_command("close chrome")

        self.assertTrue(handled)
        call_router.assert_not_called()
        planner.assert_called_once_with("close chrome", "quit app: Google Chrome")
        execute_workflow.assert_called_once_with(workflow, "close chrome")

    def test_two_stage_router_failure_falls_back_to_local_match(self):
        save_cmd, confidence, extra = apollo.match_command("save")
        self.assertIsNotNone(save_cmd)

        with mock.patch.object(apollo, "APOLLO_2STAGE_PLANNER", True):
            with mock.patch.object(apollo, "LLM_FALLBACK_ENABLED", True):
                with mock.patch.object(apollo, "match_command", return_value=(save_cmd, 0.45, extra)):
                    with mock.patch.object(apollo, "call_router", side_effect=TimeoutError("router timeout")):
                        with mock.patch.object(apollo, "execute_matched_command", return_value=True) as execute_match:
                            with mock.patch.object(apollo, "play_sound"):
                                with mock.patch.object(apollo, "update_command_state"):
                                    with mock.patch.object(apollo, "debug_event"):
                                        handled = apollo.route_command("um save the file please")

        self.assertTrue(handled)
        execute_match.assert_called_once()

    def test_two_stage_planner_timeout_reports_failure(self):
        with mock.patch.object(apollo, "APOLLO_2STAGE_PLANNER", True):
            with mock.patch.object(apollo, "LLM_FALLBACK_ENABLED", True):
                with mock.patch.object(apollo, "call_router") as call_router:
                    with mock.patch.object(apollo, "call_workflow_planner", side_effect=TimeoutError("planner timeout")):
                        with mock.patch.object(apollo, "say") as say_mock:
                            with mock.patch.object(apollo, "play_sound"):
                                with mock.patch.object(apollo, "update_command_state"):
                                    with mock.patch.object(apollo, "debug_event"):
                                        handled = apollo.route_command("open chrome and go to github")

        self.assertFalse(handled)
        call_router.assert_not_called()
        say_mock.assert_called_with("I understood but couldn't plan the steps")

    def test_two_stage_short_unknown_skips_router_and_planner(self):
        with mock.patch.object(apollo, "APOLLO_2STAGE_PLANNER", True):
            with mock.patch.object(apollo, "LLM_FALLBACK_ENABLED", True):
                with mock.patch.object(apollo, "call_router") as call_router:
                    with mock.patch.object(apollo, "call_workflow_planner") as planner:
                        with mock.patch.object(apollo, "say") as say_mock:
                            with mock.patch.object(apollo, "play_sound"):
                                with mock.patch.object(apollo, "update_command_state"):
                                    with mock.patch.object(apollo, "debug_event"):
                                        handled = apollo.route_command("try again")

        self.assertFalse(handled)
        call_router.assert_not_called()
        planner.assert_not_called()
        say_mock.assert_called_with("Sorry, I didn't understand that")

    def test_wait_for_state_success(self):
        state = {"now": 0.0, "calls": 0}

        def fake_monotonic():
            return state["now"]

        def fake_sleep(seconds):
            state["now"] += seconds

        def condition():
            state["calls"] += 1
            return state["calls"] >= 3

        with mock.patch.object(apollo.time, "monotonic", side_effect=fake_monotonic):
            with mock.patch.object(apollo.time, "sleep", side_effect=fake_sleep):
                handled = apollo.wait_for_state(
                    condition,
                    timeout_seconds=1.0,
                    poll_interval=0.2,
                    condition_name="unit_test_condition",
                )

        self.assertTrue(handled)
        self.assertEqual(state["calls"], 3)

    def test_wait_for_state_times_out(self):
        state = {"now": 0.0}

        def fake_monotonic():
            return state["now"]

        def fake_sleep(seconds):
            state["now"] += seconds

        with mock.patch.object(apollo.time, "monotonic", side_effect=fake_monotonic):
            with mock.patch.object(apollo.time, "sleep", side_effect=fake_sleep):
                handled = apollo.wait_for_state(
                    lambda: False,
                    timeout_seconds=0.5,
                    poll_interval=0.2,
                    condition_name="timeout_case",
                )

        self.assertFalse(handled)

    def test_wait_for_state_recovers_from_condition_exception(self):
        state = {"now": 0.0, "calls": 0}

        def fake_monotonic():
            return state["now"]

        def fake_sleep(seconds):
            state["now"] += seconds

        def condition():
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("transient AX issue")
            return True

        with mock.patch.object(apollo.time, "monotonic", side_effect=fake_monotonic):
            with mock.patch.object(apollo.time, "sleep", side_effect=fake_sleep):
                handled = apollo.wait_for_state(
                    condition,
                    timeout_seconds=1.0,
                    poll_interval=0.2,
                    condition_name="exception_case",
                )

        self.assertTrue(handled)

    def test_validate_workflow_output_accepts_wait_for_state(self):
        result = apollo.validate_workflow_output({
            "description": "Waiting for Finder",
            "steps": [
                {
                    "type": "wait_for_state",
                    "condition": "app_frontmost",
                    "app": "Finder",
                    "timeout_seconds": "2.5",
                    "poll_interval": "0.2",
                    "reason": "wait for finder",
                },
            ],
        })

        step = result["steps"][0]
        self.assertEqual(step["type"], "wait_for_state")
        self.assertEqual(step["condition"], "app_frontmost")
        self.assertEqual(step["app"], "Finder")
        self.assertEqual(step["timeout_seconds"], 2.5)
        self.assertEqual(step["poll_interval"], 0.2)

    def test_validate_workflow_output_rejects_wait_for_state_missing_label(self):
        with self.assertRaises(apollo.PlannerValidationError):
            apollo.validate_workflow_output({
                "description": "Waiting for sidebar item",
                "steps": [
                    {
                        "type": "wait_for_state",
                        "condition": "element_exists",
                        "app": "Finder",
                        "reason": "wait for sidebar",
                    },
                ],
            })

    def test_validate_vision_action_output_is_backward_compatible(self):
        result = apollo.validate_vision_action_output({
            "action": "click",
            "x": 20,
            "y": 10,
            "description": "Click play",
        })

        self.assertEqual(result["action"], "click")
        self.assertEqual(result["x"], 20)
        self.assertEqual(result["y"], 10)
        self.assertNotIn("confidence", result)
        self.assertNotIn("target_label", result)

    def test_execute_vision_task_rejects_low_confidence_click(self):
        resolution = {
            "source": "vision",
            "target_app": "Finder",
            "action": {
                "action": "click",
                "x": 100,
                "y": 200,
                "description": "Click risky target",
                "confidence": 0.59,
            },
            "metadata": {"coordinate_space": "global"},
        }

        with mock.patch.object(apollo, "resolve_ui_target", return_value=resolution):
            with mock.patch.object(apollo, "say"):
                with mock.patch.object(apollo, "click_at") as click_at:
                    with mock.patch.object(apollo, "verify_postcondition") as verify_postcondition:
                        handled = apollo.execute_vision_task("Click risky target", "click risky target")

        self.assertFalse(handled)
        click_at.assert_not_called()
        verify_postcondition.assert_not_called()

    def test_execute_vision_task_allows_threshold_confidence_click(self):
        resolution = {
            "source": "vision",
            "target_app": "Finder",
            "action": {
                "action": "click",
                "x": 140,
                "y": 90,
                "description": "Click safe target",
                "confidence": 0.6,
                "expected_postcondition": "The pane is visible",
            },
            "metadata": {"coordinate_space": "global"},
        }

        with mock.patch.object(apollo, "resolve_ui_target", return_value=resolution):
            with mock.patch.object(apollo, "verify_postcondition", return_value={
                "status": "verified",
                "reason": "verified",
                "method": "ax",
                "used_gemini": False,
            }):
                with mock.patch.object(apollo, "say"):
                    with mock.patch.object(apollo, "log_command"):
                        with mock.patch.object(apollo.time, "sleep"):
                            with mock.patch.object(apollo, "click_at") as click_at:
                                handled = apollo.execute_vision_task("Click safe target", "click safe target")

        self.assertTrue(handled)
        click_at.assert_called_once_with(140, 90)

    def test_should_use_accessibility_prefers_finder(self):
        self.assertTrue(apollo.should_use_accessibility("Finder", 'Click "Downloads" in the sidebar'))

    def test_should_use_accessibility_avoids_vscode_and_spotify(self):
        self.assertFalse(apollo.should_use_accessibility("Visual Studio Code", 'Click the Explorer button'))
        self.assertFalse(apollo.should_use_accessibility("Spotify", 'Click the play button'))

    def test_query_ax_element_clamps_depth_and_results(self):
        completed = mock.Mock(
            returncode=0,
            stdout='{"status":"ok","matches":[],"visited":0}',
            stderr="",
        )
        with mock.patch.object(apollo.subprocess, "run", return_value=completed) as run_mock:
            result = apollo.query_ax_element(
                "Finder",
                target_label="Downloads",
                target_role="button",
                max_depth=999,
                max_results=999,
            )

        self.assertEqual(result, [])
        command = run_mock.call_args.args[0]
        self.assertEqual(command[-3], str(apollo.AX_QUERY_MAX_DEPTH))
        self.assertEqual(command[-2], str(apollo.AX_QUERY_MAX_RESULTS))
        self.assertEqual(command[-1], str(apollo.AX_QUERY_MAX_CHILDREN))

    def test_query_ax_element_returns_none_on_timeout(self):
        with mock.patch.object(
            apollo.subprocess,
            "run",
            side_effect=apollo.subprocess.TimeoutExpired(cmd=["osascript"], timeout=apollo.AX_QUERY_TIMEOUT_SECONDS),
        ):
            result = apollo.query_ax_element("Finder", target_label="Downloads")

        self.assertIsNone(result)

    def test_resolve_ui_target_ax_hit_skips_gemini(self):
        ax_match = {
            "label": "Downloads",
            "role": "AXRow",
            "center_x": 64,
            "center_y": 128,
        }
        with mock.patch.object(apollo, "ax_get_window_count", return_value=1):
            with mock.patch.object(apollo, "ax_get_focused_element_value", return_value=""):
                with mock.patch.object(apollo, "query_ax_element", return_value=[ax_match]):
                    with mock.patch.object(apollo, "capture_vision_frame") as capture_vision_frame:
                        with mock.patch.object(apollo, "request_vision_action") as request_vision_action:
                            result = apollo.resolve_ui_target(
                                "Finder",
                                'Click the sidebar item labeled "Downloads"',
                                "open finder",
                            )

        self.assertEqual(result["source"], "ax")
        self.assertEqual(result["action"]["confidence"], 1.0)
        capture_vision_frame.assert_not_called()
        request_vision_action.assert_not_called()

    def test_resolve_ui_target_ax_miss_falls_back_to_vision(self):
        metadata = {
            "region_x": 0.0,
            "region_y": 0.0,
            "logical_width": 100.0,
            "logical_height": 100.0,
            "pixel_width": 200,
            "pixel_height": 200,
            "scale_x": 2.0,
            "scale_y": 2.0,
            "coordinate_space": "image",
            "screenshot_path": None,
            "screenshot_region": None,
        }
        action = {"action": "click", "x": 10, "y": 20, "description": "Click Downloads"}
        with mock.patch.object(apollo, "ax_get_window_count", return_value=1):
            with mock.patch.object(apollo, "ax_get_focused_element_value", return_value=""):
                with mock.patch.object(apollo, "query_ax_element", return_value=[]):
                    with mock.patch.object(apollo, "capture_vision_frame", return_value=(b"image", metadata)) as capture:
                        with mock.patch.object(apollo, "request_vision_action", return_value=action) as request:
                            result = apollo.resolve_ui_target(
                                "Finder",
                                'Click the sidebar item labeled "Downloads"',
                                "open finder",
                            )

        self.assertEqual(result["source"], "vision")
        capture.assert_called_once()
        request.assert_called_once()

    def test_execute_vision_task_click_verification_success(self):
        resolution = {
            "source": "vision",
            "target_app": "Finder",
            "action": {
                "action": "click",
                "x": 10,
                "y": 20,
                "description": "Click Downloads",
                "expected_postcondition": 'The "Downloads" view is visible',
            },
            "metadata": {"coordinate_space": "global"},
        }
        with mock.patch.object(apollo, "resolve_ui_target", return_value=resolution):
            with mock.patch.object(apollo, "verify_postcondition", return_value={
                "status": "verified",
                "reason": "verified by AX",
                "method": "ax",
                "used_gemini": False,
            }):
                with mock.patch.object(apollo, "say"):
                    with mock.patch.object(apollo, "log_command"):
                        with mock.patch.object(apollo.time, "sleep"):
                            with mock.patch.object(apollo, "click_at") as click_at:
                                handled = apollo.execute_vision_task("Click Downloads", "click downloads")

        self.assertTrue(handled)
        click_at.assert_called_once_with(10, 20)

    def test_execute_vision_task_click_verification_retry_then_success(self):
        resolution = {
            "source": "vision",
            "target_app": "Finder",
            "action": {
                "action": "click",
                "x": 10,
                "y": 20,
                "description": "Click Downloads",
                "expected_postcondition": 'The "Downloads" view is visible',
            },
            "metadata": {"coordinate_space": "global"},
        }
        with mock.patch.object(apollo, "resolve_ui_target", side_effect=[resolution, resolution]) as resolve_ui_target:
            with mock.patch.object(apollo, "verify_postcondition", side_effect=[
                {"status": "unverified", "reason": "not yet", "method": "ax", "used_gemini": False},
                {"status": "verified", "reason": "done", "method": "ax", "used_gemini": False},
            ]):
                with mock.patch.object(apollo, "say"):
                    with mock.patch.object(apollo, "log_command"):
                        with mock.patch.object(apollo.time, "sleep"):
                            with mock.patch.object(apollo, "click_at") as click_at:
                                handled = apollo.execute_vision_task("Click Downloads", "click downloads")

        self.assertTrue(handled)
        self.assertEqual(resolve_ui_target.call_count, 2)
        self.assertEqual(click_at.call_count, 2)

    def test_click_verification_failure_returns_false_and_workflow_fails(self):
        resolution = {
            "source": "vision",
            "target_app": "Finder",
            "action": {
                "action": "click",
                "x": 10,
                "y": 20,
                "description": "Click Downloads",
                "expected_postcondition": 'The "Downloads" view is visible',
            },
            "metadata": {"coordinate_space": "global"},
        }
        workflow = {
            "description": "Clicking Downloads",
            "steps": [
                {"type": "vision", "task": "Click Downloads", "reason": "open downloads"},
            ],
        }
        with mock.patch.object(apollo, "resolve_ui_target", side_effect=[resolution, resolution]):
            with mock.patch.object(apollo, "verify_postcondition", side_effect=[
                {"status": "unverified", "reason": "still closed", "method": "ax", "used_gemini": False},
                {"status": "unverified", "reason": "still closed", "method": "ax", "used_gemini": False},
            ]):
                with mock.patch.object(apollo, "say"):
                    with mock.patch.object(apollo, "log_command"):
                        with mock.patch.object(apollo.time, "sleep"):
                            with mock.patch.object(apollo, "click_at"):
                                ok, details = apollo.execute_workflow_once(workflow, "click downloads")

        self.assertFalse(ok)
        self.assertEqual(details["reason"], "step_returned_false")


if __name__ == "__main__":
    unittest.main()
