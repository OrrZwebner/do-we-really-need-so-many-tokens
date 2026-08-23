"""The agent loop, its tools, and the grounding checks."""

import pytest

from fake_client import (
    FakeClient,
    FinalMessage,
    StopDetails,
    TextBlock,
    Usage,
    text_turn,
    tool_turn,
)

from nelson_rag.agent.tools import TOOLS, ToolRunner
from nelson_rag.agent.verify import check_answer, extract_dose_claims
from nelson_rag.indexing.store import Hit


def make_hit(text: str, chapter: str = "220", page: int = 400) -> Hit:
    return Hit(
        chunk_id=f"c{page}",
        text=text,
        citation=f"Nelson, Ch. {chapter}, p. {page}",
        chapter_number=chapter,
        chapter_title="Acute Otitis Media",
        section_title="TREATMENT",
        page_start=page,
        page_end=page,
        source_file="nelson.pdf",
    )


# --------------------------------------------------------------------------
# Tool schemas and dispatch
# --------------------------------------------------------------------------


class TestToolSchemas:
    def test_all_tools_satisfy_strict_mode(self):
        for tool in TOOLS:
            schema = tool["input_schema"]
            assert tool["strict"] is True
            assert schema["additionalProperties"] is False
            # strict mode requires every property to be listed as required
            assert set(schema["required"]) == set(schema["properties"])

    def test_search_is_described_as_the_primary_tool(self):
        assert "before answering" in TOOLS[0]["description"]


class TestToolDispatch:
    def test_search_returns_numbered_passages_with_citations(self, session):
        runner = ToolRunner(session.retriever)
        out = runner.run("search_nelson", {"query": "otitis media", "top_k": 3})
        assert "[1]" in out and "Ch. 220" in out

    def test_search_records_hits_for_later_verification(self, session):
        runner = ToolRunner(session.retriever)
        runner.run("search_nelson", {"query": "febrile seizure", "top_k": 3})
        assert runner.retrieved

    def test_repeated_searches_do_not_duplicate_recorded_hits(self, session):
        runner = ToolRunner(session.retriever)
        runner.run("search_nelson", {"query": "febrile seizure", "top_k": 3})
        first = len(runner.retrieved)
        runner.run("search_nelson", {"query": "febrile seizure", "top_k": 3})
        assert len(runner.retrieved) == first

    def test_empty_query_is_an_error_not_a_crash(self, session):
        assert ToolRunner(session.retriever).run(
            "search_nelson", {"query": "  ", "top_k": 3}
        ).startswith("ERROR")

    def test_unknown_tool_is_reported_to_the_model(self, session):
        assert "unknown tool" in ToolRunner(session.retriever).run("nope", {})

    def test_out_of_range_top_k_is_clamped_not_rejected(self, session):
        out = ToolRunner(session.retriever).run(
            "search_nelson", {"query": "otitis", "top_k": 9999}
        )
        assert not out.startswith("ERROR")

    def test_list_chapters_filters_by_title(self, session):
        out = ToolRunner(session.retriever).run(
            "list_nelson_chapters", {"contains": "Otitis", "limit": 10}
        )
        assert "Acute Otitis Media" in out and "Seizures" not in out

    def test_read_chapter_returns_outline_and_passages(self, session):
        out = ToolRunner(session.retriever).run(
            "read_nelson_chapter", {"chapter_number": "611", "max_passages": 5}
        )
        assert "Sections:" in out and "TREATMENT" in out

    def test_missing_chapter_tells_the_model_how_to_recover(self, session):
        out = ToolRunner(session.retriever).run(
            "read_nelson_chapter", {"chapter_number": "9999", "max_passages": 5}
        )
        assert "list_nelson_chapters" in out


# --------------------------------------------------------------------------
# Grounding checks
# --------------------------------------------------------------------------


class TestDoseExtraction:
    def test_finds_per_kg_and_absolute_doses(self):
        claims = {c.normalised for c in extract_dose_claims(
            "Give 15 mg/kg PO q6h, max 750 mg."
        )}
        assert "15mg/kg" in claims and "750mg" in claims

    def test_finds_dose_ranges(self):
        assert "6-8mg/kg" in {
            c.normalised for c in extract_dose_claims("infuse 6-8 mg/kg/min")
        }

    def test_patient_weight_is_not_treated_as_a_dose(self):
        assert not [c for c in extract_dose_claims("the child weighs 12 kg")]

    def test_units_are_normalised(self):
        a = extract_dose_claims("50 µg/kg")[0].normalised
        b = extract_dose_claims("50 mcg/kg")[0].normalised
        assert a == b

    def test_decimal_forms_compare_equal(self):
        assert (
            extract_dose_claims("0.50 mg/kg")[0].normalised
            == extract_dose_claims("0.5 mg/kg")[0].normalised
        )


class TestGroundingReport:
    SOURCE = make_hit("Amoxicillin 90 mg/kg/day divided every 12 hours, max 4 g per day.")

    def test_quoted_dose_is_grounded(self):
        report = check_answer("Give 90 mg/kg/day [1].", [self.SOURCE])
        assert not report.ungrounded_doses
        assert report.clean

    def test_invented_dose_is_flagged(self):
        report = check_answer("Give 200 mg/kg/day [1].", [self.SOURCE])
        assert "200 mg/kg" in report.ungrounded_doses[0]
        assert not report.clean

    def test_invented_chapter_is_flagged(self):
        report = check_answer("See Ch. 999.", [self.SOURCE])
        assert report.ungrounded_chapters == ["Ch. 999"]

    def test_invented_page_is_flagged(self):
        report = check_answer("See p. 9999.", [self.SOURCE])
        assert report.ungrounded_pages == ["p. 9999"]

    def test_cited_page_inside_a_retrieved_range_passes(self):
        hit = make_hit("text", page=400)
        hit.page_end = 405
        assert not check_answer("See p. 402.", [hit]).ungrounded_pages

    def test_answering_without_searching_is_flagged(self):
        report = check_answer("Amoxicillin is first line.", [], searched=False)
        assert not report.clean
        assert "WITHOUT searching" in report.warning_text()

    def test_warning_text_is_none_when_everything_checks_out(self):
        assert check_answer("Give 90 mg/kg/day [1].", [self.SOURCE]).warning_text() is None


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


class TestAgentLoop:
    def test_search_then_answer(self, session):
        client = FakeClient([
            tool_turn("search_nelson", {"query": "acute otitis media amoxicillin", "top_k": 5}),
            text_turn("High-dose amoxicillin 90 mg/kg/day [1].\nSources: [1] Ch. 220"),
        ])
        answer = session.agent(client=client).ask("What is first-line therapy for AOM?")
        assert "90 mg/kg/day" in answer.text
        assert answer.hits, "the retrieved passages should be recorded"
        assert answer.tool_calls[0]["tool"] == "search_nelson"
        assert answer.grounding.searched

    def test_multiple_search_rounds_are_all_recorded(self, session):
        client = FakeClient([
            tool_turn("search_nelson", {"query": "febrile seizure diagnosis", "top_k": 3}, "tu_1"),
            tool_turn("search_nelson", {"query": "febrile seizure treatment", "top_k": 3}, "tu_2"),
            text_turn("Answer with both."),
        ])
        answer = session.agent(client=client).ask("Work up and treat a febrile seizure.")
        assert len(answer.tool_calls) == 2

    def test_answering_without_a_search_is_marked_unsourced(self, session):
        client = FakeClient([text_turn("Amoxicillin, obviously.")])
        answer = session.agent(client=client).ask("First line for AOM?")
        assert not answer.grounding.searched
        assert "WITHOUT searching" in answer.grounding.warning_text()

    def test_browsing_chapters_alone_does_not_count_as_searching(self, session):
        client = FakeClient([
            tool_turn("list_nelson_chapters", {"contains": None, "limit": 10}),
            text_turn("Nelson covers otitis media."),
        ])
        answer = session.agent(client=client).ask("What does the book cover?")
        assert not answer.grounding.searched

    def test_tool_round_cap_is_enforced(self, session):
        from dataclasses import replace

        capped = replace(session.settings, max_tool_rounds=2)
        session.retriever.settings = capped
        client = FakeClient([
            tool_turn("search_nelson", {"query": "x", "top_k": 3}, f"tu_{i}")
            for i in range(5)
        ])
        from nelson_rag.agent.loop import NelsonAgent

        warnings: list[str] = []
        agent = NelsonAgent(capped, session.retriever, client=client)
        agent.ask("loop forever", on_event=lambda k, t: warnings.append(t) if k == "warn" else None)
        assert len(client.requests) == 2
        assert any("tool rounds" in w for w in warnings)

    def test_usage_is_summed_across_rounds(self, session):
        client = FakeClient([
            tool_turn("search_nelson", {"query": "otitis", "top_k": 3}),
            text_turn("Done.", usage=Usage(input_tokens=200, output_tokens=80)),
        ])
        answer = session.agent(client=client).ask("q")
        assert answer.usage["input_tokens"] == 300
        assert answer.usage["output_tokens"] == 130

    def test_refusal_is_surfaced_not_swallowed(self, session):
        client = FakeClient([
            FinalMessage(
                content=[TextBlock("")],
                stop_reason="refusal",
                stop_details=StopDetails(explanation="declined by classifier"),
            )
        ])
        answer = session.agent(client=client).ask("something")
        assert answer.refusal == "declined by classifier"

    def test_falls_back_to_the_plain_endpoint_when_the_beta_is_rejected(self, session):
        client = FakeClient([text_turn("Fine.")], reject_beta=True)
        answer = session.agent(client=client).ask("q")
        assert answer.text == "Fine."

    def test_a_refusal_does_not_corrupt_the_history(self, session):
        client = FakeClient([
            FinalMessage(
                content=[TextBlock("")],
                stop_reason="refusal",
                stop_details=StopDetails(explanation="nope"),
            ),
            text_turn("Second question works."),
        ])
        agent = session.agent(client=client)
        agent.ask("refused question")
        answer = agent.ask("ordinary question")
        assert answer.text == "Second question works."
        roles = [m["role"] for m in client.requests[1]["messages"]]
        assert roles == ["user"], "the refused turn should have been rewound"

    def test_a_failed_request_does_not_corrupt_the_history(self, session):
        client = FakeClient([])  # no scripted turns -> raises
        agent = session.agent(client=client)
        with pytest.raises(AssertionError):
            agent.ask("boom")
        assert agent.messages == []

    def test_sources_block_is_not_numbered(self, session):
        """Numbering it would imply a correspondence with the model's [n]."""
        client = FakeClient([
            tool_turn("search_nelson", {"query": "otitis", "top_k": 3}),
            text_turn("See [1]."),
        ])
        block = session.agent(client=client).ask("q").sources_block()
        assert "[1]" not in block and "Nelson" in block

    def test_grounding_runs_against_what_was_actually_retrieved(self, session):
        client = FakeClient([
            tool_turn("search_nelson", {"query": "acute otitis media", "top_k": 5}),
            text_turn("Give 12345 mg/kg/day of amoxicillin."),
        ])
        answer = session.agent(client=client).ask("dose?")
        assert answer.grounding.ungrounded_doses


class TestPromptConstruction:
    def _first_request(self, session, question="q"):
        client = FakeClient([text_turn("ok")])
        session.agent(client=client).ask(question)
        return client.requests[0]

    def test_system_prompt_is_cached(self, session):
        system = self._first_request(session)["system"]
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_system_prompt_names_the_indexed_edition(self, session):
        assert "synthetic test fixture" in self._first_request(session)["system"][0]["text"]

    def test_adaptive_thinking_and_effort_are_set(self, session):
        request = self._first_request(session)
        assert request["thinking"]["type"] == "adaptive"
        assert request["output_config"]["effort"] in {
            "low", "medium", "high", "xhigh", "max"
        }

    def test_no_deprecated_budget_tokens(self, session):
        assert "budget_tokens" not in self._first_request(session)["thinking"]

    def test_cached_prefix_is_byte_identical_across_turns(self, session):
        """A changing system prompt would silently destroy the cache hit rate."""
        client = FakeClient([text_turn("one"), text_turn("two")])
        agent = session.agent(client=client)
        agent.ask("first")
        agent.ask("second")
        a, b = client.requests
        assert a["system"] == b["system"]
        assert a["tools"] == b["tools"]

    def test_history_is_carried_between_turns(self, session):
        client = FakeClient([text_turn("one"), text_turn("two")])
        agent = session.agent(client=client)
        agent.ask("first")
        agent.ask("second")
        roles = [m["role"] for m in client.requests[1]["messages"]]
        assert roles == ["user", "assistant", "user"]

    def test_reset_clears_history(self, session):
        client = FakeClient([text_turn("one"), text_turn("two")])
        agent = session.agent(client=client)
        agent.ask("first")
        agent.reset()
        agent.ask("second")
        assert len(client.requests[1]["messages"]) == 1
