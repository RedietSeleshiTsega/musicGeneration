
"""Verify the history supplied to corpus-backed priors."""

from dataclasses import replace

from aimusic.core.core_types import BeatState, Layer
from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.planning.graph import _build_prior_context
from aimusic.scoring.priors import PriorContext


def make_state(chord="Cmaj"):
    vocab = DEFAULT_VOCABULARIES

    return BeatState(
        meter_id=vocab.meters.token_for_label("4/4").id,
        beat_in_bar=0,
        boundary_lvl=0,
        key_id=vocab.keys.token_for_label("C").id,
        chord_id=vocab.chords.token_for_label(chord).id,
        role_id=vocab.roles.token_for_label("hold").id,
        head_id=vocab.heads.token_for_label("root").id,
        groove_id=vocab.grooves.token_for_label(
            "straight_8ths"
        ).id,
    )


def test_graph_context_contains_source_state():
    source = make_state("Fmaj")
    target = make_state("G7")

    end_layer = Layer(
        time_index=2,
        states=(target,),
    )

    context = _build_prior_context(
        source, end_layer, time_index=1
    )

    assert isinstance(context, PriorContext)
    assert context.history == (source,)
    assert len(context.history_tokens) == 1


def test_second_order_history_requires_predecessor():
    """A true trigram needs two preceding musical states."""

    earlier = make_state("Cmaj")
    previous = make_state("Fmaj")

    # The context needed by a trigram:
    expected_history = (earlier, previous)

    assert len(expected_history) == 2

    # The graph's current helper only receives
    # the source state, so it cannot reconstruct
    # the earlier state.
    end_layer = Layer(
        time_index=2,
        states=(make_state("G7"),),
    )

    actual = _build_prior_context(
        previous, end_layer, time_index=1
    )

    assert actual.history == (previous,)
    assert actual.history != expected_history


def test_different_paths_can_share_same_source():
    """One graph node may have multiple possible histories."""

    earlier_a = make_state("Cmaj")
    earlier_b = make_state("Dmaj")
    previous = make_state("Fmaj")

    path_a = (earlier_a, previous)
    path_b = (earlier_b, previous)

    assert path_a != path_b
    assert path_a[-1] == path_b[-1]
