"""Join represented cleanup plateaus and select a first durable interval.

This module reads no files, masks, reference contours or KDE results. A stream
is the ``states`` list emitted by sweep_uniform_alpha_epsilon_events.py. Mask
SHA256 values identify complete output geometry, so equal Betti counts alone
never cause intervals to merge. Stream order is preserved in the returned
representative states (for example, target level then raster resolution).

The observed domain is closed. Actual change boundaries are excluded on the
right; the final state includes the observed endpoint but makes no claim beyond
it. A change exactly at the final endpoint is retained as a zero-width state.
"""
from __future__ import annotations
from fractions import Fraction
import math


def _finite(value,name):
    if isinstance(value,bool):raise ValueError(name+' must be a finite number')
    try:value=float(value)
    except (ValueError,TypeError,OverflowError) as error:
        raise ValueError(name+' must be a finite number') from error
    if not math.isfinite(value):raise ValueError(name+' must be finite')
    return value


def _validate_stream(states):
    if not isinstance(states,(list,tuple)) or not states:
        raise ValueError('Each stream must be a nonempty state sequence')
    previous_right=None
    for index,state in enumerate(states):
        left=_finite(state['left_inclusive_k'],'left endpoint')
        observed=_finite(state['observed_interval_right_k'],'observed endpoint')
        censored=state['right_censored']
        if not isinstance(censored,bool):raise ValueError('right_censored must be Boolean')
        if left<1 or observed<left:raise ValueError('Invalid ratio interval')
        if not isinstance(state['mask_sha256'],str) or not state['mask_sha256']:
            raise ValueError('Each state needs its complete-mask signature')
        if index and left!=previous_right:
            raise ValueError('Stream states must be ordered, contiguous and nonoverlapping')
        if censored:
            if index!=len(states)-1 or state['right_exclusive_k'] is not None:
                raise ValueError('Only the final state may be right-censored, with no asserted right boundary')
            previous_right=observed
        else:
            right=_finite(state['right_exclusive_k'],'right endpoint')
            if not left<right or observed!=right:
                raise ValueError('An uncensored interval must have positive width and its observed change boundary')
            if index==len(states)-1:raise ValueError('A stream must include its final observed point')
            previous_right=right
    return float(states[0]['left_inclusive_k']),float(states[-1]['observed_interval_right_k'])


def joint_intervals(streams):
    """Merge the union of all stream changes into maximal joint-mask plateaus.

    All streams must cover exactly the same closed observed ratio domain.
    Returned ``states`` are the source dictionaries representing the left end
    of each joint plateau, in input stream order. No source dictionary changes.
    Their mask metrics remain valid if redundant later signatures are merged.
    """
    if not isinstance(streams,(list,tuple)):raise ValueError('Use a sequence of streams')
    if not streams:return []
    domains=[_validate_stream(s) for s in streams]
    if any(d!=domains[0] for d in domains[1:]):
        raise ValueError('Streams must share one closed observed ratio domain')
    start,end=domains[0]
    boundaries=sorted({float(s['left_inclusive_k']) for stream in streams for s in stream}|{end})
    indices=[0]*len(streams);result=[];previous_signature=None
    for index,left in enumerate(boundaries):
        representatives=[]
        for stream_index,stream in enumerate(streams):
            position=indices[stream_index]
            while position+1<len(stream) and float(stream[position+1]['left_inclusive_k'])<=left:
                position+=1
            indices[stream_index]=position;representatives.append(stream[position])
        signature=tuple(s['mask_sha256'] for s in representatives)
        right=boundaries[index+1] if index+1<len(boundaries) else None
        censored=right is None;observed=end if censored else right
        if result and signature==previous_signature:
            result[-1].update(right_exclusive_k=right,right_censored=censored,
                observed_interval_right_k=observed)
        else:
            result.append(dict(left_inclusive_k=left,right_exclusive_k=right,
                right_censored=censored,observed_interval_right_k=observed,
                states=representatives))
        previous_signature=signature
    return result


def select_first(intervals,span_factor,admissible_callback):
    """Return the first admissible left onset supporting a CLOSED ratio span.

    At candidate k, the requested interval is [k, span_factor*k]. It must lie
    strictly below an actual change boundary, or at/below the final OBSERVED
    endpoint of a censored plateau. No extrapolation beyond that endpoint is
    permitted. Only plateau left onsets are candidates; no intermediate point
    can qualify if its earlier onset lacks the required multiplicative span.

    Endpoint products are compared as exact rationals of the provided Float64
    values, avoiding accidental acceptance/rejection from product rounding.
    The callback supplies scientific admissibility, such as grid agreement,
    nonempty regions and nesting. This function imposes no biological criterion.
    """
    span=_finite(span_factor,'span factor')
    if span<=1:raise ValueError('The span factor must exceed one')
    if not callable(admissible_callback):raise TypeError('Use an admissibility callback')
    previous_left=-math.inf
    for interval in intervals:
        left=_finite(interval['left_inclusive_k'],'left endpoint')
        observed=_finite(interval['observed_interval_right_k'],'observed endpoint')
        if left<1 or left<previous_left or observed<left:
            raise ValueError('Joint intervals must be ordered with valid ratio bounds')
        previous_left=left
        if observed==left:continue
        closed_end=Fraction.from_float(left)*Fraction.from_float(span)
        if interval['right_censored']:
            if interval['right_exclusive_k'] is not None:
                raise ValueError('A censored interval must not assert a true right boundary')
            wide_enough=closed_end<=Fraction.from_float(observed)
        else:
            right=_finite(interval['right_exclusive_k'],'right endpoint')
            if right!=observed or right<=left:raise ValueError('Invalid actual change boundary')
            wide_enough=closed_end<Fraction.from_float(right)
        if wide_enough and admissible_callback(interval):return interval
    return None
