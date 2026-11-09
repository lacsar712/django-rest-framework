"""
Request-scoped runtime policy stack.

This module provides a push/pop stack of :class:`PolicyFrame` objects that is
bound to the *current execution context* (a :mod:`contextvars` context) rather
than to any ``Request`` / ``Response`` / serializer / view instance.

Each frame carries three independent gates:

* ``parse_admit``    - ``honor_cfg`` | ``deny_new`` | ``one_more``
* ``represent_gate`` - ``honor_cfg`` | ``deny_new`` | ``force_fail``
* ``render_gate``    - ``honor_cfg`` | ``deny_new`` | ``force_close``

Semantics
---------

* With an empty stack every code path behaves exactly as it always did; the
  framework's parser / renderer / permission / throttle configuration is never
  rewritten.
* The *current top frame* decides. Nested ``get_serializer`` callbacks,
  streaming parser reads and nested ``@action`` invocations always re-read the
  then-current top frame, they never reuse a snapshot taken when an outer frame
  was entered.
* Denials are *observable*: they raise dedicated exceptions (``...Denied`` /
  ``RepresentationForceFailed``) and are also recorded on the frame itself.
* ``deny_new`` only constrains operations that start afterwards. An operation
  already admitted and in flight is never cancelled and never rewritten into
  an empty success.
* ``render_gate=force_close`` closes resources (upload streams, streaming
  response iterators) that were opened while the frame was active and are
  still open when the frame pops.

Isolation
---------

The stack lives in a :class:`contextvars.ContextVar`. Two concurrent requests
running in separate threads (or ``asyncio`` tasks) therefore observe two
independent stacks and can never trample each other.  ``APIView.dispatch``
opens a fresh :func:`request_scope` per request unless an outer scope (for
example one entered by middleware) already owns the current context.

Nothing is ever attached to ``Request``, ``Response``, serializer or view
instances - the stack is purely ambient, so nested callbacks (including
``get_serializer`` callbacks invoked deep inside a view) see it automatically.
"""
from contextlib import contextmanager
import contextvars

from django.utils.translation import gettext_lazy as _

from rest_framework.exceptions import APIException

# Gate modes -----------------------------------------------------------------

PARSE_HONOR_CFG = 'honor_cfg'
PARSE_DENY_NEW = 'deny_new'
PARSE_ONE_MORE = 'one_more'

REPRESENT_HONOR_CFG = 'honor_cfg'
REPRESENT_DENY_NEW = 'deny_new'
REPRESENT_FORCE_FAIL = 'force_fail'

RENDER_HONOR_CFG = 'honor_cfg'
RENDER_DENY_NEW = 'deny_new'
RENDER_FORCE_CLOSE = 'force_close'

_PARSE_MODES = frozenset({PARSE_HONOR_CFG, PARSE_DENY_NEW, PARSE_ONE_MORE})
_REPRESENT_MODES = frozenset({
    REPRESENT_HONOR_CFG, REPRESENT_DENY_NEW, REPRESENT_FORCE_FAIL
})
_RENDER_MODES = frozenset({RENDER_HONOR_CFG, RENDER_DENY_NEW, RENDER_FORCE_CLOSE})


# Exceptions -----------------------------------------------------------------

class PolicyGateError(Exception):
    """
    Base class for every failure surfaced by the runtime policy stack.

    ``.gate`` names the gate that fired (``'parse_admit'``,
    ``'represent_gate'`` or ``'render_gate'``) and ``.mode`` is the frame mode
    that caused the rejection.
    """
    gate = None
    mode = None

    def __init__(self, mode=None, detail=None, code=None):
        self.mode = mode
        super().__init__(detail, code)


class ParseAdmissionDenied(PolicyGateError, APIException):
    """
    Raised when a new request body parse (or a new streaming upload read) is
    attempted while the current frame forbids it.
    """
    status_code = 403
    default_detail = _(
        'Opening a new request body parse is not permitted in this '
        'request policy frame.'
    )
    default_code = 'parse_admission_denied'
    gate = 'parse_admit'

    def __init__(self, mode=PARSE_DENY_NEW, detail=None, code=None):
        super().__init__(mode, detail, code)


class RepresentAdmissionDenied(PolicyGateError, APIException):
    """
    Raised when a new validation / field check / ``create`` / ``update``
    write-out is attempted while the current frame forbids it.
    """
    status_code = 403
    default_detail = _(
        'Running a new representation validation or write-out is not '
        'permitted in this request policy frame.'
    )
    default_code = 'represent_admission_denied'
    gate = 'represent_gate'

    def __init__(self, mode=REPRESENT_DENY_NEW, detail=None, code=None):
        super().__init__(mode, detail, code)


class RepresentationForceFailed(PolicyGateError, APIException):
    """
    Raised in place of a ``create`` / ``update`` write-out when the current
    frame demands that every write-out takes the failure surface.

    The caller keeps every Python object it already held; nothing is discarded
    and no successful validation is fabricated.
    """
    status_code = 403
    default_detail = _(
        'Representation write-outs are forced to fail in this request '
        'policy frame.'
    )
    default_code = 'representation_force_failed'
    gate = 'represent_gate'

    def __init__(self, mode=REPRESENT_FORCE_FAIL, detail=None, code=None):
        super().__init__(mode, detail, code)


class RenderAdmissionDenied(PolicyGateError):
    """
    Raised when a new response render (or streaming content write) is
    attempted while the current frame forbids it.

    Rendering happens after the view returned, outside the DRF exception
    handling path, so this is deliberately *not* an ``APIException``: it must
    propagate as an explicit failure instead of being marked rendered.
    """
    def __init__(self, mode=RENDER_DENY_NEW):
        self.mode = mode
        super().__init__(
            'Rendering a new response is not permitted in this request '
            'policy frame.'
        )
    gate = 'render_gate'


# Stack primitives -----------------------------------------------------------

class PolicyFrame:
    """
    One stack frame. A frame always carries all three gates; gates that are
    not explicitly set default to ``honor_cfg`` (i.e. the frame has no
    opinion and the then-current configuration keeps applying).
    """

    def __init__(self, parse_admit=PARSE_HONOR_CFG,
                 represent_gate=REPRESENT_HONOR_CFG,
                 render_gate=RENDER_HONOR_CFG):
        if parse_admit not in _PARSE_MODES:
            raise ValueError('invalid parse_admit: %r' % (parse_admit,))
        if represent_gate not in _REPRESENT_MODES:
            raise ValueError('invalid represent_gate: %r' % (represent_gate,))
        if render_gate not in _RENDER_MODES:
            raise ValueError('invalid render_gate: %r' % (render_gate,))

        self.parse_admit = parse_admit
        self.represent_gate = represent_gate
        self.render_gate = render_gate

        # ``one_more`` allows exactly one further *complete parse entry*.
        self.parse_quota = 1 if parse_admit == PARSE_ONE_MORE else 0

        # Resources registered against this frame (upload streams, streaming
        # response iterators, ...), as ``(resource, kind)`` tuples.
        self.resources = []
        # Resources that force-close actually had to close on pop.
        self.closed_resources = []
        # Observable rejection log: ``(gate, mode, where)`` tuples.
        self.rejections = []

    def __repr__(self):
        return (
            'PolicyFrame(parse_admit=%r, represent_gate=%r, render_gate=%r)'
            % (self.parse_admit, self.represent_gate, self.render_gate)
        )


class _ScopeState:
    """
    Mutable state for one request scope. Stored in the ContextVar, never on
    any framework instance.
    """
    __slots__ = ('frames', 'in_flight_parses', 'in_flight_validations',
                 'owner', 'token')

    def __init__(self, owner='ambient'):
        self.frames = []
        # Depth counters for operations running inside this scope. They are
        # tracked even with an empty stack, because a nested callback may push
        # the first frame while an operation is already in flight: entries
        # reached from within that operation must not be mistaken for new
        # operations.
        self.in_flight_parses = 0
        self.in_flight_validations = 0
        # 'scope'   -> owned by request_scope(), which resets it on exit
        # 'ambient' -> created by a bare push / operation entry; reset as soon
        #              as every frame is popped and every operation finished
        self.owner = owner
        self.token = None


_state = contextvars.ContextVar(
    'rest_framework_runtime_policy_state', default=None
)


def _ensure_state():
    """
    Return the current scope state, creating an ambient one if needed.

    Admission hooks always create state (even with an empty stack) so that
    frames pushed mid-operation by nested callbacks remain observable. The
    ambient state resets itself automatically once the stack is empty and no
    in-flight operation remains, leaving an unpushed request equivalent to a
    request for which this module never ran.
    """
    state = _state.get()
    if state is None:
        state = _ScopeState()
        state.token = _state.set(state)
    return state


def _maybe_reset(state):
    """Reset an ambient state once it is fully empty and balanced again."""
    if (
        state.owner != 'scope'
        and not state.frames
        and state.in_flight_parses == 0
        and state.in_flight_validations == 0
        and state.token is not None
    ):
        token = state.token
        state.token = None
        _state.reset(token)


def _top_frame(state):
    return state.frames[-1] if state.frames else None


def current_frame():
    """Return the current top frame, or ``None`` when the stack is empty."""
    state = _state.get()
    if state is None:
        return None
    return _top_frame(state)


def current_stack():
    """
    Return a snapshot tuple of the frames currently pushed, outermost first.
    """
    state = _state.get()
    if state is None:
        return ()
    return tuple(state.frames)


def active_modes():
    """
    Return ``{'parse_admit': ..., 'represent_gate': ..., 'render_gate': ...}``
    for the current top frame. With an empty stack every gate is ``honor_cfg``.
    """
    frame = current_frame()
    if frame is None:
        return {
            'parse_admit': PARSE_HONOR_CFG,
            'represent_gate': REPRESENT_HONOR_CFG,
            'render_gate': RENDER_HONOR_CFG,
        }
    return {
        'parse_admit': frame.parse_admit,
        'represent_gate': frame.represent_gate,
        'render_gate': frame.render_gate,
    }


# Push / pop -----------------------------------------------------------------

def push_policy_frame(*, parse_admit=PARSE_HONOR_CFG,
                      represent_gate=REPRESENT_HONOR_CFG,
                      render_gate=RENDER_HONOR_CFG):
    """
    Push a frame onto the current request's stack and return it.

    Prefer the :func:`push_policy` context manager; use this low-level pair
    only when a frame's lifetime does not line up with a Python block.
    """
    state = _ensure_state()
    frame = PolicyFrame(
        parse_admit=parse_admit,
        represent_gate=represent_gate,
        render_gate=render_gate,
    )
    state.frames.append(frame)
    return frame


def _force_close_frame_resources(frame):
    resources = frame.resources
    frame.resources = []
    for resource, kind in resources:
        closed_attr = getattr(resource, 'closed', None)
        if closed_attr is True:
            # Already finished cleanly - nothing to observe.
            continue
        close = getattr(resource, 'close', None)
        if not callable(close):
            continue
        try:
            close()
        except Exception:
            # Closing must itself surface, but a broken closer must not mask
            # the frame-pop path; the resource still counts as handled.
            pass
        frame.closed_resources.append(resource)


def pop_policy_frame(frame):
    """
    Pop (and, for ``force_close``, clean up) a previously pushed frame.

    The frame must be the current top frame of the current scope. Popping a
    frame from another scope or out of order is a programming error.
    """
    state = _state.get()
    if state is None or frame not in state.frames:
        raise RuntimeError(
            'Cannot pop a runtime policy frame that is not part of the '
            'current request scope.'
        )
    if state.frames[-1] is not frame:
        raise RuntimeError(
            'Cannot pop a runtime policy frame before the frames pushed '
            'after it have been popped.'
        )

    state.frames.pop()

    if frame.render_gate == RENDER_FORCE_CLOSE:
        _force_close_frame_resources(frame)

    _maybe_reset(state)
    return frame


@contextmanager
def push_policy(*, parse_admit=PARSE_HONOR_CFG,
                represent_gate=REPRESENT_HONOR_CFG,
                render_gate=RENDER_HONOR_CFG):
    """
    Context manager pushing one frame. The frame is always popped again, even
    if the body raises.
    """
    frame = push_policy_frame(
        parse_admit=parse_admit,
        represent_gate=represent_gate,
        render_gate=render_gate,
    )
    try:
        yield frame
    finally:
        pop_policy_frame(frame)


@contextmanager
def push_parse_admit(mode):
    """Convenience wrapper setting only ``parse_admit``."""
    with push_policy(parse_admit=mode) as frame:
        yield frame


@contextmanager
def push_represent_gate(mode):
    """Convenience wrapper setting only ``represent_gate``."""
    with push_policy(represent_gate=mode) as frame:
        yield frame


@contextmanager
def push_render_gate(mode):
    """Convenience wrapper setting only ``render_gate``."""
    with push_policy(render_gate=mode) as frame:
        yield frame


@contextmanager
def request_scope():
    """
    Open a fresh, empty request policy scope for the current context.

    Used by ``APIView.dispatch`` so each request starts with a clean stack.
    If the current context already has a scope (middleware pushed a frame, or
    one API view dispatches into another), the existing scope is shared and
    left untouched on exit.
    """
    existing = _state.get()
    if existing is not None:
        # An outer scope already owns this context (middleware or a nested
        # dispatch): share it and leave its lifecycle to that owner.
        yield existing
        return

    state = _ScopeState(owner='scope')
    state.token = _state.set(state)
    try:
        yield state
    finally:
        # Exception-safe net: any frame still on the stack is popped here, so
        # resources get their force-close even when a low-level push was not
        # balanced by a matching pop.
        while state.frames:
            leftover = state.frames.pop()
            if leftover.render_gate == RENDER_FORCE_CLOSE:
                _force_close_frame_resources(leftover)
        if state.token is not None:
            token = state.token
            state.token = None
            _state.reset(token)


# Resource registration ------------------------------------------------------

def track_resource(resource, kind='resource'):
    """
    Register an open resource (upload stream, streaming response iterator,
    ...) against the current top frame.

    Registration is a no-op with an empty stack, and a resource is only ever
    closed automatically by a ``render_gate=force_close`` frame popping.
    """
    state = _state.get()
    if state is None or not state.frames:
        return resource
    _top_frame(state).resources.append((resource, kind))
    return resource


# Parse gate -----------------------------------------------------------------

@contextmanager
def admit_parse(parser, stream=None, *, track_stream=False):
    """
    Gate a ``parser.parse()`` entry.

    Called by every built-in parser. Either yields (empty stack /
    ``honor_cfg`` / admitted ``one_more`` / an entry reached from within an
    already admitted parse) or raises :class:`ParseAdmissionDenied` *before*
    any byte is read.
    """
    state = _ensure_state()

    if state.in_flight_parses == 0:
        # A new top-level parse entry: the *then-current* top frame decides.
        frame = _top_frame(state)
        if frame is not None:
            mode = frame.parse_admit
            if mode == PARSE_DENY_NEW:
                frame.rejections.append(('parse_admit', mode, 'parse'))
                _maybe_reset(state)
                raise ParseAdmissionDenied(mode)
            if mode == PARSE_ONE_MORE:
                # The quota is re-read on every entry; it is never cached from
                # a previous entry.
                if frame.parse_quota <= 0:
                    frame.rejections.append(('parse_admit', mode, 'parse'))
                    _maybe_reset(state)
                    raise ParseAdmissionDenied(mode)
                frame.parse_quota -= 1

    # Admitted (including nested entries belonging to an in-flight parse).
    state.in_flight_parses += 1
    if track_stream and stream is not None and hasattr(stream, 'close'):
        track_resource(stream, kind='upload_stream')
    try:
        yield
    finally:
        state.in_flight_parses -= 1
        _maybe_reset(state)


def guard_upload_read():
    """
    Gate a single streaming upload read block (used by ``FileUploadParser``
    around raw stream reads / chunk pulls).

    Reads performed inside an already admitted parse always proceed, even if a
    ``deny_new`` frame was pushed after that parse started. Any *new* read
    outside an in-flight parse is rejected observably.
    """
    state = _state.get()
    if state is None or not state.frames:
        return
    if state.in_flight_parses:
        return

    frame = _top_frame(state)
    mode = frame.parse_admit
    if mode != PARSE_HONOR_CFG:
        frame.rejections.append(('parse_admit', mode, 'stream_read'))
        raise ParseAdmissionDenied(mode)


# Representation gate --------------------------------------------------------

@contextmanager
def admit_validation(serializer=None):
    """
    Gate a serializer (or field) ``run_validation()`` entry.

    ``deny_new`` rejects new validation entries. Entries reached from within a
    validation that was already admitted (nested serializers, list item
    validations, individual fields) are part of the in-flight validation and
    proceed regardless of frames pushed later.

    ``force_fail`` does not reject validation: callers must keep receiving the
    real, validated Python objects. It only forces the write-outs to fail.
    """
    state = _ensure_state()

    if state.in_flight_validations == 0:
        # A new top-level validation entry: the then-current top frame
        # decides. Nested serializers, list items and individual fields run
        # while an outer validation is in flight and always proceed.
        frame = _top_frame(state)
        if frame is not None and frame.represent_gate == REPRESENT_DENY_NEW:
            frame.rejections.append(('represent_gate',
                                     frame.represent_gate, 'validation'))
            _maybe_reset(state)
            raise RepresentAdmissionDenied(frame.represent_gate)

    # ``force_fail`` does not reject validation: callers must keep receiving
    # the real, validated Python objects. It only forces write-outs to fail.
    state.in_flight_validations += 1
    try:
        yield
    finally:
        state.in_flight_validations -= 1
        _maybe_reset(state)


def guard_representation_write(kind='create'):
    """
    Gate a ``create`` / ``update`` write-out entry point.

    * ``deny_new``   -> :class:`RepresentAdmissionDenied`
    * ``force_fail`` -> :class:`RepresentationForceFailed`
    """
    state = _state.get()
    if state is None or not state.frames:
        return

    frame = _top_frame(state)
    mode = frame.represent_gate
    if mode == REPRESENT_DENY_NEW:
        frame.rejections.append(('represent_gate', mode, kind))
        raise RepresentAdmissionDenied(mode)
    if mode == REPRESENT_FORCE_FAIL:
        frame.rejections.append(('represent_gate', mode, kind))
        raise RepresentationForceFailed(mode)


# Render gate ----------------------------------------------------------------

def guard_render(response=None, renderer=None):
    """
    Gate the single response rendering choke point
    (``Response.rendered_content``).

    ``deny_new`` raises before any renderer runs and before the response is
    marked rendered. ``force_close`` does not block rendering; it acts when the
    frame pops.
    """
    state = _state.get()
    if state is None or not state.frames:
        return

    frame = _top_frame(state)
    mode = frame.render_gate
    if mode == RENDER_DENY_NEW:
        frame.rejections.append(('render_gate', mode, 'render'))
        raise RenderAdmissionDenied(mode)


class GuardedStreamingContent:
    """
    Wraps the raw ``StreamingHttpResponse._iterator`` while a policy frame is
    active.

    Every chunk re-checks the *then-current* stack. When the registering frame
    has been popped the wrapper degenerates to a pass-through, so a response
    consumed after its segment ends follows the normal path again.

    The wrapper drives the user iterator directly and closes it explicitly on
    force-close, so shutdown does not depend on consumers releasing their
    ``map``/iterator references. The response-level ``make_bytes`` encoding
    still happens outside this wrapper.
    """

    def __init__(self, upstream, frame, response=None):
        self._frame = frame
        self.closed = False
        self._response = response
        self._iterator = self._drive(iter(upstream))

    def _drive(self, source):
        try:
            while True:
                state = _state.get()
                if state is not None and self._frame in state.frames:
                    # The registering frame is live: the current top frame
                    # decides, re-read on every single chunk.
                    top = _top_frame(state)
                    if top.render_gate == RENDER_DENY_NEW:
                        top.rejections.append(
                            ('render_gate', top.render_gate,
                             'streaming_content')
                        )
                        raise RenderAdmissionDenied(top.render_gate)
                try:
                    chunk = next(source)
                except StopIteration:
                    return
                yield chunk
        finally:
            # Explicitly shut down the user iterator (generators receive
            # GeneratorExit, upload/file wrappers run their own cleanup).
            closer = getattr(source, 'close', None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass

    def __iter__(self):
        return self._iterator

    def __next__(self):
        return next(self._iterator)

    def close(self):
        if self.closed:
            return
        self.closed = True
        iterator = self._iterator
        self._iterator = None
        if iterator is not None:
            # Closing marks the generator finished; dropping the remaining
            # references deallocates its frame, which unwinds the upstream
            # chain (GeneratorExit reaches the user generator) immediately.
            iterator.close()
            response = self._response
            if response is not None and getattr(
                    response, '_iterator', None) is iterator:
                # The streaming response still aliases the finished drive
                # generator; replace it with an empty iterator so the upstream
                # chain is released right now instead of at response GC.
                response._iterator = iter(())


def guard_streaming_response(response):
    """
    Wire a ``StreamingHttpResponse`` returned by a view to the current frame:
    chunk writes are gated and the iterator is registered for ``force_close``.

    No-op with an empty stack, and never re-wraps an already guarded response.
    """
    state = _state.get()
    if state is None or not state.frames:
        return response

    # Idempotency without stamping the response instance: Django records each
    # content's ``.close`` in its resource closer registry.
    closers = getattr(response, '_resource_closers', ())
    if any(isinstance(getattr(closer, '__self__', None),
                      GuardedStreamingContent) for closer in closers):
        return response

    if getattr(response, 'is_async', False):
        # The synchronous policy stack does not gate async streaming
        # responses; leave them on their stock code path.
        return response

    frame = _top_frame(state)
    # Drive the raw user iterator directly (response._iterator); the getter
    # keeps applying make_bytes outside the wrapper. Handing the wrapper to
    # the setter also registers its ``.close`` for normal request shutdown.
    guarded = GuardedStreamingContent(response._iterator, frame,
                                      response=response)
    response.streaming_content = guarded
    track_resource(guarded, kind='streaming_response')
    return response
