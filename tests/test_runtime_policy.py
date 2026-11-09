"""
Tests for the request-scoped runtime policy stack
(``rest_framework.runtime_policy``).

Covers the three gates (``parse_admit`` / ``represent_gate`` /
``render_gate``), in-flight protection, ``one_more`` quotas, ``force_fail``,
``force_close`` cleanup, nested frame restoration, exception safety,
concurrent scope isolation and full equivalence with the stock behavior when
the stack is empty.
"""
import io
import threading

import pytest
from django.core.files.uploadhandler import MemoryFileUploadHandler
from django.http import StreamingHttpResponse
from django.test import TestCase

from rest_framework import fields, renderers, runtime_policy as rp, serializers
from rest_framework.decorators import action
from rest_framework.generics import CreateAPIView
from rest_framework.parsers import (
    FileUploadParser, FormParser, JSONParser, MultiPartParser
)
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.settings import api_settings
from rest_framework.test import APIRequestFactory
from rest_framework.viewsets import ViewSet
from tests.models import ManyToManyTarget

factory = APIRequestFactory()


# Serializers used by the tests ----------------------------------------------

class RecordingSerializer(serializers.Serializer):
    name = serializers.CharField()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.created = []
        self.updated = []

    def create(self, validated_data):
        self.created.append(dict(validated_data))
        return validated_data

    def update(self, instance, validated_data):
        self.updated.append(dict(validated_data))
        return instance


class DenyPushingChildSerializer(serializers.Serializer):
    x = serializers.IntegerField()

    def validate(self, attrs):
        # First item pushes a deny frame while the list validation is already
        # in flight. Subsequent item validations belong to that in-flight
        # operation and must keep running.
        if getattr(self, '_pushed_frame', None) is None:
            self._pushed_frame = rp.push_policy_frame(
                represent_gate=rp.REPRESENT_DENY_NEW
            )
        return attrs


# Helpers --------------------------------------------------------------------

class MockUploadRequest:
    def __init__(self):
        self.upload_handlers = (MemoryFileUploadHandler(),)
        self.META = {
            'HTTP_CONTENT_DISPOSITION': 'inline; filename=file.txt',
            'HTTP_CONTENT_LENGTH': 14,
        }


def file_upload_context():
    return {'request': MockUploadRequest(), 'kwargs': {}}


class PolicyPushingStream:
    """Stream wrapper that pushes a deny frame on its very first read."""

    def __init__(self, raw):
        self.raw = raw
        self.frame = None

    def read(self, size=-1):
        if self.frame is None:
            self.frame = rp.push_policy_frame(
                parse_admit=rp.PARSE_DENY_NEW
            )
        return self.raw.read(size)

    def close(self):
        self.raw.close()


def closeable_stream_response(chunks=(b'a', b'b', b'c')):
    state = {'closed': False}

    def generator():
        try:
            for chunk in chunks:
                yield chunk
        except GeneratorExit:
            state['closed'] = True
            raise

    response = StreamingHttpResponse(generator())
    return response, state


def render_ready_response(data=None):
    response = Response(data if data is not None else {'ok': True})
    response.accepted_renderer = renderers.JSONRenderer()
    response.accepted_media_type = 'application/json'
    response.renderer_context = {}
    return response


# Empty stack pins the existing behavior -------------------------------------

class TestEmptyStackIsPinned:
    def test_json_parser_unaffected(self):
        assert JSONParser().parse(io.BytesIO(b'{"a": 1}')) == {'a': 1}

    def test_form_parser_unaffected(self):
        data = FormParser().parse(io.BytesIO(b'a=1&b=2'))
        assert data['a'] == '1' and data['b'] == '2'

    def test_file_upload_parser_unaffected(self):
        result = FileUploadParser().parse(
            io.BytesIO(b'Test text file'), None, file_upload_context()
        )
        assert result.files['file'].size == 14

    def test_multipart_request_unaffected(self):
        django_request = factory.post('/url/', {'field': 'value'},
                                      format='multipart')
        request = Request(django_request, parsers=[MultiPartParser()])
        assert request.data['field'] == 'value'

    def test_serializer_unaffected(self):
        serializer = RecordingSerializer(data={'name': 'Ada'})
        assert serializer.is_valid()
        assert serializer.validated_data == {'name': 'Ada'}
        assert serializer.save() == {'name': 'Ada'}

    def test_render_unaffected(self):
        response = render_ready_response()
        response.render()
        assert response.is_rendered
        assert response.content == b'{"ok":true}'

    def test_default_parser_and_renderer_classes_never_rewritten(self):
        parsers_setting = api_settings.DEFAULT_PARSER_CLASSES
        renderers_setting = api_settings.DEFAULT_RENDERER_CLASSES
        with rp.push_policy(parse_admit=rp.PARSE_DENY_NEW,
                            represent_gate=rp.REPRESENT_DENY_NEW,
                            render_gate=rp.RENDER_DENY_NEW):
            pass
        assert api_settings.DEFAULT_PARSER_CLASSES is parsers_setting
        assert api_settings.DEFAULT_RENDERER_CLASSES is renderers_setting

    def test_stack_is_empty_without_scope(self):
        assert rp.current_stack() == ()
        assert rp.current_frame() is None


# parse_admit ----------------------------------------------------------------

class TestParseAdmit:
    def test_deny_new_blocks_every_builtin_parser_entry(self):
        with rp.push_parse_admit(rp.PARSE_DENY_NEW):
            with pytest.raises(rp.ParseAdmissionDenied):
                JSONParser().parse(io.BytesIO(b'{"a": 1}'))
            with pytest.raises(rp.ParseAdmissionDenied):
                FormParser().parse(io.BytesIO(b'a=1'))
            with pytest.raises(rp.ParseAdmissionDenied):
                MultiPartParser().parse(
                    io.BytesIO(b''), 'multipart/form-data; boundary=x',
                    {'request': MockUploadRequest()}
                )
            with pytest.raises(rp.ParseAdmissionDenied):
                FileUploadParser().parse(
                    io.BytesIO(b'Test text file'), None,
                    file_upload_context()
                )

    def test_deny_new_does_not_read_any_byte(self):
        class TrackingStream(io.BytesIO):
            def __init__(self):
                super().__init__(b'{"a": 1}')
                self.reads = 0

            def read(self, *args, **kwargs):
                self.reads += 1
                return super().read(*args, **kwargs)

        stream = TrackingStream()
        with rp.push_parse_admit(rp.PARSE_DENY_NEW):
            with pytest.raises(rp.ParseAdmissionDenied):
                JSONParser().parse(stream)
        assert stream.reads == 0

    def test_deny_new_through_request_data(self):
        django_request = factory.post('/url/', {'field': 'value'},
                                      format='multipart')
        request = Request(django_request, parsers=[MultiPartParser()])
        with rp.push_parse_admit(rp.PARSE_DENY_NEW):
            with pytest.raises(rp.ParseAdmissionDenied):
                request.data

    def test_deny_is_observable_on_frame_and_exception(self):
        with rp.push_parse_admit(rp.PARSE_DENY_NEW) as frame:
            with pytest.raises(rp.ParseAdmissionDenied) as exc:
                JSONParser().parse(io.BytesIO(b'{}'))
            assert exc.value.gate == 'parse_admit'
            assert exc.value.mode == rp.PARSE_DENY_NEW
            assert ('parse_admit', 'deny_new', 'parse') in frame.rejections

    def test_one_more_allows_exactly_one_more_parse(self):
        with rp.push_parse_admit(rp.PARSE_ONE_MORE) as frame:
            assert JSONParser().parse(io.BytesIO(b'{"a": 1}')) == {'a': 1}
            # Second and third entry re-read the then-current top and fail.
            with pytest.raises(rp.ParseAdmissionDenied):
                JSONParser().parse(io.BytesIO(b'{"a": 2}'))
            with pytest.raises(rp.ParseAdmissionDenied):
                JSONParser().parse(io.BytesIO(b'{"a": 3}'))
            assert frame.parse_quota == 0

    def test_one_more_second_entry_rereads_current_top(self):
        with rp.push_parse_admit(rp.PARSE_ONE_MORE):
            assert JSONParser().parse(io.BytesIO(b'{"a": 1}')) == {'a': 1}
            # An inner frame with no opinion relaxes the exhausted outer one.
            with rp.push_parse_admit(rp.PARSE_HONOR_CFG):
                assert JSONParser().parse(io.BytesIO(b'{"a": 2}')) == {'a': 2}
            # Inner frame popped: the exhausted outer top decides again.
            with pytest.raises(rp.ParseAdmissionDenied):
                JSONParser().parse(io.BytesIO(b'{"a": 3}'))

    def test_inflight_parse_completes_after_deny_pushed(self):
        # An outer frame exists so the parse is admitted with in-flight state.
        with rp.push_parse_admit(rp.PARSE_HONOR_CFG):
            stream = PolicyPushingStream(io.BytesIO(b'{"a": 1}'))
            # First read pushes deny mid-parse; the in-flight parse still
            # completes with its real data and is not rewritten into an
            # empty success.
            assert JSONParser().parse(stream) == {'a': 1}
            # Only *new* parse entries are constrained afterwards.
            with pytest.raises(rp.ParseAdmissionDenied):
                JSONParser().parse(io.BytesIO(b'{"a": 2}'))
            rp.pop_policy_frame(stream.frame)

    def test_inflight_streaming_upload_reads_continue(self):
        with rp.push_parse_admit(rp.PARSE_HONOR_CFG):
            stream = PolicyPushingStream(io.BytesIO(b'Test text file'))
            result = FileUploadParser().parse(
                stream, None, file_upload_context()
            )
            assert result.files['file'].size == 14
            with pytest.raises(rp.ParseAdmissionDenied):
                FileUploadParser().parse(
                    io.BytesIO(b'Test text file'), None,
                    file_upload_context()
                )
            rp.pop_policy_frame(stream.frame)

    def test_stray_stream_read_is_rejected(self):
        with rp.push_parse_admit(rp.PARSE_DENY_NEW):
            with pytest.raises(rp.ParseAdmissionDenied):
                rp.guard_upload_read()


# represent_gate -------------------------------------------------------------

class TestRepresentGate:
    def test_deny_new_blocks_serializer_validation(self):
        serializer = RecordingSerializer(data={'name': 'Ada'})
        with rp.push_represent_gate(rp.REPRESENT_DENY_NEW):
            with pytest.raises(rp.RepresentAdmissionDenied):
                serializer.is_valid()

    def test_deny_new_blocks_list_serializer_entry(self):
        serializer = RecordingSerializer(data=[{'name': 'a'}, {'name': 'b'}],
                                         many=True)
        with rp.push_represent_gate(rp.REPRESENT_DENY_NEW):
            with pytest.raises(rp.RepresentAdmissionDenied):
                serializer.is_valid()

    def test_deny_new_blocks_standalone_field_validation(self):
        with rp.push_represent_gate(rp.REPRESENT_DENY_NEW):
            with pytest.raises(rp.RepresentAdmissionDenied):
                fields.IntegerField().run_validation(5)

    def test_denied_validation_is_not_swallowed_as_field_error(self):
        # The denial propagates instead of becoming an empty/errored dict.
        serializer = RecordingSerializer(data={'name': 'Ada'})
        with rp.push_represent_gate(rp.REPRESENT_DENY_NEW):
            with pytest.raises(rp.RepresentAdmissionDenied):
                serializer.run_validation({'name': 'Ada'})

    def test_inflight_list_validation_completes_after_deny_pushed(self):
        list_serializer = DenyPushingChildSerializer(
            data=[{'x': 1}, {'x': 2}, {'x': 3}], many=True
        )
        assert list_serializer.is_valid()
        # Real values for every item, none cancelled or emptied.
        assert list_serializer.validated_data == [
            {'x': 1}, {'x': 2}, {'x': 3}
        ]
        child = list_serializer.child
        # A fresh validation after the push is rejected ...
        fresh = RecordingSerializer(data={'name': 'new'})
        with pytest.raises(rp.RepresentAdmissionDenied):
            fresh.is_valid()
        # ... then the frame pushed mid-flight is cleaned up.
        rp.pop_policy_frame(child._pushed_frame)

    def test_deny_new_blocks_create_and_update_writeouts(self):
        create_serializer = RecordingSerializer(data={'name': 'Ada'})
        create_serializer.is_valid()
        with rp.push_represent_gate(rp.REPRESENT_DENY_NEW):
            with pytest.raises(rp.RepresentAdmissionDenied):
                create_serializer.save()
        assert create_serializer.created == []

        instance = {'name': 'old'}
        update_serializer = RecordingSerializer(instance, data={'name': 'new'})
        update_serializer.is_valid()
        with rp.push_represent_gate(rp.REPRESENT_DENY_NEW):
            with pytest.raises(rp.RepresentAdmissionDenied):
                update_serializer.save()
        assert update_serializer.updated == []

    def test_force_fail_does_not_swallow_writes_as_success(self):
        # Direct write-outs on a plain Serializer are reached via save(); the
        # user object returned by validation stays untouched and no fake
        # success is produced.
        serializer = RecordingSerializer(data={'name': 'Ada'})
        serializer.is_valid()
        with rp.push_represent_gate(rp.REPRESENT_FORCE_FAIL):
            with pytest.raises(rp.RepresentationForceFailed):
                serializer.save()
        assert serializer.instance is None
        assert serializer.validated_data == {'name': 'Ada'}

    def test_deny_new_blocks_list_writeout(self):
        list_serializer = RecordingSerializer(
            data=[{'name': 'a'}, {'name': 'b'}], many=True
        )
        list_serializer.is_valid()
        with rp.push_represent_gate(rp.REPRESENT_DENY_NEW):
            with pytest.raises(rp.RepresentAdmissionDenied):
                list_serializer.save()

    def test_force_fail_keeps_validated_object_and_fails_the_writeout(self):
        serializer = RecordingSerializer(data={'name': 'Ada'})
        with rp.push_represent_gate(rp.REPRESENT_FORCE_FAIL):
            # Validation still runs and returns the real Python object.
            assert serializer.is_valid()
            assert serializer.validated_data == {'name': 'Ada'}
            with pytest.raises(rp.RepresentationForceFailed) as exc:
                serializer.save()
            assert exc.value.gate == 'represent_gate'
            assert exc.value.mode == rp.REPRESENT_FORCE_FAIL
            # The legal object is still in the caller's hands afterwards.
            assert serializer.validated_data == {'name': 'Ada'}
        assert serializer.created == []

    def test_force_fail_list_writeout(self):
        list_serializer = RecordingSerializer(
            data=[{'name': 'a'}, {'name': 'b'}], many=True
        )
        list_serializer.is_valid()
        with rp.push_represent_gate(rp.REPRESENT_FORCE_FAIL):
            with pytest.raises(rp.RepresentationForceFailed):
                list_serializer.save()


class TargetModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = ManyToManyTarget
        fields = ('id', 'name')


class TestModelSerializerWriteGates(TestCase):
    def test_deny_new_blocks_direct_model_create_and_update(self):
        with rp.push_represent_gate(rp.REPRESENT_DENY_NEW):
            with pytest.raises(rp.RepresentAdmissionDenied):
                TargetModelSerializer().create({'name': 'Ada'})
            instance = ManyToManyTarget.objects.create(name='old')
            with pytest.raises(rp.RepresentAdmissionDenied):
                TargetModelSerializer(instance).update(
                    instance, {'name': 'new'}
                )
        assert ManyToManyTarget.objects.filter(name='Ada').count() == 0
        assert ManyToManyTarget.objects.get(pk=instance.pk).name == 'old'

    def test_force_fail_blocks_direct_model_create(self):
        with rp.push_represent_gate(rp.REPRESENT_FORCE_FAIL):
            with pytest.raises(rp.RepresentationForceFailed):
                TargetModelSerializer().create({'name': 'Ada'})
        assert ManyToManyTarget.objects.filter(name='Ada').count() == 0


# render_gate ----------------------------------------------------------------

class TestRenderGate:
    def test_deny_new_blocks_response_render_and_marks_nothing(self):
        response = render_ready_response()
        with rp.push_render_gate(rp.RENDER_DENY_NEW) as frame:
            with pytest.raises(rp.RenderAdmissionDenied):
                response.render()
            # Never marked rendered, and accessing content explicitly fails
            # because rendering did not happen.
            assert not response.is_rendered
            with pytest.raises(Exception):
                response.content
            assert ('render_gate', 'deny_new', 'render') in frame.rejections
        # Once the segment ends the response renders normally.
        response.render()
        assert response.content == b'{"ok":true}'

    def test_deny_new_blocks_streaming_writes(self):
        response, state = closeable_stream_response()
        with rp.push_render_gate(rp.RENDER_DENY_NEW):
            rp.guard_streaming_response(response)
            chunks = []
            with pytest.raises(rp.RenderAdmissionDenied):
                for chunk in response.streaming_content:
                    chunks.append(chunk)
            assert chunks == []
        assert state['closed'] is False

    def test_streaming_passes_through_after_registering_frame_popped(self):
        response, state = closeable_stream_response()
        with rp.push_render_gate(rp.RENDER_DENY_NEW):
            rp.guard_streaming_response(response)
        # Frame restored: consumption follows the normal path again.
        assert b''.join(response.streaming_content) == b'abc'
        assert state['closed'] is False

    def test_force_close_closes_still_open_streaming_response_on_pop(self):
        response, state = closeable_stream_response()
        with rp.push_render_gate(rp.RENDER_FORCE_CLOSE) as frame:
            rp.guard_streaming_response(response)
            iterator = iter(response.streaming_content)
            assert next(iterator) == b'a'
        assert state['closed'] is True
        assert len(frame.closed_resources) == 1

    def test_force_close_does_not_touch_other_frames_resources(self):
        outer_response, outer_state = closeable_stream_response()
        inner_response, inner_state = closeable_stream_response()
        with rp.push_render_gate(rp.RENDER_FORCE_CLOSE):
            rp.guard_streaming_response(outer_response)
            with rp.push_render_gate(rp.RENDER_HONOR_CFG):
                rp.guard_streaming_response(inner_response)
                next(iter(inner_response.streaming_content))
            # Inner frame pops normally: its stream stays open.
            assert inner_state['closed'] is False
            next(iter(outer_response.streaming_content))
        assert outer_state['closed'] is True
        assert inner_state['closed'] is False


# force_close on upload streams ----------------------------------------------

class TestForceCloseUploads:
    def test_force_close_closes_open_upload_resources(self):
        raw = io.BytesIO(b'Test text file')
        with rp.push_render_gate(rp.RENDER_FORCE_CLOSE) as frame:
            result = FileUploadParser().parse(raw, None, file_upload_context())
            file_obj = result.files['file']
            assert file_obj.closed is False
            assert raw.closed is False
        assert file_obj.closed is True
        assert raw.closed is True
        # Pop consumed the resource list and closed both the raw upload
        # stream and the uploaded file object.
        assert frame.resources == []
        assert len(frame.closed_resources) >= 2

    def test_already_finished_resources_are_not_reported(self):
        raw = io.BytesIO(b'Test text file')
        with rp.push_render_gate(rp.RENDER_FORCE_CLOSE) as frame:
            result = FileUploadParser().parse(raw, None, file_upload_context())
            file_obj = result.files['file']
            file_obj.close()
            raw.close()
        # Both already finished cleanly: force-close has nothing to observe.
        assert frame.closed_resources == []


# Nested frames, exceptions, scope -------------------------------------------

class TestStackDiscipline:
    def test_inner_pop_restores_outer(self):
        with rp.push_parse_admit(rp.PARSE_DENY_NEW):
            with rp.push_parse_admit(rp.PARSE_HONOR_CFG):
                assert JSONParser().parse(io.BytesIO(b'{"a": 1}')) == {'a': 1}
            with pytest.raises(rp.ParseAdmissionDenied):
                JSONParser().parse(io.BytesIO(b'{"a": 1}'))

    def test_frame_is_popped_when_body_raises(self):
        with pytest.raises(RuntimeError):
            with rp.push_render_gate(rp.RENDER_FORCE_CLOSE):
                raise RuntimeError('boom')
        assert rp.current_stack() == ()

    def test_request_scope_cleans_unpopped_frame(self):
        raw = io.BytesIO(b'x')
        with pytest.raises(ValueError):
            with rp.request_scope():
                rp.push_policy_frame(render_gate=rp.RENDER_FORCE_CLOSE)
                rp.track_resource(raw, kind='upload_stream')
                raise ValueError('boom')
        assert raw.closed is True
        assert rp.current_stack() == ()

    def test_existing_scope_is_shared(self):
        with rp.request_scope() as outer:
            with rp.request_scope() as inner:
                assert inner is outer

    def test_pop_from_other_scope_is_an_error(self):
        frame = rp.push_policy_frame(parse_admit=rp.PARSE_DENY_NEW)

        def elsewhere():
            with pytest.raises(RuntimeError):
                rp.pop_policy_frame(frame)

        thread = threading.Thread(target=elsewhere)
        thread.start()
        thread.join()
        rp.pop_policy_frame(frame)


# Concurrency isolation ------------------------------------------------------

class TestConcurrencyIsolation:
    def test_pushed_and_unpushed_requests_run_concurrently(self):
        results = {}
        barrier = threading.Barrier(2)

        def denied_thread():
            with rp.request_scope():
                with rp.push_parse_admit(rp.PARSE_DENY_NEW):
                    barrier.wait()
                    try:
                        JSONParser().parse(io.BytesIO(b'{"who": "a"}'))
                    except rp.ParseAdmissionDenied:
                        results['a'] = 'denied'

        def plain_thread():
            with rp.request_scope():
                barrier.wait()
                results['b'] = JSONParser().parse(io.BytesIO(b'{"who": "b"}'))

        threads = [threading.Thread(target=denied_thread),
                   threading.Thread(target=plain_thread)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert results == {'a': 'denied', 'b': {'who': 'b'}}
        assert rp.current_stack() == ()

    def test_two_concurrent_frames_do_not_cross(self):
        observed = {}
        barrier = threading.Barrier(2)

        def thread_a():
            with rp.request_scope():
                with rp.push_policy(
                        parse_admit=rp.PARSE_DENY_NEW,
                        represent_gate=rp.REPRESENT_HONOR_CFG):
                    barrier.wait()
                    observed['a'] = rp.active_modes()

        def thread_b():
            with rp.request_scope():
                with rp.push_policy(
                        parse_admit=rp.PARSE_HONOR_CFG,
                        represent_gate=rp.REPRESENT_FORCE_FAIL):
                    barrier.wait()
                    observed['b'] = rp.active_modes()

        threads = [threading.Thread(target=thread_a),
                   threading.Thread(target=thread_b)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert observed['a']['parse_admit'] == rp.PARSE_DENY_NEW
        assert observed['a']['represent_gate'] == rp.REPRESENT_HONOR_CFG
        assert observed['b']['parse_admit'] == rp.PARSE_HONOR_CFG
        assert observed['b']['represent_gate'] == rp.REPRESENT_FORCE_FAIL

    def test_force_close_in_one_request_spares_other_request_stream(self):
        other_response, other_state = closeable_stream_response()
        barrier = threading.Barrier(2)
        outcome = {}

        def other_request():
            with rp.request_scope():
                with rp.push_render_gate(rp.RENDER_HONOR_CFG):
                    rp.guard_streaming_response(other_response)
                    next(iter(other_response.streaming_content))
                    barrier.wait()
                    barrier.wait()
                    outcome['other_closed'] = other_state['closed']

        thread = threading.Thread(target=other_request)
        thread.start()

        with rp.request_scope():
            barrier.wait()
            response, state = closeable_stream_response()
            with rp.push_render_gate(rp.RENDER_FORCE_CLOSE):
                rp.guard_streaming_response(response)
                next(iter(response.streaming_content))
        barrier.wait()
        thread.join()

        assert state['closed'] is True
        assert outcome['other_closed'] is False


# End-to-end through APIView / ViewSet @action --------------------------------

class PolicyViewSet(ViewSet):
    @action(detail=False, methods=['post'])
    def plain_echo(self, request):
        return Response({'echo': request.data})

    @action(detail=False, methods=['post'])
    def parse_denied(self, request):
        with rp.push_parse_admit(rp.PARSE_DENY_NEW):
            return Response({'echo': request.data})

    @action(detail=False, methods=['post'])
    def represent_denied(self, request):
        serializer = RecordingSerializer(data=request.data)
        with rp.push_represent_gate(rp.REPRESENT_DENY_NEW):
            serializer.is_valid(raise_exception=True)
        return Response({'ok': True})

    @action(detail=False, methods=['post'])
    def write_force_failed(self, request):
        serializer = RecordingSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        with rp.push_represent_gate(rp.REPRESENT_FORCE_FAIL):
            serializer.save()
        return Response({'ok': True})


def view_for(action_name):
    return PolicyViewSet.as_view({'post': action_name})


class TestActionFailureSurface:
    def test_plain_action_is_200(self):
        response = view_for('plain_echo')(
            factory.post('/x/', {'hello': 'world'}, format='json')
        )
        assert response.status_code == 200
        assert response.data == {'echo': {'hello': 'world'}}

    def test_parse_denied_action_failure_surface(self):
        response = view_for('parse_denied')(
            factory.post('/x/', {'hello': 'world'}, format='json')
        )
        assert response.status_code == 403
        assert response.data['detail'].code == 'parse_admission_denied'

    def test_represent_denied_action_failure_surface(self):
        response = view_for('represent_denied')(
            factory.post('/x/', {'name': 'Ada'}, format='json')
        )
        assert response.status_code == 403
        assert response.data['detail'].code == 'represent_admission_denied'

    def test_force_fail_action_failure_surface(self):
        response = view_for('write_force_failed')(
            factory.post('/x/', {'name': 'Ada'}, format='json')
        )
        assert response.status_code == 403
        assert response.data['detail'].code == 'representation_force_failed'

    def test_frame_in_one_request_does_not_change_the_next_request(self):
        denied = view_for('parse_denied')(
            factory.post('/x/', {'hello': 'world'}, format='json')
        )
        assert denied.status_code == 403

        plain = view_for('plain_echo')(
            factory.post('/x/', {'hello': 'world'}, format='json')
        )
        assert plain.status_code == 200
        assert rp.current_stack() == ()

    def test_frame_pushed_like_middleware_is_honored_inside_dispatch(self):
        # A frame pushed before dispatch (middleware/decorator style) must
        # bind the whole request even though the view pushes nothing itself.
        with rp.push_parse_admit(rp.PARSE_DENY_NEW):
            response = view_for('plain_echo')(
                factory.post('/x/', {'hello': 'world'}, format='json')
            )
        assert response.status_code == 403
        assert response.data['detail'].code == 'parse_admission_denied'
        assert rp.current_stack() == ()

    def test_middleware_style_frame_does_not_leak_across_requests(self):
        with rp.push_parse_admit(rp.PARSE_DENY_NEW):
            view_for('plain_echo')(
                factory.post('/x/', {'hello': 'world'}, format='json')
            )
        # Frame popped: an unrelated later request follows stock DRF.
        later = view_for('plain_echo')(
            factory.post('/x/', {'hello': 'world'}, format='json')
        )
        assert later.status_code == 200


# GenericAPIView get_serializer nested callback path -------------------------

class NestedSerializerView(CreateAPIView):
    serializer_class = RecordingSerializer
    seen = []

    def get_serializer(self, *args, **kwargs):
        # A nested callback reads the *then-current* top frame, including a
        # frame pushed by an outer (middleware/decorator-style) segment.
        NestedSerializerView.seen.append(
            rp.active_modes()['represent_gate']
        )
        with rp.push_represent_gate(rp.REPRESENT_DENY_NEW):
            NestedSerializerView.seen.append(
                rp.active_modes()['represent_gate']
            )
            return super().get_serializer(*args, **kwargs)


class ForceFailCreateView(CreateAPIView):
    serializer_class = TargetModelSerializer

    def perform_create(self, serializer):
        with rp.push_represent_gate(rp.REPRESENT_FORCE_FAIL):
            super().perform_create(serializer)


class TestGenericViewNesting(TestCase):
    def test_get_serializer_nested_callback_reads_live_stack(self):
        NestedSerializerView.seen = []
        with rp.push_represent_gate(rp.REPRESENT_HONOR_CFG):
            response = NestedSerializerView.as_view()(
                factory.post('/x/', {'name': 'Ada'}, format='json')
            )
        # Outer frame seen first, then the frame pushed by the nested
        # callback itself: no entry snapshot is reused.
        assert NestedSerializerView.seen == [
            rp.REPRESENT_HONOR_CFG, rp.REPRESENT_DENY_NEW
        ]
        # The nested frame popped inside get_serializer, so validation still
        # runs on the stock path and the request succeeds normally.
        assert response.status_code == 201
        assert rp.current_stack() == ()

    def test_generic_create_honors_force_fail_end_to_end(self):
        response = ForceFailCreateView.as_view()(
            factory.post('/x/', {'name': 'Ada'}, format='json')
        )
        assert response.status_code == 403
        assert response.data['detail'].code == 'representation_force_failed'
        assert ManyToManyTarget.objects.filter(name='Ada').count() == 0
