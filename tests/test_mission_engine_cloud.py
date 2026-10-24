"""CPU-only cloud transport and ZMQ contract tests; no external API calls."""

import base64
import copy
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import zmq
from httpx import Request
from openai import APIStatusError, APITimeoutError
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import mission_engine
import mission_engine_server1 as mission_server
from mission_engine import CloudClient, QwenContext


def response_body():
    return {
        "id": "resp_test", "object": "response", "created_at": 0,
        "model": "gpt-6-luna", "status": "completed",
        "output": [{
            "type": "message", "id": "msg_test", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "OK", "annotations": []}],
        }],
        "usage": {
            "input_tokens": 90, "output_tokens": 5, "total_tokens": 95,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


class CloudTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.http_status = 200
        self.body = response_body()
        test = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                test.requests.append((self.path, None, None))
                self.send_response(405)
                self.end_headers()

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                test.requests.append((self.path, self.headers["Authorization"], payload))
                self.send_response(test.http_status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(test.body).encode())

            def log_message(self, *args):
                pass

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.http.server_port}/v1"
        self.client = CloudClient(
            model="gpt-6-luna", api_key="test-cloud-key", base_url=self.base_url,
            reasoning_effort="none",
        )
        self.ctx = QwenContext(enabled=True, model_family="luna", cloud_client=self.client)

    def tearDown(self):
        self.client._client.close()
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()

    def test_responses_matches_notebook_and_preserves_prompt(self):
        prompt = "  exact rendered prompt\n"
        self.assertEqual(mission_engine.generate_text_only_response(self.ctx, prompt, 123), "OK")
        path, auth, payload = self.requests[-1]
        self.assertEqual(path, "/v1/responses")
        self.assertEqual(auth, "Bearer test-cloud-key")
        self.assertEqual(payload, {
            "model": "gpt-6-luna", "input": [{
                "role": "user", "content": [{"type": "input_text", "text": prompt}],
            }],
            "max_output_tokens": 123, "reasoning": {"effort": "none"}, "store": False,
        })
        self.assertEqual(self.client._client.max_retries, 0)
        self.assertEqual(self.client._client.timeout, 60.0)
        self.assertEqual(self.client.last_usage["input_tokens"], 90)
        self.assertNotIn("test-cloud-key", repr(self.client))

    def test_images_paths_and_frames_use_full_resolution_in_order(self):
        images = [Image.new("RGBA", (800, 400), color) for color in ("red", "blue")]
        self.assertEqual(mission_engine.generate_text_video_response(self.ctx, images, "status", 77), "OK")
        content = self.requests[-1][2]["input"][0]["content"]
        self.assertEqual(content[-1], {"type": "input_text", "text": "status"})
        self.assertEqual(len(content), 3)
        for item, channel in zip(content[:-1], (0, 2)):
            self.assertEqual(item["detail"], "high")
            with Image.open(io.BytesIO(base64.b64decode(item["image_url"].split(",", 1)[1]))) as image:
                self.assertEqual(image.size, (800, 400))
                self.assertEqual(image.mode, "RGB")
                self.assertGreater(image.getpixel((0, 0))[channel], 240)
        self.assertEqual(images[0].mode, "RGBA")
        self.assertEqual(self.requests[-1][2]["max_output_tokens"], 77)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            images[-1].save(path)
            self.assertEqual(mission_engine.generate_text_image_response_from_path(self.ctx, str(path), "look"), "OK")
            self.assertEqual(mission_engine.generate_text_image_response(self.ctx, images[0], "look"), "OK")
        self.assertEqual(len(self.requests[-1][2]["input"][0]["content"]), 2)

    def test_chat_completions_supports_generic_cloud_models(self):
        client = CloudClient(
            model="provider/vision-model", api_key="test-cloud-key",
            base_url=self.base_url, api="chat_completions",
        )
        self.addCleanup(client._client.close)
        self.body = {
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
        }
        image = Image.new("RGB", (12, 12), "red")
        self.assertEqual(client.generate("unchanged", [image], 44), "answer")
        path, _, payload = self.requests[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(payload["model"], "provider/vision-model")
        self.assertEqual(payload["max_completion_tokens"], 44)
        self.assertFalse(payload["store"])
        self.assertNotIn("reasoning_effort", payload)
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertEqual(payload["messages"][0]["content"][-1], {"type": "text", "text": "unchanged"})
        self.assertEqual(payload["messages"][0]["content"][0]["type"], "image_url")
        self.assertEqual(client.last_usage["prompt_tokens"], 9)
        client.reasoning_effort = "none"
        client.generate("hello", [], 44)
        self.assertEqual(self.requests[-1][2]["reasoning_effort"], "none")
        for change, message in (
            ({"choices": []}, "no completion"),
            ({"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}, "length"),
            ({"choices": [{"message": {"content": "blocked"}, "finish_reason": "content_filter"}]}, "content_filter"),
            ({"choices": [{"message": {"content": "no", "refusal": "blocked"}, "finish_reason": "stop"}]}, "refused"),
            ({"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}, "no assistant text"),
            ({"usage": None}, "no usage"),
            ({"usage": {"completion_tokens_details": {"reasoning_tokens": 2}}}, "reasoning tokens"),
        ):
            with self.subTest(change=change):
                saved = copy.deepcopy(self.body)
                self.body.update(change)
                with self.assertRaisesRegex(RuntimeError, message):
                    client.generate("hello", [], 44)
                self.assertIsNone(client.last_usage)
                self.body = saved

    def test_responses_failures_are_not_audit_results(self):
        for change, message in (
            ({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}, "max_output_tokens"),
            ({"status": "failed"}, "not completed"),
            ({"output": []}, "no assistant text"),
            ({"output": [{
                "type": "message", "id": "m", "role": "assistant", "status": "completed",
                "content": [{"type": "refusal", "refusal": "blocked"}],
            }]}, "refused"),
            ({"usage": None}, "no usage"),
            ({"usage": {"output_tokens_details": {"reasoning_tokens": 2}}}, "reasoning tokens"),
        ):
            with self.subTest(change=change):
                self.body = response_body()
                self.body.update(change)
                with self.assertRaisesRegex(RuntimeError, message):
                    self.client.generate("hello", [], 44)
                self.assertIsNone(self.client.last_usage)

    def test_http_errors_and_timeout_are_not_retried(self):
        for status in (401, 429, 500):
            with self.subTest(status=status):
                self.http_status = status
                self.body = {"error": {"message": "provider error", "type": "test"}}
                count = len(self.requests)
                with self.assertRaises(APIStatusError) as caught:
                    self.client.generate("hello", [], 44)
                self.assertEqual(caught.exception.status_code, status)
                self.assertEqual(len(self.requests), count + 1)
                self.assertIsNone(self.client.last_usage)
        with patch.object(
            self.client._client.responses, "create",
            side_effect=APITimeoutError(request=Request("POST", self.base_url)),
        ) as create:
            with self.assertRaises(APITimeoutError):
                self.client.generate("hello", [], 44)
            create.assert_called_once()

    def test_validation_and_disabled_dispatch(self):
        for override in (
            {"model": ""}, {"api_key": ""}, {"timeout_s": 0}, {"timeout_s": float("nan")},
            {"timeout_s": float("inf")}, {"base_url": "not-a-url"},
            {"base_url": "http://provider.example/v1"},
            {"base_url": "https://user:secret@provider.example/v1"},
            {"base_url": "https://provider.example/v1?key=secret"},
            {"api": "unsupported"}, {"reasoning_effort": "unsupported"},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                CloudClient(**{"model": "test", "api_key": "test", **override})
        for prompt, tokens in (("", 10), (" ", 10), ("hello", 0)):
            with self.assertRaises(ValueError):
                self.client.generate(prompt, [], tokens)
        self.assertEqual(self.requests, [])
        self.ctx.enabled = False
        self.assertIsNone(mission_engine.generate_text_only_response(self.ctx, "hello"))
        self.assertIsNone(mission_engine.generate_text_video_response(self.ctx, [], "hello"))
        self.assertEqual(self.requests, [])
        self.ctx.enabled = True
        with patch.object(self.client._client, "close") as close:
            mission_engine.release_qwen_gpu(self.ctx)
            close.assert_called_once()
        self.assertFalse(self.ctx.enabled)
        self.assertIsNone(self.ctx.cloud_client)

    @contextmanager
    def server(self, *extra_args):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        context = zmq.Context()
        sock = context.socket(zmq.REQ)
        sock.LINGER = 0
        sock.RCVTIMEO = 5000
        sock.connect(f"tcp://127.0.0.1:{port}")
        with tempfile.TemporaryFile(mode="w+") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", str(SCRIPTS / "mission_engine_server1.py"),
                 "--model", "luna", "--port", str(port), "--cloud_base_url", self.base_url,
                 "--cloud_api_key_env", "TEST_CLOUD_KEY", *extra_args],
                env={**os.environ, "TEST_CLOUD_KEY": "test-cloud-key"},
                stdout=log, stderr=subprocess.STDOUT,
            )
            try:
                sock.send_json({"mode": "health"})
                deadline = time.monotonic() + 20
                while not sock.poll(100):
                    if process.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        self.fail(f"Cloud server did not become ready:\n{log.read()}")
                health = sock.recv_json()
                self.assertTrue(health["ok"], health)
                yield sock, health
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                sock.close()
                context.term()

    def test_zmq_cloud_contract_and_health_never_calls_provider(self):
        with self.server() as (sock, health):
            self.assertEqual(self.requests, [])
            self.assertEqual(health["backend"], "cloud")
            self.assertEqual(health["cloud_model"], "gpt-6-luna")
            self.assertEqual(health["cloud_reasoning_effort"], "none")
            self.assertEqual(health["health_scope"], "local_configuration_only")
            self.assertFalse(health["provider_checked"])
            self.assertEqual(health["torch_version"], "not_loaded")
            self.assertEqual(health["transformers_version"], "not_loaded")
            self.assertEqual(health["device_arg"], "remote")
            self.assertEqual(health["video_input"], "frames")
            self.assertNotIn("test-cloud-key", json.dumps(health))

            def request(payload):
                sock.send_json(payload)
                return sock.recv_json()

            result = request({"mode": "text", "prompt_text": " exact\n", "max_new_tokens": 42})
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["usage"]["input_tokens"], 90)
            self.assertEqual(self.requests[-1][2]["max_output_tokens"], 42)
            self.assertEqual(self.requests[-1][2]["input"][0]["content"][-1]["text"], " exact\n")
            frames = []
            for color in ("red", "blue"):
                buffer = io.BytesIO()
                Image.new("RGB", (32, 32), color).save(buffer, "JPEG")
                frames.append(base64.b64encode(buffer.getvalue()).decode())
            for mode in ("bt", "text_image", "text_video", "video", "scene"):
                with self.subTest(mode=mode):
                    result = request({
                        "mode": mode, "frames_b64": frames, "prompt_text": "status",
                        "task_text": "Pick up the cup.", "bt_variant": "btaudit",
                        "bt_max_new_tokens": 111,
                    })
                    self.assertTrue(result["ok"], result)
                    self.assertEqual(result["text"], "OK")
                    self.assertEqual(result["backend"], "cloud")
                    self.assertEqual(result["usage"]["output_tokens"], 5)
                    content = self.requests[-1][2]["input"][0]["content"]
                    self.assertEqual(len(content), 3 if mode in {"video", "text_video"} else 2)
                    if mode == "bt":
                        self.assertEqual(result["bt_max_new_tokens"], 111)
                        self.assertEqual(self.requests[-1][2]["max_output_tokens"], 111)
                        self.assertIn("Pick up the cup.", content[-1]["text"])
                    else:
                        self.assertEqual(content[-1]["text"], "status")
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "frame.jpg"
                Image.new("RGB", (32, 32), "red").save(path)
                self.assertTrue(request({
                    "mode": "text_image", "image_path": str(path),
                    "frames_b64": ["ignored"], "prompt_text": "look",
                })["ok"])
            for payload in (
                {"mode": "text", "prompt_text": ""},
                {"mode": "text_image", "frames_b64": frames},
                {"mode": "bt", "frames_b64": frames},
                {"mode": "text_image", "frames_b64": ["not-an-image"], "prompt_text": "look"},
            ):
                count = len(self.requests)
                self.assertFalse(request(payload)["ok"])
                self.assertEqual(len(self.requests), count)
            self.http_status = 429
            self.body = {"error": {"message": "rate limited"}}
            count = len(self.requests)
            result = request({"mode": "text", "prompt_text": "hello"})
            self.assertFalse(result["ok"])
            self.assertIn("429", result["error"])
            self.assertEqual(len(self.requests), count + 1)
            self.assertNotIn("usage", result)
            self.assertTrue(request({"mode": "health"})["ok"])
            self.assertEqual(len(self.requests), count + 1)
            self.http_status = 200
            self.body = response_body()
            self.assertTrue(request({"mode": "text", "prompt_text": "hello"})["ok"])

    def test_generic_cloud_cli_and_explicit_warmup(self):
        with self.server("--model", "cloud", "--cloud_model", "provider-model") as (_, health):
            self.assertEqual(self.requests, [])
            self.assertEqual(health["cloud_model"], "provider-model")
            self.assertIsNone(health["cloud_reasoning_effort"])
        self.body = {
            "choices": [{"message": {"content": "chat answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
        }
        with self.server(
            "--model", "cloud", "--cloud_model", "provider-model", "--cloud_api", "chat_completions",
        ) as (sock, health):
            self.assertEqual(health["cloud_api"], "chat_completions")
            sock.send_json({"mode": "text", "prompt_text": "hello"})
            result = sock.recv_json()
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["text"], "chat answer")
            self.assertEqual(result["usage"]["completion_tokens"], 2)
            self.assertEqual(self.requests[-1][0], "/v1/chat/completions")
        self.requests.clear()
        self.body = response_body()
        with self.server("--cloud_startup_warmup"):
            self.assertEqual(len(self.requests), 1)
            self.assertEqual(self.requests[0][2]["max_output_tokens"], 8)
        self.requests.clear()
        with self.server("--cloud_startup_warmup", "--disable_startup_warmup"):
            self.assertEqual(self.requests, [])

    def test_invalid_cloud_startup_fails_without_provider_calls(self):
        for extra, message in (
            (["--model", "cloud"], "--cloud_model"),
            (["--enable_thinking"], "--cloud_reasoning_effort"),
            (["--cloud_reasoning_effort", "high"], "Luna requires"),
            (["--cloud_timeout_s", "0"], "finite positive"),
            (["--cloud_api_key_env", "TEST_EMPTY_KEY"], "No cloud API key"),
        ):
            with self.subTest(extra=extra):
                result = subprocess.run(
                    [sys.executable, str(SCRIPTS / "mission_engine_server1.py"),
                     "--model", "luna", "--cloud_base_url", self.base_url,
                     "--cloud_api_key_env", "TEST_CLOUD_KEY", *extra],
                    env={**os.environ, "TEST_CLOUD_KEY": "test-cloud-key", "TEST_EMPTY_KEY": ""},
                    capture_output=True, text=True, timeout=10,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertNotIn("test-cloud-key", result.stdout + result.stderr)
        self.assertEqual(self.requests, [])

    def test_credentials_precedence_and_demo_fallback(self):
        import argparse

        args = argparse.Namespace(
            model="luna", cloud_api_key_env="TEST_CLOUD_KEY",
            cloud_api_key_file=None, cloud_base_url="https://api.openai.com/v1",
        )
        with tempfile.TemporaryDirectory() as directory:
            key_path = Path(directory) / ".config" / "dabtroll" / "openai_api_key"
            key_path.parent.mkdir(parents=True)
            key_path.write_text("file-key\n")
            with patch.object(Path, "home", return_value=Path(directory)):
                with patch.dict(os.environ, {"TEST_CLOUD_KEY": ""}):
                    self.assertEqual(mission_server._load_cloud_api_key(args), "file-key")
                    args.cloud_base_url = self.base_url
                    with self.assertRaisesRegex(ValueError, "No cloud API key"):
                        mission_server._load_cloud_api_key(args)
                with patch.dict(os.environ, {"TEST_CLOUD_KEY": "env-key"}):
                    self.assertEqual(mission_server._load_cloud_api_key(args), "env-key")
                    args.cloud_api_key_file = str(key_path)
                    self.assertEqual(mission_server._load_cloud_api_key(args), "file-key")
                    key_path.write_text("")
                    with self.assertRaises(ValueError):
                        mission_server._load_cloud_api_key(args)


if __name__ == "__main__":
    unittest.main()
