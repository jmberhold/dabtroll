"""CPU-only adapter and ZMQ contract tests against a mock vLLM HTTP server."""

import argparse
import base64
import io
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

import zmq
from openai import APIStatusError, APITimeoutError
from httpx import Request
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import mission_engine
import mission_engine_server1 as mission_server
from mission_engine import QwenContext, VllmClient


class VllmTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.models = ["qwen38"]
        self.completion = {
            "choices": [{"message": {"content": "OK", "reasoning": "not the answer"}, "finish_reason": "stop"}]
        }
        self.http_status = 200
        test = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": model} for model in test.models]}).encode())

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                test.requests.append((self.path, self.headers["Authorization"], payload))
                self.send_response(test.http_status)
                self.end_headers()
                self.wfile.write(json.dumps(test.completion).encode())

            def log_message(self, *args):
                pass

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.client = VllmClient(base_url=f"http://127.0.0.1:{self.http.server_port}/v1")
        self.ctx = QwenContext(enabled=True, model_family="qwen3_8", vllm_client=self.client)

    def tearDown(self):
        self.client._client.close()
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()

    def test_text_payload_matches_notebook(self):
        self.client.check_health()
        self.assertEqual(mission_engine.generate_text_only_response(self.ctx, "hello", 123), "OK")
        path, auth, payload = self.requests[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(auth, "Bearer EMPTY")
        self.assertEqual(payload, {
            "model": "qwen38",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            "max_tokens": 123,
            "temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5,
            "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        self.client.enable_thinking = True
        self.client.generate("hello", [], 123)
        self.assertTrue(self.requests[-1][2]["chat_template_kwargs"]["enable_thinking"])

    def _last_image(self):
        content = self.requests[-1][2]["messages"][0]["content"]
        images = [item for item in content if item["type"] == "image_url"]
        self.assertEqual(len(images), 1)
        url = images[0]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        self.assertEqual(image.mode, "RGB")
        self.assertLessEqual(max(image.size), 512)
        return image, content[-1]["text"]

    def test_image_and_path_share_encoding(self):
        image = Image.new("RGBA", (1200, 600), (255, 0, 0, 255))
        self.assertEqual(mission_engine.generate_text_image_response(self.ctx, image, "look", 77), "OK")
        encoded, prompt = self._last_image()
        self.assertEqual(encoded.size, (512, 256))
        self.assertEqual(prompt, "look")
        self.assertEqual(image.size, (1200, 600))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            image.save(path)
            self.assertEqual(
                mission_engine.generate_text_image_response_from_path(self.ctx, str(path), "look", 77),
                "OK",
            )
        self.assertEqual(self._last_image()[0].size, (512, 256))
        self.assertEqual(self.requests[-1][2]["max_tokens"], 77)

    def test_video_uses_one_chronological_contact_sheet(self):
        frames = [Image.new("RGB", (300, 300), color) for color in ("red", "green", "blue")]
        self.assertEqual(mission_engine.generate_text_video_response(self.ctx, frames, "status", 91), "OK")
        image, prompt = self._last_image()
        self.assertIn("3 chronological frames", prompt)
        self.assertIn("Frame 3 is newest", prompt)
        self.assertTrue(prompt.endswith("status"))
        self.assertGreater(image.getpixel((100, 100))[0], 240)
        self.assertGreater(image.getpixel((350, 100))[1], 110)
        self.assertGreater(image.getpixel((100, 350))[2], 240)
        self.assertEqual(self.requests[-1][2]["max_tokens"], 91)

    def test_invalid_configuration_and_model_mismatch(self):
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                VllmClient(timeout_s=timeout)
        with self.assertRaises(ValueError):
            VllmClient(base_url="not-a-url")
        self.models = ["other-model"]
        with self.assertRaisesRegex(RuntimeError, "does not advertise"):
            self.client.check_health()

    def test_backend_errors_are_not_successful_completions(self):
        for completion, message in (
            ({"choices": []}, "no valid completion"),
            ({"choices": [{"message": {"content": None, "reasoning": "thinking"}}]}, "no assistant content"),
            ({"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}, "token limit"),
        ):
            with self.subTest(completion=completion):
                self.completion = completion
                with self.assertRaisesRegex(RuntimeError, message):
                    self.client.generate("hello", [], 10)
        self.http_status = 400
        self.completion = {"error": {"message": "maximum context length exceeded"}}
        with self.assertRaisesRegex(APIStatusError, "400.*maximum context"):
            self.client.generate("hello", [], 10)
        self.http_status = 200
        with patch.object(
            self.client._client.chat.completions, "create",
            side_effect=APITimeoutError(request=Request("POST", self.client.base_url)),
        ):
            with self.assertRaises(APITimeoutError):
                self.client.generate("hello", [], 10)

    def test_local_and_disabled_contexts_preserve_dispatch(self):
        local = QwenContext(enabled=True)
        with patch.object(mission_engine, "run_qwen_generation", return_value="local") as generate:
            self.assertEqual(mission_engine.generate_text_only_response(local, "hello", 99), "local")
            generate.assert_called_once_with(local, mission_engine.build_text_messages("hello"), max_new_tokens=99)
        with patch.object(mission_engine, "run_qwen_generation_with_image", return_value="image") as generate:
            self.assertEqual(mission_engine.generate_text_image_response(local, "pixels", "look", 99), "image")
            generate.assert_called_once()
        with patch.object(mission_engine, "run_qwen_generation_with_image_path", return_value="path"):
            self.assertEqual(mission_engine.generate_text_image_response_from_path(local, "frame.jpg", "look"), "path")
        with patch.object(mission_engine, "run_qwen_generation_with_images", return_value="video"):
            self.assertEqual(mission_engine.generate_text_video_response(local, ["frame"], "look"), "video")
        self.ctx.enabled = False
        self.assertIsNone(mission_engine.generate_text_only_response(self.ctx, "hello"))
        self.assertEqual(self.requests, [])

    def test_release_closes_http_client_without_stopping_vllm(self):
        with patch.object(self.client._client, "close") as close:
            mission_engine.release_qwen_gpu(self.ctx)
            close.assert_called_once()
        self.assertFalse(self.ctx.enabled)
        self.assertIsNone(self.ctx.vllm_client)
        self.client.check_health()

    def test_zmq_server_contract(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        context = zmq.Context()
        sock = context.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 5000)
        sock.connect(f"tcp://127.0.0.1:{port}")
        with tempfile.TemporaryFile(mode="w+") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", str(SCRIPTS / "mission_engine_server1.py"),
                 "--model", "qwen38", "--port", str(port),
                 "--vllm_external", "--vllm_base_url", self.client.base_url, "--vllm_timeout_s", "2"],
                stdout=log, stderr=subprocess.STDOUT,
            )
            try:
                sock.send_json({"mode": "health"})
                deadline = time.monotonic() + 60
                while not sock.poll(100):
                    if process.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        self.fail(f"ZMQ server did not become ready:\n{log.read()}")
                health = sock.recv_json()
                self.assertTrue(health["ok"], health)
                self.assertEqual(health["backend"], "vllm")
                self.assertEqual(health["model_family"], "qwen3_8")
                self.assertEqual(health["vllm_model"], "qwen38")
                self.assertFalse(health["vllm_managed"])
                self.assertIsNone(health["vllm_pid"])
                self.assertEqual(health["video_input"], "contact_sheet")
                self.assertNotIn("vllm_api_key", health)

                def request(payload):
                    sock.send_json(payload)
                    return sock.recv_json()

                self.assertTrue(request({"mode": "text", "prompt_text": "hello", "max_new_tokens": 42})["ok"])
                self.assertEqual(self.requests[-1][2]["max_tokens"], 42)
                buffer = io.BytesIO()
                Image.new("RGB", (32, 32), "red").save(buffer, "JPEG")
                frame = base64.b64encode(buffer.getvalue()).decode()
                for mode in ("bt", "text_image", "text_video", "video", "scene"):
                    with self.subTest(mode=mode):
                        result = request({
                            "mode": mode, "frames_b64": [frame, frame], "prompt_text": "status",
                            "task_text": "Pick up the cup.", "bt_variant": "btaudit", "bt_max_new_tokens": 111,
                        })
                        self.assertTrue(result["ok"], result)
                        self.assertEqual(result["text"], "OK")
                        if mode == "bt":
                            self.assertEqual(result["bt_max_new_tokens"], 111)
                            self.assertEqual(self.requests[-1][2]["max_tokens"], 111)
                        self._last_image()
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "frame.jpg"
                    Image.new("RGB", (32, 32), "blue").save(path)
                    for mode in ("bt", "text_image", "scene"):
                        self.assertTrue(request({
                            "mode": mode, "image_path": str(path),
                            "task_text": "Pick up the cup.", "prompt_text": "look",
                        })["ok"])
                self.http_status = 400
                result = request({"mode": "text", "prompt_text": "bad"})
                self.assertFalse(result["ok"])
                self.assertIn("400", result["error"])
                self.http_status = 200
                self.models = []
                self.assertFalse(request({"mode": "health"})["ok"])
                self.models = ["qwen38"]
                self.assertTrue(request({"mode": "health"})["ok"])
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                sock.close()
                context.term()


class ManagedVllmTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.args = argparse.Namespace(
            vllm_base_url=f"http://127.0.0.1:{port}/v1",
            vllm_model_path=str(self.directory),
            vllm_model="qwen38",
            vllm_api_key="EMPTY",
            vllm_startup_timeout_s=30.0,
        )
        self.client = Mock(spec=VllmClient)
        self.client.base_url = self.args.vllm_base_url
        self.process = Mock()
        self.process.pid = 12345
        self.process.poll.return_value = None

    def test_managed_command_and_environment_match_tested_launch(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1", "LD_LIBRARY_PATH": "/conflicting/lib"}):
            with patch.object(mission_server.subprocess, "Popen", return_value=self.process) as start:
                with patch.object(mission_server, "_stop_vllm") as stop:
                    with mission_server._managed_vllm(self.args, self.client) as process:
                        self.assertIs(process, self.process)
                    stop.assert_called_once_with(self.process)
        command = start.call_args.args[0]
        self.assertEqual(command, [
            sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(self.directory),
            "--served-model-name", "qwen38", "--host", "127.0.0.1",
            "--port", str(self.args.vllm_base_url.split(":")[2].split("/")[0]),
            "--api-key", "EMPTY", "--dtype", "bfloat16",
            "--max-model-len", "4096", "--max-num-seqs", "1",
            "--gpu-memory-utilization", "0.90",
            "--limit-mm-per-prompt", '{"image":1,"video":0}',
            "--reasoning-parser", "qwen3", "--enforce-eager", "--skip-mm-profiling",
        ])
        self.assertEqual(start.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
        self.assertNotIn("LD_LIBRARY_PATH", start.call_args.kwargs["env"])
        self.assertTrue(start.call_args.kwargs["start_new_session"])
        self.client.check_health.assert_called_once_with(timeout_s=2.0)

    def test_startup_exit_timeout_and_model_errors_clean_up(self):
        for error in ("exit", "timeout", "model"):
            with self.subTest(error=error):
                self.process.poll.return_value = 7 if error == "exit" else None
                self.client.check_health.side_effect = RuntimeError("wrong model") if error == "model" else None
                with patch.object(mission_server.subprocess, "Popen", return_value=self.process):
                    with patch.object(mission_server, "_stop_vllm") as stop:
                        times = [0, 31] if error == "timeout" else [0, 0]
                        with patch.object(mission_server.time, "monotonic", side_effect=times):
                            with self.assertRaises((RuntimeError, TimeoutError)):
                                with mission_server._managed_vllm(self.args, self.client):
                                    self.fail("unready backend was accepted")
                        stop.assert_called_once_with(self.process)

    def test_waits_for_connection_then_cleans_up_on_interrupt(self):
        from openai import APIConnectionError

        self.client.check_health.side_effect = [
            APIConnectionError(request=Request("GET", self.client.base_url)),
            None,
        ]
        with patch.object(mission_server.subprocess, "Popen", return_value=self.process):
            with patch.object(mission_server, "_stop_vllm") as stop:
                with patch.object(mission_server.time, "sleep"):
                    with self.assertRaises(KeyboardInterrupt):
                        with mission_server._managed_vllm(self.args, self.client):
                            raise KeyboardInterrupt
                self.assertEqual(self.client.check_health.call_count, 2)
                stop.assert_called_once_with(self.process)

    def test_rejects_busy_port_missing_checkpoint_and_nonlocal_endpoint(self):
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            self.args.vllm_base_url = f"http://127.0.0.1:{occupied.getsockname()[1]}/v1"
            self.client.base_url = self.args.vllm_base_url
            with patch.object(mission_server.subprocess, "Popen") as start:
                with self.assertRaisesRegex(RuntimeError, "existing service"):
                    with mission_server._managed_vllm(self.args, self.client):
                        self.fail("busy port was accepted")
                start.assert_not_called()
        self.args.vllm_model_path = str(self.directory / "missing")
        with self.assertRaises(FileNotFoundError):
            mission_server._vllm_launch_command(self.args)
        self.args.vllm_model_path = str(self.directory)
        for url in ("http://example.com:8000/v1", "http://127.0.0.1:8000/not-v1"):
            self.args.vllm_base_url = url
            with self.assertRaises(ValueError):
                mission_server._vllm_launch_command(self.args)
        self.args.vllm_startup_timeout_s = 0
        with self.assertRaises(ValueError):
            with mission_server._managed_vllm(self.args, self.client):
                self.fail("invalid timeout was accepted")

    def test_shutdown_targets_owned_process_group_and_escalates(self):
        with patch.object(mission_server.os, "killpg") as kill:
            mission_server._stop_vllm(self.process)
            kill.assert_called_once_with(self.process.pid, signal.SIGTERM)
        self.process.wait.side_effect = [subprocess.TimeoutExpired("vllm", 15), 0]
        with patch.object(mission_server.os, "killpg") as kill:
            mission_server._stop_vllm(self.process)
            self.assertEqual(kill.call_args_list[-1].args, (self.process.pid, signal.SIGKILL))

    def test_one_command_starts_serves_and_stops_backend(self):
        self._assert_one_command_lifecycle()

    def test_backend_exit_stops_mission_server(self):
        self._assert_one_command_lifecycle(backend_failure=True)

    def test_ctrl_c_stops_managed_backend(self):
        self._assert_one_command_lifecycle(shutdown_signal=signal.SIGINT)

    def _assert_one_command_lifecycle(self, backend_failure=False, shutdown_signal=signal.SIGTERM):
        package = self.directory / "vllm" / "entrypoints" / "cli"
        package.mkdir(parents=True)
        for folder in (package, package.parent, package.parent.parent):
            (folder / "__init__.py").write_text("")
        (package / "main.py").write_text(
            "import argparse, json, os, sys\n"
            "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
            "from pathlib import Path\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--port', type=int)\n"
            "parser.add_argument('--host')\n"
            "args, _ = parser.parse_known_args()\n"
            "Path(os.environ['TEST_VLLM_STATE']).write_text(json.dumps({\n"
            "    'pid': os.getpid(), 'argv': sys.argv,\n"
            "    'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),\n"
            "    'ld_library_path': os.environ.get('LD_LIBRARY_PATH')}))\n"
            "class Handler(BaseHTTPRequestHandler):\n"
            "    def do_GET(self):\n"
            "        self.send_response(200)\n"
            "        self.end_headers()\n"
            "        self.wfile.write(b'{\"data\":[{\"id\":\"qwen38\"}]}')\n"
            "    def do_POST(self):\n"
            "        self.rfile.read(int(self.headers['Content-Length']))\n"
            "        self.send_response(200)\n"
            "        self.end_headers()\n"
            "        self.wfile.write(b'{\"choices\":[{\"message\":{\"content\":\"OK\"},\"finish_reason\":\"stop\"}]}')\n"
            "    def log_message(self, *args): pass\n"
            "HTTPServer((args.host, args.port), Handler).serve_forever()\n"
        )
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            mission_port = probe.getsockname()[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.directory)
        env["CUDA_VISIBLE_DEVICES"] = "1"
        env["TEST_VLLM_STATE"] = str(self.directory / "backend.json")
        context = zmq.Context()
        sock = context.socket(zmq.REQ)
        sock.LINGER = 0
        sock.RCVTIMEO = 5000
        sock.connect(f"tcp://127.0.0.1:{mission_port}")
        with tempfile.TemporaryFile(mode="w+") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", str(SCRIPTS / "mission_engine_server1.py"),
                 "--model", "qwen38", "--port", str(mission_port),
                 "--vllm_model_path", str(self.directory),
                 "--vllm_base_url", self.args.vllm_base_url,
                 "--vllm_startup_timeout_s", "15"],
                env=env, stdout=log, stderr=subprocess.STDOUT,
            )
            backend_pid = None
            try:
                sock.send_json({"mode": "health"})
                deadline = time.monotonic() + 60
                while not sock.poll(100):
                    if process.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        self.fail(f"Managed server failed to become ready:\n{log.read()}")
                health = sock.recv_json()
                self.assertTrue(health["ok"], health)
                self.assertTrue(health["vllm_managed"])
                backend_pid = health["vllm_pid"]
                state = json.loads(Path(env["TEST_VLLM_STATE"]).read_text())
                self.assertEqual(state["pid"], backend_pid)
                self.assertEqual(state["gpu"], "1")
                self.assertIsNone(state["ld_library_path"])
                self.assertIn("--skip-mm-profiling", state["argv"])
                sock.send_json({"mode": "text", "prompt_text": "hello"})
                self.assertEqual(sock.recv_json()["text"], "OK")
                if backend_failure:
                    os.kill(backend_pid, signal.SIGTERM)
                    process.wait(timeout=5)
                    self.assertNotEqual(process.returncode, 0)
            finally:
                process.send_signal(shutdown_signal)
                try:
                    process.wait(timeout=20)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=5)
                    sock.close()
                    context.term()
                    if backend_pid is None and Path(env["TEST_VLLM_STATE"]).exists():
                        backend_pid = json.loads(Path(env["TEST_VLLM_STATE"]).read_text())["pid"]
                    if backend_pid is not None:
                        try:
                            os.kill(backend_pid, 0)
                        except ProcessLookupError:
                            pass
                        else:
                            os.killpg(backend_pid, signal.SIGKILL)
                            self.fail("Managed vLLM process survived mission server shutdown")


if __name__ == "__main__":
    unittest.main()
