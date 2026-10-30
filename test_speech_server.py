"""Speech on the GPU host (app/speech_server.py): the request is checked, the routes are split, no model needed."""
import json
import unittest

from app import speech_server as s


class TestSpeechRequest(unittest.TestCase):
    def test_a_request_is_checked_and_a_foreign_voice_falls_back(self):
        got = s.parse_request(json.dumps({"model": "kokoro-82m", "input": "  Hello\n  world ", "voice": "alloy"}).encode())
        self.assertEqual({"text": "Hello world", "voice": "af_heart", "format": "mp3", "speed": 1.0}, got)
        self.assertEqual("af_bella", s.parse_request(json.dumps({"input": "x", "voice": "af_bella"}).encode())["voice"])

    def test_another_model_or_an_empty_text_is_refused_by_name(self):
        for body in ({"model": "tts-1", "input": "x"}, {"input": ""}, {"input": "x", "response_format": "flac"}):
            with self.subTest(body), self.assertRaises(s.SpeechRequestError):
                s.parse_request(json.dumps(body).encode())

    def test_a_passed_through_content_type_is_one_of_ours_never_copied(self):
        self.assertEqual("image/png", s.safe_type("image/png; charset=binary"))
        self.assertEqual("application/json", s.safe_type("text/html\r\nSet-Cookie: x=1"))
        self.assertEqual("application/json", s.safe_type(None))

    def test_a_proxied_path_cannot_choose_the_host_or_walk_out(self):
        self.assertEqual("http://127.0.0.1:8089/v1/images?x=1", s.upstream_url("127.0.0.1:8089", "/v1/images?x=1"))
        for bad in ("@evil.test/x", "//evil.test/x", "v1/x", "/a/../b", "/a\\b", "", "/a/%2e%2e/b", "/%2Fevil.test/x", "/a%5cb"):
            with self.subTest(bad), self.assertRaises(ValueError):
                s.upstream_url("127.0.0.1:8089", bad)

    def test_only_the_speech_path_is_answered_here(self):
        self.assertTrue(s.routes_to_speech("POST", "/v1/audio/speech"))
        self.assertFalse(s.routes_to_speech("POST", "/v1/images/generations"))
        self.assertFalse(s.routes_to_speech("GET", "/v1/audio/speech"))


if __name__ == "__main__":
    unittest.main()
