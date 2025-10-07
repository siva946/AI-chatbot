import unittest
from app import app

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_generate_text_no_prompt(self):
        response = self.app.post('/generate_text', json={})
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()