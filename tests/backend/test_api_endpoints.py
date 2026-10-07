"""
Integration tests for FastAPI endpoints.
"""

import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.app.main import app
from backend.app.services.ml_service import ml_service



class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ml_service.load_models()
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

    def test_health_check(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["yield_model_loaded"])
        self.assertTrue(data["recommendation_model_loaded"])

    def test_get_crops(self):
        res = self.client.get("/api/crops")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("crops", data)
        self.assertGreater(len(data["crops"]), 0)

    def test_get_states(self):
        res = self.client.get("/api/states")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("states", data)
        self.assertGreater(len(data["states"]), 0)

    def test_predict_yield_endpoint(self):
        payload = {
            "crop": "Rice",
            "season": "Kharif",
            "state": "Punjab",
            "area": 10.0,
            "fertilizer": 1500.0,
            "pesticide": 25.0,
            "year": 2024,
            "N": 150.0,
            "P": 50.0,
            "K": 40.0,
            "pH": 8.0,
            "avg_temp_c": 25.0,
            "total_rainfall_mm": 800.0,
            "avg_humidity_percent": 65.0,
        }
        res = self.client.post("/api/predict-yield", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("predicted_yield_tonnes_ha", data)
        self.assertGreater(data["predicted_yield_tonnes_ha"], 0)
        self.assertEqual(data["crop"], "Rice")

    def test_recommend_crop_endpoint(self):
        payload = {
            "N": 90,
            "P": 42,
            "K": 43,
            "ph": 6.5,
            "temperature": 20.87,
            "humidity": 82.0,
            "rainfall": 202.93,
        }
        res = self.client.post("/api/recommend-crop", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("recommended_crop", data)
        self.assertIn("top_crops", data)

    def test_auth_and_agent_endpoints(self):
        import uuid
        unique_email = f"farmer_{uuid.uuid4().hex[:8]}@example.com"
        
        # 1. Register
        reg_res = self.client.post("/api/auth/register", json={
            "name": "Test Farmer",
            "email": unique_email,
            "password": "FarmPassword123"
        })
        self.assertEqual(reg_res.status_code, 201)
        reg_data = reg_res.json()
        self.assertIn("access_token", reg_data)
        self.assertEqual(reg_data["user"]["email"], unique_email)

        # 2. Login
        login_res = self.client.post("/api/auth/login", json={
            "email": unique_email,
            "password": "FarmPassword123"
        })
        self.assertEqual(login_res.status_code, 200)
        token = login_res.json()["access_token"]

        # 3. Protected Me endpoint
        me_res = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(me_res.status_code, 200)
        self.assertEqual(me_res.json()["email"], unique_email)

        # 4. Agent chat
        agent_res = self.client.post("/api/agent/chat", json={
            "message": "Which crop is best with N=120, P=45, K=35, pH=7.5 in Punjab?"
        })
        self.assertEqual(agent_res.status_code, 200)
        self.assertTrue(len(agent_res.json()["reply"]) > 0)


if __name__ == "__main__":
    unittest.main()

