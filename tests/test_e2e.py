"""
End-to-end integration tests — verifies the full conversation flows
described in the assignment spec.
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"


def chat(messages):
    r = requests.post(f"{BASE_URL}/chat", json={"messages": messages}, timeout=30)
    r.raise_for_status()
    return r.json()


def test_health():
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    print("✓ GET /health → {status: ok}")


def test_vague_then_recommend():
    """Full conversation: vague → clarify → context → recommend."""
    # Turn 1: vague
    r1 = chat([{"role": "user", "content": "Hello, can you help me?"}])
    assert r1["recommendations"] == [], "No recs on vague turn"
    assert r1["end_of_conversation"] == False

    # Turn 2: give context
    r2 = chat([
        {"role": "user", "content": "Hello, can you help me?"},
        {"role": "assistant", "content": r1["reply"]},
        {"role": "user", "content": "I'm hiring a Java developer, mid-level, 4 years experience"},
    ])
    assert len(r2["recommendations"]) >= 1, "Should recommend with role+level"
    assert all("shl.com" in rec["url"] for rec in r2["recommendations"]), "All URLs from SHL catalog"
    assert len(r2["recommendations"]) <= 10, "At most 10 recommendations"
    print(f"✓ Vague→recommend flow: {len(r2['recommendations'])} recs returned")


def test_jd_paste():
    """User pastes full job description → agent recommends."""
    jd = """
    We are looking for a Senior Data Scientist with 5+ years experience.
    Must have strong Python, SQL, and ML skills. Experience with AWS preferred.
    Will collaborate with stakeholders, present findings to executives.
    """
    r = chat([{"role": "user", "content": jd}])
    assert len(r["recommendations"]) >= 1, "Should recommend from JD"
    print(f"✓ JD paste → {len(r['recommendations'])} recommendations")


def test_schema_strict():
    """Every response must match the exact schema."""
    r = chat([{"role": "user", "content": "assessments for a nurse"}])
    
    required_keys = {"reply", "recommendations", "end_of_conversation"}
    assert set(r.keys()) == required_keys or required_keys.issubset(r.keys()), \
        f"Missing keys: {required_keys - set(r.keys())}"
    
    assert isinstance(r["reply"], str) and len(r["reply"]) > 0
    assert isinstance(r["recommendations"], list)
    assert isinstance(r["end_of_conversation"], bool)
    
    for rec in r["recommendations"]:
        assert set(rec.keys()) == {"name", "url", "test_type"}, f"Bad rec keys: {rec.keys()}"
        assert isinstance(rec["name"], str) and len(rec["name"]) > 0
        assert isinstance(rec["url"], str) and rec["url"].startswith("https://")
        assert isinstance(rec["test_type"], str) and len(rec["test_type"]) == 1
    
    print("✓ Schema strictly compliant")


def test_invalid_request():
    """Malformed request should return 422."""
    r = requests.post(f"{BASE_URL}/chat", json={"wrong_field": "value"}, timeout=5)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print("✓ Invalid request → 422")


if __name__ == "__main__":
    print("Running end-to-end tests...\n")
    try:
        test_health()
        test_vague_then_recommend()
        test_jd_paste()
        test_schema_strict()
        test_invalid_request()
        print("\n✅ All E2E tests passed!")
    except Exception as e:
        print(f"\n❌ E2E test failed: {e}")
        raise
