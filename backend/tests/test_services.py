from app.services.recommendations import get_recommendations


def test_get_recommendations():
    recs = get_recommendations("user-1")
    assert len(recs) >= 1
    assert recs[0].symbol


