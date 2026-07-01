import pytest

from app import create_app, db


@pytest.fixture
def client():
    app = create_app('testing')
    with app.app_context():
        db.create_all()
        with app.test_client() as client:
            yield client


def test_index_page(client):
    response = client.get('/')
    assert response.status_code == 200


def test_dashboard_page(client):
    response = client.get('/dashboard')
    assert response.status_code == 200


def test_optimization_unknown_method(client):
    response = client.post('/api/optimization/run', json={'method': 'inconnu'})
    assert response.status_code == 400
