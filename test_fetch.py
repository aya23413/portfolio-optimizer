from app import create_app

app = create_app('development')

with app.app_context():
    from app.services.data_collector import fetch_historical_data
    result = fetch_historical_data(['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL'], '2020-01-01', '2025-01-01')
    print(result)