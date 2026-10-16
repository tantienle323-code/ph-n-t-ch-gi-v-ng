import pandas as pd


def prepare_walk_forward_backtest_data(data, oos_predictions):
    """Align stored out-of-sample signals with prices, including the final exit price."""
    if 'Date' not in data.columns:
        raise ValueError("Dữ liệu backtest thiếu cột Date.")
    if not oos_predictions:
        raise ValueError("Chưa có dự báo ngoài mẫu walk-forward.")

    date_values = pd.to_datetime(data['Date']).dt.normalize().reset_index(drop=True)
    predictions_by_date = {}
    for item in oos_predictions:
        prediction_date = pd.Timestamp(item['date']).normalize()
        if prediction_date in predictions_by_date:
            raise ValueError("Danh sách dự báo walk-forward có ngày bị trùng.")
        predictions_by_date[prediction_date] = int(item['prediction'])

    prediction_dates = list(predictions_by_date)
    if prediction_dates != sorted(prediction_dates):
        raise ValueError("Dự báo walk-forward không được sắp xếp theo ngày.")

    positions_by_date = {
        current_date: index
        for index, current_date in enumerate(date_values)
    }
    missing_dates = [current_date for current_date in prediction_dates if current_date not in positions_by_date]
    if missing_dates:
        raise ValueError("Dữ liệu hiện tại không khớp ngày với dự báo walk-forward; hãy huấn luyện lại.")

    positions = [positions_by_date[current_date] for current_date in prediction_dates]
    if positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError("Các ngày dự báo walk-forward không liên tục trong dữ liệu.")
    if positions[-1] + 1 >= len(data):
        raise ValueError("Thiếu giá phiên kế tiếp để tính lợi nhuận của giai đoạn kiểm định.")

    backtest_data = data.iloc[positions[0]:positions[-1] + 2].copy().reset_index(drop=True)
    signals = [predictions_by_date[current_date] for current_date in prediction_dates]
    signals.append(0)
    return backtest_data, signals


def run_backtest(
    test_data,
    signals,
    initial_capital,
    stop_loss_pct,
    take_profit_pct,
    transaction_cost_bps=0.0,
    slippage_bps=0.0,
    volatility_position_sizing=True,
):
    """Simulate one-day long trades with per-side costs in basis points."""
    if len(test_data) < 2:
        raise ValueError("Backtest cần ít nhất hai phiên dữ liệu.")
    if len(signals) != len(test_data):
        raise ValueError("Số tín hiệu phải bằng số dòng dữ liệu backtest.")
    if initial_capital <= 0:
        raise ValueError("Vốn ban đầu phải lớn hơn 0.")
    if min(stop_loss_pct, take_profit_pct, transaction_cost_bps, slippage_bps) < 0:
        raise ValueError("Stop-loss, take-profit, phí và trượt giá không được âm.")

    result = test_data.copy().reset_index(drop=True)
    result['Signal'] = signals
    result['Gross_Return'] = 0.0
    result['Trading_Cost_Return'] = 0.0
    result['Net_Return'] = 0.0
    capital = float(initial_capital)
    portfolio = [capital]
    round_trip_cost = 2 * (transaction_cost_bps + slippage_bps) / 10_000

    for index in range(len(result) - 1):
        signal = result.loc[index, 'Signal']
        scale = 1.0
        if volatility_position_sizing and result.loc[index].get('Volatility_Spike_Flag', 0) == 1:
            scale = 0.5

        if signal == 1:
            current_price = result.loc[index, 'Close']
            next_price = result.loc[index + 1, 'Close']
            raw_return = (next_price - current_price) / current_price
            gross_return = max(-stop_loss_pct, min(raw_return, take_profit_pct))
            cost_return = round_trip_cost
            net_return = scale * (gross_return - cost_return)
            result.loc[index, 'Gross_Return'] = gross_return
            result.loc[index, 'Trading_Cost_Return'] = cost_return
            result.loc[index, 'Net_Return'] = net_return
            capital *= 1 + net_return

        portfolio.append(capital)

    result['AI_Portfolio'] = portfolio[:len(result)]
    return result
