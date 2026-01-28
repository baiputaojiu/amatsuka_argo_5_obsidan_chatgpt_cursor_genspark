# 初期ビュー設定機能の実装ガイド

## 必要な変更点

### 1. レイアウトに追加する要素

```python
# 「初期ビューを設定」ボタンを追加
html.Button(
    '初期ビューを設定',
    id='set-initial-view-button',
    n_clicks=0,
    style={
        'backgroundColor': DARK_THEME['secondary'],
        'color': 'white',
        'border': 'none',
        'padding': '10px 20px',
        'borderRadius': '5px',
        'cursor': 'pointer',
        'marginBottom': '10px',
        'marginRight': '10px',
        'fontSize': '14px',
        'fontWeight': 'bold'
    }
),
# 初期ビューを保存するStoreコンポーネント
dcc.Store(id='stored-initial-view', data=None)
```

### 2. コールバックの追加

```python
# 初期ビューを設定するコールバック
@callback(
    Output('stored-initial-view', 'data'),
    Input('set-initial-view-button', 'n_clicks'),
    State('main-3d-graph', 'relayoutData'),
    prevent_initial_call=True
)
def set_initial_view(n_clicks, relayout_data):
    """現在のカメラ位置を初期ビューとして保存"""
    if n_clicks and n_clicks > 0 and relayout_data:
        if 'scene.camera' in relayout_data:
            return relayout_data['scene.camera']
        elif isinstance(relayout_data, dict) and 'eye' in relayout_data:
            return relayout_data
    return dash.no_update
```

### 3. update_graphsコールバックの修正

```python
# Stateに追加
State('stored-initial-view', 'data')

# 関数シグネチャに追加
def update_graphs(selected_country, hover_data, click_data, reset_clicks, relayout_data, stored_click_data, stored_initial_view):

# カメラ状態の決定ロジックを変更
camera_state = None
if reset_clicks and reset_clicks > 0:
    # リセットボタンが押された場合は保存された初期ビューを使用
    camera_state = stored_initial_view if stored_initial_view else None
elif relayout_data:
    # 通常は現在のカメラ状態を保持
    if 'scene.camera' in relayout_data:
        camera_state = relayout_data['scene.camera']
    elif isinstance(relayout_data, dict) and 'eye' in relayout_data:
        camera_state = relayout_data
```

### 4. create_3d_surface関数の修正

```python
# 固定の初期ビュー位置の計算コードを削除
# 代わりに、camera_stateがNoneの場合のデフォルト値を設定
camera=camera_state if camera_state else dict(
    # デフォルトの初期ビュー（保存された初期ビューがない場合）
    eye=dict(x=-1.5, y=1.5, z=1.2),
    center=dict(x=0, y=0, z=0),
    up=dict(x=0, y=0, z=1)
),
```
