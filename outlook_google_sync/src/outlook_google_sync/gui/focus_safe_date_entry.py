"""Focus-safe DateEntry widget.

tkcalendar 1.5.0 の ``DateEntry`` は Windows 上で以下の症状を抱えている。

- ドロップダウンカレンダー（``overrideredirect(True)`` の Toplevel）を開いた状態で、
  月送り／年送りの矢印ボタン（``_l_month`` / ``_r_month`` / ``_l_year`` / ``_r_year``）を
  押すと、``_select`` が過去に呼ばれていた（＝1度以上日付を選択した）場合、
  ``<FocusOut>`` ハンドラ ``_on_focus_out_cal`` が ``focus_get() == self``（Entry）と
  判定し、``self._top_cal.withdraw()`` でポップアップを閉じてしまう。
- その結果、ボタンの ``<ButtonRelease-1>`` が既に withdraw 済みの領域へ届き、
  ``command=_prev_month`` 等が発火しないため「月が変わらずプルダウンだけ消える」。

``FocusSafeDateEntry`` は ``_on_focus_out_cal`` を差し替え、
マウスポインタがドロップダウンカレンダーの矩形内にある間は閉じず、
外に出たときのみ withdraw する挙動にする。矢印ボタンを連続で押しても
ポップアップが維持され、``command`` が通常どおり発火する。
"""

from __future__ import annotations

from tkcalendar import DateEntry


class FocusSafeDateEntry(DateEntry):
    """月／年矢印ボタン連打でポップアップが消えないようにした DateEntry。"""

    def _on_focus_out_cal(self, event):  # type: ignore[override]
        try:
            px, py = self._top_cal.winfo_pointerxy()
            cx = self._top_cal.winfo_rootx()
            cy = self._top_cal.winfo_rooty()
            w = self._top_cal.winfo_width()
            h = self._top_cal.winfo_height()
        except Exception:
            self._top_cal.withdraw()
            self.state(["!pressed"])
            return

        if cx <= px <= cx + w and cy <= py <= cy + h:
            self._calendar.focus_force()
            return

        self._top_cal.withdraw()
        self.state(["!pressed"])
