"""Дымовые тесты интерфейса через streamlit.testing (без браузера)."""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def _app():
    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    return at


def _click(at, label_part):
    for b in at.button:
        if label_part in b.label:
            b.click()
            return
    raise AssertionError(f"нет кнопки {label_part!r}: {[b.label for b in at.button]}")


def test_app_starts_without_errors():
    at = _app()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Запустить" in b.label for b in at.button)


def test_full_run_instant_speed_produces_results():
    at = _app()
    at.session_state["speed"] = "Мгновенно (только итог)"
    at.session_state["w::n_months"] = 36
    at.run()
    _click(at, "Запустить")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["finished"] is True
    assert len(at.session_state["records"]) == 36
    # после завершения скрипт перерисован: появились вкладки с результатами
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_preset_change_resets_and_loads_values():
    at = _app()
    at.sidebar.selectbox[0].select("random_jumps").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["w::scenario.kind"] == "random_jumps"
    assert at.session_state["model"] is None


def test_step_button_and_shock_buttons():
    at = _app()
    _click(at, "Шаг")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["model"] is not None
    _click(at, "Ставка +3")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["model"].interventions[-1]["type"] == "key_rate"


def _by_label(items, part):
    for w in items:
        if part in w.label:
            return w
    raise AssertionError(f"нет виджета {part!r}: {[w.label for w in items]}")


def _finished_app(n_months=36):
    at = _app()
    at.session_state["speed"] = "Мгновенно (только итог)"
    at.session_state["w::n_months"] = n_months
    at.run()
    _click(at, "Запустить")
    at.run()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_result_tabs_after_run_render():
    at = _finished_app()
    texts = " ".join(m.value for m in at.markdown)
    assert "Передача ставки" in texts or "Сценарий" in texts          # вкладка интерпретации
    assert any("Критерии" in s.value for s in at.subheader)
    assert len(at.dataframe) >= 3                                      # таблица банков, проверки, события


def test_ensemble_tab_runs():
    at = _finished_app(24)
    _by_label(at.number_input, "Число прогонов").set_value(6)
    at.run()
    _click(at, "Запустить ансамбль")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["ensemble"] is not None and at.session_state["ensemble"]["run"].nunique() == 6


def test_calibration_tab_moments_and_fit():
    at = _finished_app(24)
    _click(at, "Моменты при текущем конфиге")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["calib"]["moments"] is not None
    _by_label(at.slider, "Бюджет").set_value(40)
    _by_label(at.slider, "Зёрен").set_value(2)
    at.run()
    _click(at, "Откалибровать")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["calib"]["theta"]


def test_synthetic_tab_generates():
    at = _finished_app(24)
    at.number_input(key="syn_n").set_value(10)
    at.run()
    _click(at, "Сгенерировать синтетику")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["synth"]["data"]["run"].nunique() == 10
