"""Выгрузка месячных рядов Банка России для калибровки (шаг 3 плана).

Скачивает с cbr.ru и собирает ``abm_banks/data/cbr_monthly.csv`` и
``abm_banks/data/key_rate_monthly.csv``:

* кредиты и вклады (Обзор кредитных организаций, ``survey_odc.xlsx``, с 2001 г.);
* средневзвешенные ставки по новым кредитам и вкладам (с января 2014 г.);
* ключевая ставка (среднее за месяц; с сентября 2013 г.);
* дефицит/профицит ликвидности и требования ЦБ к банкам (с 2014 г., среднее за месяц).

Запуск (скачанные файлы — недоверенные данные, поэтому изоляция интерпретатора)::

    python -I scripts/fetch_cbr_data.py

Файлы скачиваются во временную папку и удаляются. Скрипт проверяет подписи строк
Excel-файлов: если ЦБ изменил разметку, он остановится с понятной ошибкой, а не
подставит чужой ряд.

ПРЕДУПРЕЖДЕНИЕ о ключевой ставке до сентября 2013 г.: ЦБ публикует её с 17.09.2013.
Для месяцев июнь 2010 – август 2013 в таблицу подставляется ставка рефинансирования
(7,75; 8,0; 8,25; 8,0; 8,25 %) из публичной истории решений ЦБ; значения внесены
вручную и помечены ``key_rate_source = refi_manual``. Сверьте их с cbr.ru
перед публикацией результатов.
"""
from __future__ import annotations

import re
import shutil
import ssl
import sys
import tempfile
import urllib.request
from datetime import date, datetime
from pathlib import Path

import openpyxl
import pandas as pd

BASE = "https://www.cbr.ru"
UA = {"User-Agent": "Mozilla/5.0 (research data fetch)"}
OUT_DIR = Path(__file__).resolve().parents[1] / "abm_banks" / "data"

MONTHS = {"январь": 1, "февраль": 2, "март": 3, "апрель": 4, "май": 5, "июнь": 6, "июль": 7,
          "август": 8, "сентябрь": 9, "октябрь": 10, "ноябрь": 11, "декабрь": 12}

# Ставка рефинансирования до введения ключевой ставки (вручную, сверить с cbr.ru)
# Ставка рефинансирования до введения ключевой ставки: (с месяца, по месяц, %). Вручную, сверить с cbr.ru
REFI_RANGES = [("2010-06", "2011-02", 7.75), ("2011-03", "2011-04", 8.0), ("2011-05", "2011-12", 8.25),
               ("2012-01", "2012-08", 8.0), ("2012-09", "2013-08", 8.25)]


def _ssl_context() -> ssl.SSLContext:
    """Проверка сертификата включена; корневые сертификаты берём из certifi (python.org-сборки
    для macOS часто не содержат системных)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download(url: str, dest: Path) -> Path:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120, context=_ssl_context()) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    return dest


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120, context=_ssl_context()) as r:
        return r.read().decode("utf-8", errors="ignore")


# --------------------------------------------------------------------------
def parse_survey_odc(path: Path) -> pd.DataFrame:
    """Запасы на 1-е число месяца → значения на конец предыдущего месяца, млрд руб."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(_find_sheet(wb, "Агрегированная форма обзора").iter_rows(values_only=True))
    hdr = rows[0]

    def label(i: int) -> str:                       # i — номер строки Excel (с 1)
        return str(rows[i - 1][0] or "").strip().lower()

    def expect(i: int, text: str) -> None:
        if not label(i).startswith(text.lower()):
            raise RuntimeError(f"строка {i}: ожидалось «{text}», найдено «{label(i)}» — "
                               "ЦБ изменил разметку survey_odc.xlsx")

    expect(53, "Требования к нефинансовым организациям"); expect(56, "кредиты и займы")
    expect(60, "кредиты и займы"); expect(62, "Требования к домашним хозяйствам")
    expect(65, "кредиты и займы"); expect(69, "кредиты и займы")
    expect(72, "Депозиты, включаемые в широкую денежную массу")
    for r in (76, 80, 85, 89):
        expect(r, "Нефинансовые организации")
    for r in (77, 81, 86, 90):
        expect(r, "Домашние хозяйства")
    expect(71, "Обязательства перед центральным банком")

    def series(row_ids: list[int]) -> list[float | None]:
        out = []
        for c in range(1, len(hdr)):
            vals = [rows[r - 1][c] for r in row_ids]
            out.append(None if any(v is None or isinstance(v, str) for v in vals) else sum(vals) / 1000.0)
        return out

    idx = []
    for c in range(1, len(hdr)):
        d = hdr[c]
        if not isinstance(d, (datetime, date)):
            continue
        prev = pd.Period(d, "M") - 1                   # 1 января → конец декабря
        idx.append(prev)
    df = pd.DataFrame({
        "loans_corp": series([56, 60]), "loans_retail": series([65, 69]),
        "deposits_corp": series([76, 80, 85, 89]), "deposits_retail": series([77, 81, 86, 90]),
        "cb_debt_stock": series([71]),
    }, index=pd.PeriodIndex(idx, freq="M", name="date"))
    return df


def _month_label(text: str) -> pd.Period | None:
    m = re.match(r"\s*([А-Яа-я]+)\s+(\d{4})", str(text or ""))
    if not m or m.group(1).lower() not in MONTHS:
        return None
    return pd.Period(f"{m.group(2)}-{MONTHS[m.group(1).lower()]:02d}", "M")


def _num(x) -> float:
    try:
        return float(str(x).replace(",", ".").strip())
    except ValueError:
        return float("nan")


_LOOKALIKE = str.maketrans({"c": "с", "a": "а", "o": "о", "p": "р", "e": "е", "x": "х"})


def _find_sheet(wb, name: str):
    """Лист по имени с учётом латинских «двойников» кириллицы в названиях листов ЦБ."""
    norm = lambda t: t.lower().translate(_LOOKALIKE).replace(" ", "")      # noqa: E731
    for n in wb.sheetnames:
        if norm(n) == norm(name):
            return wb[n]
    raise RuntimeError(f"нет листа {name!r} среди {wb.sheetnames}")


def _sheet_table(path: Path, sheet: str):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = {}
    for row in _find_sheet(wb, sheet).iter_rows(values_only=True):
        p = _month_label(row[0])
        if p is not None:
            out[p] = list(row[1:])
    return out


def _weighted(rates: dict, shares: dict, cols: list[int]) -> pd.Series:
    vals = {}
    for p, r in rates.items():
        s = shares.get(p)
        if s is None:
            continue
        w = [_num(s[c - 1]) for c in cols]          # c — номер столбца Excel (A = 0)
        x = [_num(r[c - 1]) for c in cols]
        tot = sum(w)
        if tot > 0 and not any(pd.isna(v) for v in w + x):
            vals[p] = sum(a * b for a, b in zip(w, x)) / tot
    return pd.Series(vals).sort_index()


def parse_rates(tmp: Path) -> pd.DataFrame:
    """Средневзвешенные ставки по новым кредитам/вкладам в рублях, % годовых."""
    corp_l = _weighted(_sheet_table(tmp / "loans_nonfin.xlsx", "ставки_руб"),
                       _sheet_table(tmp / "loans_nonfin.xlsx", "cтруктура_руб"), [5, 8])
    hh_l_tab = _sheet_table(tmp / "loans_ind_new.xlsx", "ставки_руб")
    hh_l = pd.Series({p: _num(r[8]) for p, r in hh_l_tab.items()}).sort_index()   # столбец J: все виды кредитов
    dep_rates = _sheet_table(tmp / "deposits.xlsx", "ставки_руб")
    dep_shares = _sheet_table(tmp / "deposits.xlsx", "cтруктура_руб")
    hh_d = _weighted(dep_rates, dep_shares, [7, 11])                                # физлица: до 1 года / свыше
    corp_d = _weighted(dep_rates, dep_shares, [16, 19])                             # НФО
    df = pd.DataFrame({"loan_rate_corp": corp_l, "loan_rate_retail": hh_l,
                       "deposit_rate_corp": corp_d, "deposit_rate_retail": hh_d})
    df.index = pd.PeriodIndex(df.index, freq="M", name="date")
    return df


def parse_key_rate() -> pd.Series:
    html = fetch_text(f"{BASE}/hd_base/KeyRate/?UniDbQuery.Posted=True&UniDbQuery.From=01.09.2013"
                      f"&UniDbQuery.To={date.today():%d.%m.%Y}")
    rows = re.findall(r"<tr>\s*<td>(\d{2}\.\d{2}\.\d{4})</td>\s*<td>([\d,]+)</td>", html)
    if len(rows) < 100:
        raise RuntimeError("не удалось разобрать страницу ключевой ставки")
    s = pd.Series({datetime.strptime(d, "%d.%m.%Y"): _num(v) for d, v in rows}).sort_index()
    s = s.resample("D").ffill()                      # ставка действует и в дни без записи
    m = s.groupby(s.index.to_period("M")).mean()
    m.index.name = "date"
    return m


def parse_liquidity() -> pd.DataFrame:
    html = fetch_text(f"{BASE}/hd_base/bliquidity/?UniDbQuery.Posted=True&UniDbQuery.From=01.01.2014"
                      f"&UniDbQuery.To={date.today():%d.%m.%Y}")
    recs = {}
    for tr in re.findall(r"<tr>(.*?)</tr>", html, flags=re.S):
        cells = [re.sub(r"<[^>]+>", "", c).replace("&nbsp;", "").replace("\xa0", "").replace(" ", "").strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)]
        if not cells or not re.match(r"\d{2}\.\d{2}\.\d{4}$", cells[0]) or len(cells) < 15:
            continue
        v = [_num(c.replace("−", "-")) for c in cells[1:15]]        # столбцы 2…15 шапки ЦБ
        # проверка разметки по тождествам шапки: 4 = 5+6+7+8 и 9 = 10+11+12
        if abs(v[2] - sum(v[3:7])) > 0.6 or abs(v[7] - sum(v[8:11])) > 0.6:
            raise RuntimeError(f"разметка таблицы ликвидности изменилась: {cells[0]}")
        recs[datetime.strptime(cells[0], "%d.%m.%Y")] = {
            "liquidity_balance": v[0],          # дефицит (+) / профицит (−), млрд руб.
            "cb_obligations_to_banks": v[2],    # депозиты, КОБР и пр. (обязательства ЦБ перед КО)
            "cb_claims_on_banks": v[7],         # требования ЦБ к КО (рефинансирование)
        }
    df = pd.DataFrame.from_dict(recs, orient="index").sort_index()
    out = df.groupby(df.index.to_period("M")).mean()
    out.index.name = "date"
    return out


def main(out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="cbr_dl_"))
    try:
        for name, url in {
            "survey_odc.xlsx": "/vfs/statistics/credit_statistics/survey/survey_odc.xlsx",
            "loans_nonfin.xlsx": "/vfs/statistics/pdko/int_rat/loans_nonfin.xlsx",
            "loans_ind_new.xlsx": "/vfs/statistics/pdko/int_rat/loans_ind_new.xlsx",
            "deposits.xlsx": "/vfs/statistics/pdko/int_rat/deposits.xlsx",
        }.items():
            print("скачиваю", name)
            download(BASE + url, tmp / name)
        vol = parse_survey_odc(tmp / "survey_odc.xlsx")
        rates = parse_rates(tmp)
        key = parse_key_rate()
        liq = parse_liquidity()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    df = vol.join(rates, how="left").join(liq, how="left")
    df["key_rate"] = key.reindex(df.index)
    df["key_rate_source"] = df["key_rate"].map(lambda x: "cbr_keyrate" if pd.notna(x) else "")
    for a, b, v in REFI_RANGES:
        sel = (df.index >= pd.Period(a, "M")) & (df.index <= pd.Period(b, "M")) & df["key_rate"].isna()
        df.loc[sel, "key_rate"] = v
        df.loc[sel, "key_rate_source"] = "refi_manual"
    df = df.dropna(subset=["loans_corp", "deposits_corp"])
    df = df[df.index >= pd.Period("2010-01", "M")]
    out = df.copy()
    out.insert(0, "date", out.index.to_timestamp().strftime("%Y-%m-%d"))
    cols = ["date", "key_rate", "key_rate_source", "loans_corp", "loans_retail", "deposits_corp",
            "deposits_retail", "cb_debt_stock", "loan_rate_corp", "loan_rate_retail",
            "deposit_rate_corp", "deposit_rate_retail", "liquidity_balance",
            "cb_claims_on_banks", "cb_obligations_to_banks"]
    out[cols].round(4).to_csv(out_dir / "cbr_monthly.csv", index=False)
    k = out.loc[out["key_rate_source"] == "cbr_keyrate", ["date", "key_rate"]]
    k.round(4).to_csv(out_dir / "key_rate_monthly.csv", index=False)
    print(f"готово: {len(out)} месяцев ({out['date'].iloc[0]} … {out['date'].iloc[-1]}) → {out_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR)
