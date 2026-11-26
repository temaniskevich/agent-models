"""Консервативные даты доступности банковских ставок по страницам ЦБ."""

import csv
import html
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def fetch(period):
    url = f"https://www.cbr.ru/statistics/bank_sector/int_rat/{period.strftime('%m%y')}/"
    response = subprocess.run(["curl", "--silent", "--show-error", "--fail", "--location", "--retry", "1",
                               "--connect-timeout", "6", "--max-time", "20", url],
                              text=True, encoding="utf-8", capture_output=True, check=True)
    page = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", response.stdout)).split())
    match = re.search(r"(?:Дата (?:последнего )?обновления|Последнее обновление страницы)\s*:?\s*(\d{2}\.\d{2}\.\d{4})",
                      page, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"На странице нет даты: {url}")
    available = datetime.strptime(match[1], "%d.%m.%Y").date()
    if pd.Timestamp(available).to_period("M") <= period:
        raise ValueError(f"Дата архива не соответствует отчётному месяцу: {url}")
    return str(period), str(available), url


def main():
    dates = pd.period_range("2023-03", "2026-09", freq="M")
    rates = pd.read_csv(HERE.parent / "confirmation/cbr_release_dates.csv").set_index("period")
    mixed = pd.read_csv(HERE.parent / "pilnik_meeting/release_dates.csv").set_index("period")
    known = {period: (period, row.rates_available_by, row.rates_url) for period, row in mixed.iterrows()}
    known.update({period: (period, row.available_by, row.source_url) for period, row in rates.iterrows()})
    missing = [month for month in dates if str(month) not in known]
    with ThreadPoolExecutor(max_workers=5) as pool:
        requests = {month: pool.submit(fetch, month) for month in missing}
        for month, request in requests.items():
            try:
                known[str(month)] = request.result()
            except (ValueError, subprocess.CalledProcessError) as exc:
                print(f"{month}: {exc}", flush=True)
    results = [known[str(month)] for month in dates if str(month) in known]
    with (HERE / "release_dates.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["period", "available_by", "source_url"])
        writer.writerows(results)
    print(f"Проверены {len(results)} архивных страниц Банка России")


if __name__ == "__main__":
    main()
