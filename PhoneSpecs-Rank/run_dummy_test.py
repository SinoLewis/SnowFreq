# tests/run_dummy_test.py

from tests.dummy_data import DUMMY_PHONES
from scoring.categories import score_performance, score_display, score_battery
from scoring.composite import total_score
from scoring.persona import persona_score
from scoring.value import value_index

def run_test():
    results = []

    for phone in DUMMY_PHONES:
        scores = {}

        scores["performance"] = score_performance(
            phone["cpu_score"],
            phone["gpu_score"],
            phone["ram_gb"]
        )

        scores["display"] = score_display(
            phone["display_type"],
            phone["refresh_rate"],
            phone["resolution_ppi"]
        )

        scores["battery"] = score_battery(
            phone["battery_mah"],
            phone["charging_watt"]
        )

        # Simplified placeholders for now
        scores["camera"] = phone["main_camera_mp"]
        scores["build"] = phone["os_update_years"] * 10
        scores["connectivity"] = 80

        phone["total_score"] = total_score(scores)
        phone["gamer_score"] = persona_score(scores, "gamer")
        phone["value_index"] = value_index(phone["total_score"], phone["price_usd"])

        results.append(phone)

    results.sort(key=lambda x: x["total_score"], reverse=True)

    return results


if __name__ == "__main__":
    ranked = run_test()
    for r in ranked:
        print(f"{r['brand']} {r['model']} → Score: {r['total_score']} | Value: {r['value_index']}")
