#💰 PART 5: Price-to-Value Index
def value_index(total_score, price):
    if not price:
        return 0
    return round(total_score / price * 100, 2)

#📊 PART 6: Ranking Pipeline
def rank_phones(phone_records):
    for phone in phone_records:
        phone["value_index"] = value_index(phone["total_score"], phone["price_usd"])
    return sorted(phone_records, key=lambda x: x["total_score"], reverse=True)
