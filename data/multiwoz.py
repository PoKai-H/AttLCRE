import json
import os
import random
import re

BASE_DIR = "/Users/reeseliu/Desktop/Linear attention/multiwoz/data/MultiWOZ_2.2"
SPLITS = ["train", "dev", "test"]
OUTPUT_BASE = "multiwoz_hotel_restaurant_train_json"

ALLOWED_DOMAINS = {"hotel", "restaurant", "train"}
random.seed(42)

os.makedirs(OUTPUT_BASE, exist_ok=True)

DOMAIN_TEMPLATES = {
    "train": {
        "signals": [
            {"type": "time_constraint", "value": "after_17_forbidden", "user": "I cannot take trains after 17:00.", "system": "Sure, I will avoid trains after 17:00."},
            {"type": "time_constraint", "value": "before_18_required", "user": "I need to arrive before 18:00.", "system": "Okay, I will only consider trains arriving before 18:00."},
            {"type": "price_constraint", "value": "cheap_only", "user": "Please find me the cheapest train option.", "system": "Got it, I will prioritize the cheapest train."},
            {"type": "departure_constraint", "value": "morning_only", "user": "I only want a morning train.", "system": "Understood, I will look for morning departures."},
            {"type": "departure_constraint", "value": "afternoon_only", "user": "I prefer an afternoon train.", "system": "Okay, I will search for afternoon trains."},
            {"type": "day_constraint", "value": "monday_only", "user": "I need the train on Monday.", "system": "Sure, I will search for Monday trains."},
            {"type": "destination_constraint", "value": "cambridge", "user": "I need a train going to Cambridge.", "system": "Alright, I will look for trains to Cambridge."},
            {"type": "departure_city_constraint", "value": "from_london", "user": "I will depart from London.", "system": "Okay, I will use London as the departure city."},
            {"type": "seat_preference", "value": "window_seat", "user": "If possible, I would like a window seat.", "system": "Sure, I will note your window seat preference."},
            {"type": "ticket_count", "value": "two_tickets", "user": "I need two train tickets.", "system": "Understood, I will search for two tickets."},
        ],
        "queries": [
            "Which train should I take?",
            "Which train fits my earlier request?",
            "What is the best train option for me?",
            "Which train matches my constraints?",
            "Can you tell me the right train to choose?",
            "Which departure should I book?",
            "What train would you recommend?",
            "Which one satisfies my request?",
            "Can you select the correct train for me?",
            "What is the suitable train choice?"
        ],
        "distractors": [
            {"type": "conflicting_preference", "value": "later_train_ok", "text_a": "Actually I might be okay with a later train.", "text_b": "Alright, I will keep that in mind."},
            {"type": "price_conflict", "value": "expensive_ok", "text_a": "Maybe price does not matter that much anymore.", "text_b": "Okay, budget is less strict now."},
            {"type": "schedule_conflict", "value": "evening_ok", "text_a": "An evening train could also be fine.", "text_b": "I see, evening trains might work too."},
            {"type": "destination_conflict", "value": "oxford_ok", "text_a": "Oxford could also work instead.", "text_b": "Okay, I will also consider Oxford."},
            {"type": "day_conflict", "value": "tuesday_ok", "text_a": "Tuesday may also be acceptable.", "text_b": "Got it, Tuesday can be considered too."},
            {"type": "ticket_conflict", "value": "one_ticket_ok", "text_a": "I might only need one ticket actually.", "text_b": "Sure, one ticket may also work."},
            {"type": "seat_conflict", "value": "any_seat_ok", "text_a": "Any seat is probably fine.", "text_b": "Okay, seat preference is flexible."},
            {"type": "time_conflict", "value": "after_17_ok", "text_a": "Maybe a train after 17:00 is not a big problem.", "text_b": "Understood, later trains are also possible."},
            {"type": "departure_conflict", "value": "from_cambridge_ok", "text_a": "I could depart from Cambridge instead.", "text_b": "Alright, Cambridge can also be a departure point."},
            {"type": "fare_conflict", "value": "first_class_ok", "text_a": "First class might be acceptable too.", "text_b": "Okay, I can include first-class options."},
        ],
        "candidate_sets": [
            ["You should take the 16:36 train.", "You should take the 17:20 train.", "You should book a restaurant first.", "There is not enough information."],
            ["The 08:15 train is the best choice.", "The 19:10 train is the best choice.", "You should change your hotel booking.", "There is not enough information."],
            ["You should book the cheapest morning train.", "You should choose the most expensive evening train.", "You should pick a hotel instead.", "There is not enough information."],
            ["Take the Monday departure.", "Take the Tuesday departure.", "Reserve a restaurant table first.", "There is not enough information."],
            ["The Cambridge train is the correct option.", "The Oxford train is the correct option.", "You should ask about parking.", "There is not enough information."],
            ["You should choose the earlier train.", "You should choose the later train.", "You should cancel your ticket.", "There is not enough information."],
            ["The train arriving before 18:00 is correct.", "The train arriving after 20:00 is correct.", "You should book a taxi instead.", "There is not enough information."],
            ["Choose the affordable ticket.", "Choose the premium ticket.", "You should reserve a museum ticket.", "There is not enough information."],
            ["You should take the London to Cambridge service.", "You should take the Cambridge to London service.", "You should focus on restaurant booking.", "There is not enough information."],
            ["The afternoon departure is better.", "The late-night departure is better.", "You should change the hotel area.", "There is not enough information."],
        ]
    },

    "hotel": {
        "signals": [
            {"type": "price_constraint", "value": "cheap_only", "user": "I need a cheap hotel.", "system": "Sure, I will only consider cheap hotels."},
            {"type": "area_constraint", "value": "centre", "user": "I want a hotel in the centre.", "system": "Okay, I will search in the centre."},
            {"type": "stars_constraint", "value": "four_star", "user": "I prefer a four-star hotel.", "system": "Understood, I will look for four-star hotels."},
            {"type": "parking_constraint", "value": "parking_required", "user": "The hotel must have parking.", "system": "Sure, I will only keep hotels with parking."},
            {"type": "internet_constraint", "value": "wifi_required", "user": "I need free Wi-Fi.", "system": "Okay, I will look for hotels with free Wi-Fi."},
            {"type": "stay_length", "value": "two_nights", "user": "I need the hotel for two nights.", "system": "Got it, I will search for a two-night stay."},
            {"type": "guest_count", "value": "two_people", "user": "It is for two people.", "system": "Understood, I will search for two guests."},
            {"type": "day_constraint", "value": "friday_checkin", "user": "I want to check in on Friday.", "system": "Alright, I will use Friday as the check-in day."},
            {"type": "hotel_type", "value": "guesthouse", "user": "I would prefer a guesthouse.", "system": "Sure, I will prioritize guesthouses."},
            {"type": "name_constraint", "value": "specific_name", "user": "I am interested in a place like the Acorn Guest House.", "system": "Okay, I will keep similar options in mind."},
        ],
        "queries": [
            "Which hotel should I book?",
            "What hotel matches my request?",
            "Which hotel is the right choice?",
            "Can you recommend the correct hotel?",
            "Which hotel fits my earlier constraints?",
            "What is the best hotel option?",
            "Which place should I reserve?",
            "Can you pick the suitable hotel?",
            "Which hotel would work for me?",
            "What hotel satisfies my needs?"
        ],
        "distractors": [
            {"type": "price_conflict", "value": "expensive_ok", "text_a": "Actually, a more expensive hotel might be okay.", "text_b": "Alright, budget is more flexible now."},
            {"type": "area_conflict", "value": "north_ok", "text_a": "The north area could also be fine.", "text_b": "Okay, I will also consider the north."},
            {"type": "stars_conflict", "value": "three_star_ok", "text_a": "A three-star place may also work.", "text_b": "Understood, three-star options are possible too."},
            {"type": "parking_conflict", "value": "parking_not_needed", "text_a": "Parking may not be necessary after all.", "text_b": "Sure, parking is optional then."},
            {"type": "wifi_conflict", "value": "wifi_optional", "text_a": "Wi-Fi is not a strict requirement now.", "text_b": "Okay, free Wi-Fi is optional."},
            {"type": "stay_conflict", "value": "one_night_ok", "text_a": "I may only stay one night.", "text_b": "Got it, one night may also work."},
            {"type": "guest_conflict", "value": "one_person_ok", "text_a": "It might just be for one person.", "text_b": "Understood, one guest is also possible."},
            {"type": "day_conflict", "value": "saturday_checkin_ok", "text_a": "Saturday check-in could also work.", "text_b": "Okay, Saturday is also possible."},
            {"type": "type_conflict", "value": "hotel_instead_guesthouse", "text_a": "A regular hotel is fine too.", "text_b": "Sure, I will include hotels as well."},
            {"type": "area_conflict", "value": "east_ok", "text_a": "The east side may also be acceptable.", "text_b": "Okay, I can consider the east area too."},
        ],
        "candidate_sets": [
            ["You should book the cheap hotel in the centre.", "You should choose the expensive hotel in the north.", "You should focus on a train instead.", "There is not enough information."],
            ["The four-star hotel with parking is the best choice.", "The three-star hotel without parking is the best choice.", "You should book a restaurant first.", "There is not enough information."],
            ["Choose the hotel with free Wi-Fi.", "Choose the hotel without internet.", "You should ask about train tickets.", "There is not enough information."],
            ["The Friday check-in option is correct.", "The Saturday check-in option is correct.", "You should change your dinner plan.", "There is not enough information."],
            ["You should reserve the guesthouse.", "You should reserve the luxury resort.", "You should book a taxi instead.", "There is not enough information."],
            ["The centre-area hotel is the right one.", "The far north hotel is the right one.", "You should look for a museum ticket.", "There is not enough information."],
            ["Pick the budget-friendly option.", "Pick the premium suite.", "You should switch to restaurant booking.", "There is not enough information."],
            ["The hotel for two people is correct.", "The single-room option is correct.", "You should ignore the hotel request.", "There is not enough information."],
            ["The parking-included hotel is suitable.", "The no-parking hotel is suitable.", "You should take a later train.", "There is not enough information."],
            ["The guesthouse in the centre is best.", "The hotel in the east is best.", "You should order food first.", "There is not enough information."],
        ]
    },

    "restaurant": {
        "signals": [
            {"type": "price_constraint", "value": "cheap_only", "user": "I want a cheap restaurant.", "system": "Sure, I will look for cheap restaurants."},
            {"type": "area_constraint", "value": "centre", "user": "I prefer a restaurant in the centre.", "system": "Okay, I will search in the centre."},
            {"type": "food_constraint", "value": "italian", "user": "I want Italian food.", "system": "Understood, I will look for Italian restaurants."},
            {"type": "booking_constraint", "value": "table_for_two", "user": "I need a table for two.", "system": "Sure, I will search for a table for two."},
            {"type": "day_constraint", "value": "friday", "user": "I need the reservation for Friday.", "system": "Alright, I will use Friday for the booking."},
            {"type": "time_constraint", "value": "19_00", "user": "I want the booking at 19:00.", "system": "Okay, I will search for 19:00 availability."},
            {"type": "food_constraint", "value": "chinese", "user": "I would like Chinese food.", "system": "Sure, I will look for Chinese restaurants."},
            {"type": "food_constraint", "value": "indian", "user": "I am looking for Indian food.", "system": "Understood, I will search for Indian restaurants."},
            {"type": "guest_count", "value": "four_people", "user": "The table should be for four people.", "system": "Got it, I will search for four guests."},
            {"type": "price_constraint", "value": "moderate", "user": "I want something moderately priced.", "system": "Okay, I will focus on moderate-price restaurants."},
        ],
        "queries": [
            "Which restaurant should I choose?",
            "What restaurant matches my request?",
            "Which restaurant is the best option?",
            "Can you recommend the right restaurant?",
            "Which place fits my earlier constraints?",
            "What restaurant should I book?",
            "Which restaurant works for me?",
            "Can you pick the suitable restaurant?",
            "Which one should I reserve?",
            "What is the correct restaurant choice?"
        ],
        "distractors": [
            {"type": "price_conflict", "value": "expensive_ok", "text_a": "Actually an expensive restaurant could be okay.", "text_b": "Alright, budget is not so strict now."},
            {"type": "area_conflict", "value": "north_ok", "text_a": "The north side might also be fine.", "text_b": "Okay, I will also consider the north area."},
            {"type": "food_conflict", "value": "british_ok", "text_a": "British food may also work.", "text_b": "Understood, British restaurants are possible too."},
            {"type": "time_conflict", "value": "20_30_ok", "text_a": "A later dinner around 20:30 is also fine.", "text_b": "Sure, later dinner times can be included."},
            {"type": "day_conflict", "value": "saturday_ok", "text_a": "Saturday may also work for the reservation.", "text_b": "Okay, Saturday is also possible."},
            {"type": "guest_conflict", "value": "two_people_ok", "text_a": "It might only be for two people.", "text_b": "Got it, a smaller table may also work."},
            {"type": "food_conflict", "value": "french_ok", "text_a": "French cuisine would also be acceptable.", "text_b": "Alright, French restaurants can also be considered."},
            {"type": "price_conflict", "value": "moderate_ok", "text_a": "Moderate pricing may be fine too.", "text_b": "Okay, I will also consider moderate options."},
            {"type": "area_conflict", "value": "east_ok", "text_a": "The east part of town could work too.", "text_b": "Understood, east-side restaurants are possible."},
            {"type": "booking_conflict", "value": "no_booking_needed", "text_a": "A reservation might not even be necessary.", "text_b": "Sure, walk-in options are also possible."},
        ],
        "candidate_sets": [
            ["You should choose the cheap restaurant in the centre.", "You should choose the expensive restaurant in the north.", "You should book a hotel instead.", "There is not enough information."],
            ["The Italian restaurant is the correct choice.", "The British restaurant is the correct choice.", "You should change your train ticket.", "There is not enough information."],
            ["Book the table for two at 19:00.", "Book the table for four at 20:30.", "You should reserve a hotel room.", "There is not enough information."],
            ["The Friday reservation is the right option.", "The Saturday reservation is the right option.", "You should focus on transportation.", "There is not enough information."],
            ["Choose the moderate-price restaurant.", "Choose the luxury fine-dining restaurant.", "You should skip the meal plan.", "There is not enough information."],
            ["The Chinese restaurant is best.", "The French restaurant is best.", "You should take an earlier train.", "There is not enough information."],
            ["The restaurant in the centre is suitable.", "The restaurant in the east is suitable.", "You should choose a guesthouse instead.", "There is not enough information."],
            ["Reserve the Indian restaurant.", "Reserve the British restaurant.", "You should ask about hotel parking.", "There is not enough information."],
            ["The 19:00 booking fits your request.", "The 20:30 booking fits your request.", "You should ignore the restaurant plan.", "There is not enough information."],
            ["The cheap dining option is correct.", "The expensive dining option is correct.", "You should arrange a taxi instead.", "There is not enough information."],
        ]
    }
}

def normalize_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_domain_from_dialog(dialog):
    found = set()

    services = dialog.get("services", [])
    for s in services:
        s = str(s).lower()
        if s in ALLOWED_DOMAINS:
            found.add(s)

    domains = dialog.get("domains", [])
    for d in domains:
        d = str(d).lower()
        if d in ALLOWED_DOMAINS:
            found.add(d)

    frames = dialog.get("frames", [])
    for fr in frames:
        service = str(fr.get("service", "")).lower()
        if service in ALLOWED_DOMAINS:
            found.add(service)

    dialogue_id = str(dialog.get("dialogue_id", "")).lower()
    for d in ALLOWED_DOMAINS:
        if d in dialogue_id:
            found.add(d)
    if not found:
        return None
    return random.choice(list(found))

def extract_turns(dialog):
    results = []
    turns = dialog.get("turns", [])
    for turn in turns:
        speaker = str(turn.get("speaker", "")).strip().upper()
        text = normalize_text(turn.get("utterance", ""))

        if not text:
            continue

        if speaker == "USER":
            role = "A"
        elif speaker == "SYSTEM":
            role = "B"
        else:
            continue

        results.append({"speaker": role, "text": text})

    return results

def sample_noise_turns(original_turns, min_n=2, max_n=4):
    if not original_turns:
        return []

    n = min(len(original_turns), random.randint(min_n, max_n))
    chosen = random.sample(original_turns, n)
    return [{"speaker": t["speaker"], "text": t["text"], "role": "noise"} for t in chosen]

def sample_distractor_turns(domain, min_pairs=3, max_pairs=4):
    pool = DOMAIN_TEMPLATES[domain]["distractors"]
    chosen = random.sample(pool, random.randint(min_pairs, max_pairs))

    turns = []
    chosen_meta = []

    for d in chosen:
        turns.append({"speaker": "A", "text": d["text_a"], "role": "distractor"})
        turns.append({"speaker": "B", "text": d["text_b"], "role": "distractor"})
        chosen_meta.append(d)

    target_len = random.randint(5, 7)
    turns = turns[:target_len]

    main_d = chosen_meta[0] if chosen_meta else None
    return turns, main_d

def build_candidates(domain, signal_obj):
    full_set = random.choice(DOMAIN_TEMPLATES[domain]["candidate_sets"])
    k = random.randint(3, 4)
    candidates = full_set[:k]
    correct_index = 0
    return candidates, correct_index

def build_query(domain):
    return random.choice(DOMAIN_TEMPLATES[domain]["queries"])

def build_signal(domain):
    return random.choice(DOMAIN_TEMPLATES[domain]["signals"])

def estimate_difficulty(num_noise_turns, signal_distance, has_distractor):
    score = 0
    if num_noise_turns >= 3:
        score += 1
    if signal_distance >= 7:
        score += 1
    if has_distractor:
        score += 1

    if score <= 1:
        return "easy"
    elif score == 2:
        return "medium"
    else:
        return "hard"

def insert_signal_early(dialogue, signal_turns):
    if not dialogue:
        return signal_turns[:]

    max_pos = min(5, len(dialogue))
    insert_pos = random.randint(0, max_pos - 1) if max_pos > 0 else 0

    new_dialogue = dialogue[:]
    for i, t in enumerate(signal_turns):
        new_dialogue.insert(insert_pos + i, t)

    return new_dialogue, insert_pos

def count_tokens_text(text):
    if not text:
        return 0
    return len(text.split())

def count_dialogue_tokens(dialogue):
    total = 0
    for turn in dialogue:
        total += count_tokens_text(turn.get("text", ""))
    return total

def create_augmented_example(dialog, sample_id):
    domain = get_domain_from_dialog(dialog)
    if domain not in ALLOWED_DOMAINS:
        return None

    original_turns = extract_turns(dialog)
    if len(original_turns) < 2:
        return None

    signal_obj = build_signal(domain)
    query_text = build_query(domain)
    distractor_turns, main_distractor = sample_distractor_turns(domain)
    noise_turns = sample_noise_turns(original_turns, 2, 4)
    candidates, correct_index = build_candidates(domain, signal_obj)

    signal_turns = [
        {"speaker": "A", "text": signal_obj["user"], "role": "signal"},
        {"speaker": "B", "text": signal_obj["system"], "role": "signal"},
    ]

    query_context = None
    for t in reversed(original_turns):
        if t["speaker"] == "B":
            query_context = {
                "speaker": "B",
                "text": t["text"],
                "role": "query_context"
            }
            break

    if query_context is None:
        query_context = {
            "speaker": "B",
            "text": "Here are several options available.",
            "role": "query_context"
        }

    query_turn = {
        "speaker": "A",
        "text": query_text,
        "role": "query"
    }

    middle_dialogue = []
    middle_dialogue.extend(noise_turns)
    middle_dialogue.extend(distractor_turns)

    base_dialogue = middle_dialogue[:]
    if len(base_dialogue) == 0:
        base_dialogue = noise_turns + distractor_turns

    full_dialogue, signal_pos = insert_signal_early(base_dialogue, signal_turns)

    full_dialogue.append(query_context)
    full_dialogue.append(query_turn)

    distractor_pos = None
    for idx, turn in enumerate(full_dialogue):
        if turn["role"] == "distractor":
            distractor_pos = idx
            break

    signal_distance = len(full_dialogue) - 1 - signal_pos
    num_noise_turns = sum(1 for x in full_dialogue if x["role"] == "noise")
    has_distractor = any(x["role"] == "distractor" for x in full_dialogue)
    dialogue_token_count = count_dialogue_tokens(full_dialogue)

    difficulty = estimate_difficulty(
        num_noise_turns=num_noise_turns,
        signal_distance=signal_distance,
        has_distractor=has_distractor
    )

    result = {
        "sample_id": sample_id,
        "dialogue": full_dialogue,
        "signal": {
            "type": signal_obj["type"],
            "value": signal_obj["value"],
            "position": signal_pos
        },
        "distractor": {
            "type": main_distractor["type"] if main_distractor else "none",
            "value": main_distractor["value"] if main_distractor else "none",
            "position": distractor_pos if distractor_pos is not None else -1
        },
        "query": query_text,
        "candidates": candidates,
        "correct_index": correct_index,
        "metadata": {
            "num_turns": len(full_dialogue),
            "signal_distance": signal_distance,
            "num_noise_turns": num_noise_turns,
            "has_distractor": has_distractor,
            "difficulty": difficulty,
            "dialogue_token_count": dialogue_token_count
        }
    }

    return result

def load_dialogues_from_split(split_dir):
    all_dialogues = []

    if not os.path.exists(split_dir):
        print(f"[Warning] Split directory not found: {split_dir}")
        return all_dialogues

    for filename in os.listdir(split_dir):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(split_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Skip] Failed to read {path}: {e}")
            continue

        if isinstance(data, dict):
            items = list(data.values())
        elif isinstance(data, list):
            items = data
        else:
            continue

        for x in items:
            if isinstance(x, dict):
                all_dialogues.append(x)

    return all_dialogues

global_id = 1

for split in SPLITS:
    split_dir = os.path.join(BASE_DIR, split)
    output_dir = os.path.join(OUTPUT_BASE, split)
    os.makedirs(output_dir, exist_ok=True)

    dialogues = load_dialogues_from_split(split_dir)

    augmented = []

    for dialog in dialogues:
        sample_id = f"ex_{global_id:04d}"
        item = create_augmented_example(dialog, sample_id)
        if item is None:
            continue

        augmented.append(item)
        global_id += 1

    output_path = os.path.join(output_dir, f"{split}_augmented.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)