"""
Expand the dataset with more car maintenance Q&A pairs.
"""

import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))

from _project_root import PROJECT_ROOT
from collect_dataset import AutoMaintenanceDatasetCollector


def scrape_manual_qa_pairs():
    """Manually curated Q&A pairs. Add more from FAQs/manuals as needed."""
    manual_qa = [
        {
            "instruction": "How do I jump start a car?",
            "response": "To jump start: 1) Park both cars close but not touching, both off, 2) Connect red clamp to dead battery's positive terminal, 3) Connect other red clamp to good battery's positive terminal, 4) Connect black clamp to good battery's negative terminal, 5) Connect last black clamp to unpainted metal on dead car (not battery), 6) Start good car, wait 2-3 minutes, 7) Start dead car, 8) Remove clamps in reverse order. Never connect negative to negative on dead battery - can cause explosion.",
            "category": "electrical",
            "difficulty": "beginner"
        },
        {
            "instruction": "What does it mean when my car smells like burning rubber?",
            "response": "Burning rubber smell can indicate: slipping serpentine belt, overheating brakes, electrical wire insulation melting, or oil leak on hot engine parts. Check belt condition and tension first. If smell persists, have vehicle inspected immediately as electrical issues can be fire hazards.",
            "category": "diagnostics",
            "difficulty": "intermediate"
        },
        {
            "instruction": "How do I change a flat tire?",
            "response": "To change flat tire: 1) Park on level, safe ground, engage parking brake, 2) Loosen lug nuts slightly before jacking (they're easier when wheel is on ground), 3) Place jack at proper lift point (check manual), lift until tire clears ground, 4) Remove lug nuts completely, 5) Remove flat tire, 6) Mount spare tire, 7) Hand-tighten lug nuts in star pattern, 8) Lower vehicle, fully tighten lug nuts, 9) Check pressure on spare. Don't drive far on donut spares - they're temporary.",
            "category": "tires",
            "difficulty": "beginner"
        },
        {
            "instruction": "What is the difference between DOT 3 and DOT 4 brake fluid?",
            "response": "DOT 3 and DOT 4 are brake fluid specifications. DOT 4 has higher boiling point (446°F vs 401°F), better performance in high-heat situations, and absorbs moisture slower. However, DOT 4 is more expensive and can damage paint. Always use the type specified in your owner's manual - mixing types or using wrong type can cause brake failure. Brake fluid should be changed every 2-3 years.",
            "category": "brakes",
            "difficulty": "intermediate"
        },
        {
            "instruction": "Why is my car's AC not blowing cold air?",
            "response": "AC issues can be caused by: low refrigerant (most common), faulty compressor, clogged condenser, electrical problems, or blend door actuator issues. First check if AC clutch engages when AC is on. If not, likely electrical or compressor issue. If clutch engages but no cold air, likely low refrigerant or leak. AC systems require professional service - refrigerant handling requires certification.",
            "category": "hvac",
            "difficulty": "intermediate"
        },
        {
            "instruction": "How do I know if my spark plugs need replacement?",
            "response": "Signs of bad spark plugs: rough idle, misfiring, poor acceleration, increased fuel consumption, difficulty starting. Most vehicles need spark plug replacement every 30,000-100,000 miles depending on type (copper vs platinum vs iridium). Check your owner's manual. Bad spark plugs can damage ignition coils and reduce fuel efficiency significantly.",
            "category": "engine",
            "difficulty": "intermediate"
        },
        {
            "instruction": "What should I do if I put the wrong fuel in my car?",
            "response": "If you put diesel in a gas car (or vice versa): DO NOT start the engine. Have vehicle towed to mechanic to drain fuel system. If you put regular gas in a car requiring premium: it's usually okay for occasional use, but may cause knocking. Fill with premium on next tank. If you put E85 in non-flex-fuel vehicle: don't start engine, have fuel system drained. Wrong fuel can cause expensive engine damage.",
            "category": "fuel_system",
            "difficulty": "beginner"
        },
        {
            "instruction": "How often should I replace my cabin air filter?",
            "response": "Cabin air filter should be replaced every 15,000 to 30,000 miles, or annually. If you drive in dusty/pollen-heavy areas, replace more frequently. Signs it needs replacement: reduced airflow from vents, musty odors, or increased allergens. Cabin filters are usually behind glove box and easy to replace yourself. They improve air quality and AC efficiency.",
            "category": "hvac",
            "difficulty": "beginner"
        },
        {
            "instruction": "What causes white smoke from exhaust?",
            "response": "White smoke (actually steam) from exhaust usually indicates coolant leaking into combustion chamber - often from blown head gasket, cracked cylinder head, or damaged engine block. This is serious and requires immediate attention. Other causes: condensation in exhaust (normal on cold starts), or transmission fluid in intake (automatic transmission issue). Thick white smoke that doesn't dissipate quickly is a problem.",
            "category": "engine",
            "difficulty": "advanced"
        },
        {
            "instruction": "How do I maintain my car's paint and prevent rust?",
            "response": "To maintain paint and prevent rust: wash regularly (especially in winter to remove salt), wax every 3-6 months, fix paint chips immediately, keep car garaged when possible, use rust inhibitor in rust-prone areas, check for rust spots regularly, and have professional detailing annually. Rust spreads quickly once it starts. Touch-up paint pens can fix small chips before they become problems.",
            "category": "body",
            "difficulty": "beginner"
        },
        {
            "instruction": "What is a serpentine belt and when should it be replaced?",
            "response": "Serpentine belt drives multiple engine accessories (alternator, power steering, AC compressor, water pump). Signs of wear: squealing noise, visible cracks/fraying, glazing, or accessories not working. Most belts last 60,000-100,000 miles. If belt breaks, you'll lose power steering, battery charging, and cooling - don't drive. Replacement is relatively inexpensive and critical for reliability.",
            "category": "engine",
            "difficulty": "intermediate"
        },
        {
            "instruction": "How do I check if my alternator is working?",
            "response": "Signs of failing alternator: dimming headlights, battery warning light, dead battery (especially if recently replaced), electrical issues, or growling/whining noise. Test with multimeter: with engine running, battery voltage should be 13.5-14.5V. If lower, alternator likely failing. If alternator fails, battery will drain and car will eventually stall. Have it tested at auto parts store if unsure.",
            "category": "electrical",
            "difficulty": "intermediate"
        },
        {
            "instruction": "What maintenance does a hybrid car need?",
            "response": "Hybrids need standard maintenance (oil changes, brakes, tires) plus: high-voltage battery system checks, inverter coolant service, and brake system service (regenerative braking reduces brake wear but system needs periodic service). Hybrid batteries typically last 8-15 years. Some hybrids have special transmission fluid requirements. Always follow manufacturer's maintenance schedule - hybrids have unique systems.",
            "category": "hybrid",
            "difficulty": "intermediate"
        },
        {
            "instruction": "How do I know if my car needs new shocks or struts?",
            "response": "Signs of worn shocks/struts: excessive bouncing after bumps, nose-diving when braking, body roll in turns, uneven tire wear, leaking fluid on shocks, or vehicle sits lower than normal. Worn suspension affects handling and safety. Most shocks/struts last 50,000-100,000 miles. Have suspension inspected if you notice these signs - it's a safety issue.",
            "category": "suspension",
            "difficulty": "intermediate"
        },
        {
            "instruction": "What should I check before a long road trip?",
            "response": "Before long trip: check tire pressure and tread depth, inspect tires for damage, check all fluid levels (oil, coolant, brake, transmission, power steering), test battery, check lights (headlights, brake lights, turn signals), inspect belts and hoses, ensure spare tire is inflated, pack emergency kit (jumper cables, tools, first aid), and have recent oil change if due. Consider professional inspection if trip is very long.",
            "category": "general",
            "difficulty": "beginner"
        }
    ]
    return manual_qa


def expand_dataset_from_manual():
    """Expand dataset with manually curated Q&A pairs."""
    data_dir = PROJECT_ROOT / "data"
    existing_file = data_dir / "auto_maintenance_dataset.json"

    collector = AutoMaintenanceDatasetCollector(output_dir=str(data_dir))
    if existing_file.exists():
        collector.load_from_json(str(existing_file))
        print(f"Loaded {len(collector.dataset)} existing Q&A pairs")

    for qa in scrape_manual_qa_pairs():
        collector.add_qa_pair(
            question=qa["instruction"],
            answer=qa["response"],
            category=qa["category"],
            difficulty=qa["difficulty"],
            source="manual_curation"
        )

    print(f"\nTotal Q&A pairs: {len(collector.dataset)}")
    collector.save_json("auto_maintenance_dataset.json")
    collector.save_csv("auto_maintenance_dataset.csv")
    return collector


if __name__ == "__main__":
    print("Expanding dataset with manual Q&A pairs...")
    expand_dataset_from_manual()
    print("Dataset expansion complete!")
    print("Next step: python scripts/prepare_training_data.py")
