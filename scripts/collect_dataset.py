"""
Dataset Collection Script for Auto-Maintenance Assistant
This script helps collect and structure car maintenance Q&A pairs from various sources.
"""

import json
import csv
from typing import List, Dict
from pathlib import Path

from _project_root import PROJECT_ROOT


class AutoMaintenanceDatasetCollector:
    """Collects car maintenance Q&A pairs from various sources."""

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = []

    def add_qa_pair(self, question: str, answer: str, category: str = None,
                    source: str = None, difficulty: str = "beginner"):
        """Add a Q&A pair to the dataset."""
        self.dataset.append({
            "instruction": question,
            "response": answer,
            "category": category or "general",
            "source": source or "manual",
            "difficulty": difficulty
        })

    def save_json(self, filename: str = "auto_maintenance_dataset.json"):
        """Save dataset as JSON."""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.dataset)} Q&A pairs to {filepath}")
        return filepath

    def save_csv(self, filename: str = "auto_maintenance_dataset.csv"):
        """Save dataset as CSV."""
        filepath = self.output_dir / filename
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["instruction", "response", "category", "source", "difficulty"])
            writer.writeheader()
            writer.writerows(self.dataset)
        print(f"Saved {len(self.dataset)} Q&A pairs to {filepath}")
        return filepath

    def load_from_json(self, filepath: str):
        """Load existing dataset from JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        print(f"Loaded {len(self.dataset)} Q&A pairs from {filepath}")

    def generate_basic_maintenance_qa(self):
        """Generate basic car maintenance Q&A pairs."""

        basic_qa = [
            {
                "instruction": "How often should I change my engine oil?",
                "response": "Most vehicles require an oil change every 5,000 to 7,500 miles, or every 6 months, whichever comes first. However, you should always check your vehicle's owner's manual for the manufacturer's specific recommendations. Some newer vehicles with synthetic oil can go up to 10,000 miles between changes. Regular oil changes are crucial for engine health and longevity.",
                "category": "oil_change",
                "difficulty": "beginner"
            },
            {
                "instruction": "What does the check engine light mean?",
                "response": "The check engine light (CEL) indicates that your vehicle's onboard diagnostic system has detected a problem. It could be something minor like a loose gas cap or something more serious like an engine misfire. You should have your vehicle diagnosed by a mechanic or use an OBD-II scanner to read the error codes. Don't ignore it - even minor issues can lead to expensive repairs if left unchecked.",
                "category": "diagnostics",
                "difficulty": "beginner"
            },
            {
                "instruction": "How do I check my tire pressure?",
                "response": "To check tire pressure: 1) Use a tire pressure gauge (digital or analog), 2) Remove the valve cap from your tire, 3) Press the gauge firmly onto the valve stem, 4) Read the pressure reading, 5) Compare it to the recommended PSI found in your owner's manual or on the driver's side door jamb sticker. Check when tires are cold (before driving) for accurate readings. Most passenger cars require 30-35 PSI.",
                "category": "tires",
                "difficulty": "beginner"
            },
            {
                "instruction": "When should I replace my car battery?",
                "response": "Car batteries typically last 3-5 years. Signs you need a replacement include: slow engine cranking, dimming headlights, battery warning light, corrosion on terminals, or if your battery is over 3 years old and you're experiencing starting issues. Extreme temperatures can shorten battery life. Have your battery tested annually after it's 3 years old.",
                "category": "electrical",
                "difficulty": "beginner"
            },
            {
                "instruction": "What is the difference between synthetic and conventional oil?",
                "response": "Synthetic oil is chemically engineered to provide better protection and performance than conventional oil. It flows better in cold temperatures, resists breakdown at high temperatures, and can last longer between changes. Conventional oil is derived directly from crude oil. Synthetic oil is more expensive but offers superior engine protection, especially in extreme conditions. Many modern vehicles require synthetic oil.",
                "category": "oil_change",
                "difficulty": "intermediate"
            },
            {
                "instruction": "How often should I rotate my tires?",
                "response": "Tires should be rotated every 5,000 to 8,000 miles, or according to your vehicle's maintenance schedule. Tire rotation helps ensure even wear across all tires, extending their lifespan and improving handling. The rotation pattern depends on whether your vehicle is front-wheel drive, rear-wheel drive, or all-wheel drive. Some vehicles have directional tires that can only rotate front-to-back.",
                "category": "tires",
                "difficulty": "beginner"
            },
            {
                "instruction": "What should I do if my car overheats?",
                "response": "If your car overheats: 1) Turn off the AC and turn on the heater to help dissipate heat, 2) Pull over safely as soon as possible, 3) Turn off the engine and let it cool for at least 15-20 minutes, 4) Check coolant level (only when engine is cool - never open a hot radiator), 5) If coolant is low, add coolant/water mix, 6) If problem persists, call for roadside assistance. Never continue driving an overheating vehicle as it can cause severe engine damage.",
                "category": "cooling_system",
                "difficulty": "intermediate"
            },
            {
                "instruction": "How do I know if my brake pads need replacement?",
                "response": "Signs of worn brake pads include: squealing or grinding noises when braking, vibration in steering wheel or brake pedal, longer stopping distances, brake pad warning light, or visible wear (pads less than 1/4 inch thick). Most brake pads last 30,000-70,000 miles depending on driving habits. Have your brakes inspected if you notice any of these signs - brake safety is critical.",
                "category": "brakes",
                "difficulty": "beginner"
            },
            {
                "instruction": "What does it mean if my car is making a clicking sound when starting?",
                "response": "A clicking sound when starting usually indicates a dead or weak battery. The clicking is the starter solenoid trying to engage but not having enough power. Other possible causes include: corroded battery terminals, loose battery connections, or a faulty starter motor. First, check battery voltage and connections. If battery is good, the starter motor may need replacement.",
                "category": "electrical",
                "difficulty": "intermediate"
            },
            {
                "instruction": "How often should I replace my air filter?",
                "response": "The air filter should typically be replaced every 12,000 to 15,000 miles, or once a year. However, if you drive in dusty conditions, you may need to replace it more frequently. A dirty air filter reduces fuel efficiency and engine performance. You can check it visually - if it looks dirty or clogged, replace it. Air filters are inexpensive and easy to replace yourself.",
                "category": "engine",
                "difficulty": "beginner"
            },
            {
                "instruction": "What is a timing belt and when should it be replaced?",
                "response": "A timing belt synchronizes the rotation of the crankshaft and camshaft, ensuring valves open and close at the correct times. Most vehicles require timing belt replacement every 60,000 to 100,000 miles (check your owner's manual). If a timing belt breaks, it can cause catastrophic engine damage. Some vehicles use timing chains which typically last longer. This is a critical maintenance item that shouldn't be ignored.",
                "category": "engine",
                "difficulty": "advanced"
            },
            {
                "instruction": "Why is my car shaking when I brake?",
                "response": "Shaking when braking usually indicates warped brake rotors. This can happen from excessive heat buildup, uneven tightening of lug nuts, or normal wear. Other causes include: worn brake pads, loose brake components, or wheel alignment issues. Have your brakes inspected - warped rotors may need resurfacing or replacement. This is a safety issue and should be addressed promptly.",
                "category": "brakes",
                "difficulty": "intermediate"
            },
            {
                "instruction": "How do I check my transmission fluid?",
                "response": "To check transmission fluid: 1) Park on level ground with engine running and transmission in Park, 2) Locate the transmission dipstick (usually red or yellow handle, different from oil dipstick), 3) Pull it out, wipe clean, reinsert fully, then pull out again, 4) Check the level against the markings - should be between 'Full' and 'Add' marks, 5) Check color - should be red/pink, not brown or burnt-smelling. If low or discolored, have it serviced.",
                "category": "transmission",
                "difficulty": "intermediate"
            },
            {
                "instruction": "What should I do if my car won't start?",
                "response": "If your car won't start, check these common causes: 1) Dead battery (try jump-starting), 2) Empty gas tank, 3) Faulty starter (clicking sound), 4) Ignition switch issues, 5) Fuel pump problems (no sound when turning key), 6) Security system preventing start. Start with the simplest checks first. If battery is good and you hear the engine trying to turn over, it's likely a fuel or ignition issue requiring professional diagnosis.",
                "category": "diagnostics",
                "difficulty": "beginner"
            },
            {
                "instruction": "How often should I get my car's alignment checked?",
                "response": "Get your alignment checked if you notice: uneven tire wear, vehicle pulling to one side, steering wheel off-center, or after hitting a pothole or curb. Most manufacturers recommend checking alignment every 2-3 years or 20,000-30,000 miles. Proper alignment improves tire life, fuel efficiency, and handling. It's relatively inexpensive and quick to fix.",
                "category": "suspension",
                "difficulty": "beginner"
            }
        ]

        for qa in basic_qa:
            self.add_qa_pair(
                question=qa["instruction"],
                answer=qa["response"],
                category=qa["category"],
                difficulty=qa["difficulty"],
                source="generated_basic"
            )

    def generate_advanced_maintenance_qa(self):
        """Generate more advanced car maintenance Q&A pairs."""

        advanced_qa = [
            {
                "instruction": "What is the difference between AWD and 4WD?",
                "response": "AWD (All-Wheel Drive) automatically distributes power to all wheels and is always engaged, providing better traction in various conditions. 4WD (Four-Wheel Drive) can be manually engaged/disengaged and is designed for off-road use and low-speed traction. AWD is better for on-road performance and handling, while 4WD excels in off-road situations. AWD systems are more complex and can be more expensive to maintain.",
                "category": "drivetrain",
                "difficulty": "intermediate"
            },
            {
                "instruction": "How do I diagnose a P0420 error code?",
                "response": "P0420 indicates 'Catalyst System Efficiency Below Threshold' - meaning your catalytic converter isn't working efficiently. Common causes: failing oxygen sensors, exhaust leaks, damaged catalytic converter, or engine misfires causing unburned fuel. Start by checking for exhaust leaks, then test oxygen sensors. If sensors are good, the catalytic converter likely needs replacement. This code won't cause immediate problems but will fail emissions tests.",
                "category": "diagnostics",
                "difficulty": "advanced"
            },
            {
                "instruction": "What causes engine knocking and how do I fix it?",
                "response": "Engine knocking (pinging) is caused by premature fuel ignition in the cylinder. Common causes: using lower octane fuel than required, carbon buildup, incorrect ignition timing, or engine overheating. Solutions: use higher octane fuel, clean carbon deposits, check timing, ensure proper cooling. Persistent knocking can damage pistons and should be addressed immediately. Modern engines have knock sensors that adjust timing automatically.",
                "category": "engine",
                "difficulty": "advanced"
            },
            {
                "instruction": "How do I flush my cooling system?",
                "response": "To flush cooling system: 1) Let engine cool completely, 2) Drain old coolant from radiator drain plug, 3) Remove thermostat housing and flush with water, 4) Use cooling system flush solution per instructions, 5) Rinse thoroughly with distilled water, 6) Refill with proper coolant/water mix (usually 50/50), 7) Bleed air from system. Use only the coolant type specified in your owner's manual. Flush every 2-3 years or 30,000 miles.",
                "category": "cooling_system",
                "difficulty": "advanced"
            },
            {
                "instruction": "What is a CVT transmission and how do I maintain it?",
                "response": "CVT (Continuously Variable Transmission) uses belts and pulleys instead of gears for seamless acceleration. Maintenance: change CVT fluid every 30,000-60,000 miles (check manual), avoid aggressive towing, don't 'neutral drop', and have it serviced by qualified technicians. CVT fluid is specific - never use regular ATF. Signs of problems: slipping, jerking, or whining noises. CVTs require specialized service.",
                "category": "transmission",
                "difficulty": "advanced"
            }
        ]

        for qa in advanced_qa:
            self.add_qa_pair(
                question=qa["instruction"],
                answer=qa["response"],
                category=qa["category"],
                difficulty=qa["difficulty"],
                source="generated_advanced"
            )


def main():
    """Main function to generate and save dataset."""
    collector = AutoMaintenanceDatasetCollector()

    print("Generating basic maintenance Q&A pairs...")
    collector.generate_basic_maintenance_qa()

    print("Generating advanced maintenance Q&A pairs...")
    collector.generate_advanced_maintenance_qa()

    print(f"\nTotal Q&A pairs collected: {len(collector.dataset)}")

    # Save in multiple formats
    collector.save_json("auto_maintenance_dataset.json")
    collector.save_csv("auto_maintenance_dataset.csv")

    print("\nDataset collection complete!")
    print("\nNext step: python scripts/expand_dataset.py")


if __name__ == "__main__":
    main()
