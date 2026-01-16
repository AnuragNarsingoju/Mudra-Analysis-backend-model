"""
Mudra Knowledge Base
Comprehensive meanings for Bharatanatyam mudras.
"""

# Single-Hand Mudras (Asamyuktha Hasta)
MUDRA_MEANINGS = {
    # Asamyuktha Hastas (Single Hand Mudras)
    "Pataka": "Flag - represents cloud, forest, ocean, denial, or a sword",
    "Tripataka": "Three parts of flag - represents a crown, tree, thunderbolt, or flame",
    "Ardhapataka": "Half flag - represents a leaf, peg, or knife",
    "Kartarimukha": "Scissors face - represents lightning, separation, or the beak of a bird",
    "Mayura": "Peacock - represents a peacock, applying tilak, or holding flowers",
    "Ardhachandra": "Half moon - represents the moon, meditation, or the phases of the moon",
    "Arala": "Bent - represents drinking poison, tasting nectar, or determination",
    "Shukatundaka": "Parrot's beak - represents shooting an arrow, whispering, or lifting,",
    "Mushti": "Fist - represents firmness, steadiness, or holding something",
    "Shikhara": "Peak - represents Lord Shiva, lingam, or questioning",
    "Kapitta": "Elephant apple - represents holding Lakshmi, goddess Saraswati",
    "Katakamukha": "Opening in a bracelet - represents picking flowers, holding a necklace",
    "Suchi": "Needle - represents one, unity, or pointing",
    "Chandrakala": "Digit of moon - represents the crescent moon, reading, or tracing",
    "Padmakosha": "Lotus bud - represents a lotus bud, fruit, or offering",
    "Sarpashirsha": "Snake's head - represents snakes, sprinkling water, or demons",
    "Mrigashirsha": "Deer's head - represents calling, fear, affection, or shyness",
    "Simhamukha": "Lion's face - represents perfume vessel, lion, or Ganesha",
    "Kangula": "Tail - represents Goddess Saraswati, plucking flowers",
    "Alapadma": "Fully bloomed lotus - represents lotus, teaching, giving, receiving",
    "Chatura": "Four - represents the number four, picking up, or holding",
    "Bhramara": "Bee - represents a bee, union of Shiva and Shakti",
    "Hamsasya": "Swan's head - represents a swan, beauty, grace",
    "Hamsapaksha": "Swan's wing - represents marking tilak, massage",
    "Sandamsha": "Tongs - represents picking up things, holding arrows",
    "Mukula": "Bud - represents a flower bud, food items, combining elements",
    "Tamrachuda": "Rooster - represents a rooster, hen, writing",
    "Trishula": "Trident - represents Lord Shiva's weapon, the number three",
    
    # Samyuktha Hastas (Double Hand Mudras)
    "Anjali": "Prayer - represents prayer, offering, salutation",
    "Kapota": "Dove - represents embracing, carrying child, support",
    "Karkata": "Crab - represents Yama (god of death), crab, prison",
    "Swastika": "Auspicious - represents being firm, rigid devotion",
    "Dola": "Swing - represents swinging, sporting in water",
    "Pushpaputa": "Handful of flowers - represents offering flowers, giving",
    "Utsanga": "Embrace - represents resting on the lap, sitting",
    "Shivalinga": "Lord Shiva's icon - represents lingam, union",
    "Katakavardana": "Bracelet chain - represents wearing garlands, decorating",
    "Kartariswastika": "Crossed scissors - represents rejection, denial, prohibition",
    "Shakata": "Cart - represents cart, bed, moving",
    "Shankha": "Conch - represents playing conch, conch shell",
    "Chakra": "Wheel - represents discus, rotation, cycle",
    "Samputa": "Casket - represents a box, container, hiding",
    "Pasha": "Noose - represents tying, capturing, bondage",
    "Kilaka": "Lock - represents bolt, togetherness, locking",
    "Matsya": "Fish - represents swimming, fish, mermaid",
    "Kurma": "Tortoise - represents tortoise, stability, support",
    "Varaha": "Boar - represents Lord Vishnu's avatar, wild boar",
    "Garuda": "Eagle - represents Garuda (divine bird), flying",
    "Nagabandha": "Snake knot - represents tied snakes, bondage of serpents",
    "Khatva": "Bed - represents bed, platform, throne",
    
    # Additional modern/regional variations
    "Mukha": "Face - represents the face, identity",
    "Netra": "Eye - represents eyes, vision, seeing",
    "Karna": "Ear - represents ears, listening, hearing",
    "Vaktra": "Mouth - represents mouth, speech, eating"
}

# Emotional context for mudras
MUDRA_EMOTIONS = {
    "Pataka": "neutral",
    "Anjali": "devotional",
    "Mayura": "joyful",
    "Simhamukha": "powerful",
    "Bhramara": "romantic",
    "Ardhachandra": "serene",
    "Mushti": "determined",
    "Alapadma": "graceful"
}

# Category mapping
MUDRA_CATEGORIES = {
    "single_hand": [
        "Pataka", "Tripataka", "Ardhapataka", "Kartarimukha", "Mayura",
        "Ardhachandra", "Arala", "Shukatundaka", "Mushti", "Shikhara",
        "Kapitta", "Katakamukha", "Suchi", "Chandrakala", "Padmakosha",
        "Sarpashirsha", "Mrigashirsha", "Simhamukha", "Kangula", "Alapadma",
        "Chatura", "Bhramara", "Hamsasya", "Hamsapaksha", "Sandamsha",
        "Mukula", "Tamrachuda", "Trishula"
    ],
    "double_hand": [
        "Anjali", "Kapota", "Karkata", "Swastika", "Dola",
        "Pushpaputa", "Utsanga", "Shivalinga", "Katakavardana", "Kartariswastika",
        "Shakata", "Shankha", "Chakra", "Samputa", "Pasha",
        "Kilaka", "Matsya", "Kurma", "Varaha", "Garuda",
        "Nagabandha", "Khatva"
    ]
}


def get_mudra_meaning(mudra_name: str) -> str:
    """Get the meaning of a mudra."""
    return MUDRA_MEANINGS.get(mudra_name, "A traditional hand gesture with symbolic meaning.")


def get_mudra_category(mudra_name: str) -> str:
    """Get the category of a mudra."""
    if mudra_name in MUDRA_CATEGORIES["single_hand"]:
        return "Asamyuktha Hasta (Single Hand)"
    elif mudra_name in MUDRA_CATEGORIES["double_hand"]:
        return "Samyuktha Hasta (Double Hand)"
    else:
        return "Unknown Category"


def generate_mudra_narrative(mudra_detections: list) -> str:
    """
    Generate narrative from detected mudras.
    
    Args:
        mudra_detections: List of {'frame': int, 'mudra': str, 'confidence': float}
    
    Returns:
        str: Narrative description
    """
    if not mudra_detections:
        return "No mudras were detected in this performance."
    
    from collections import Counter
    
    # Count mudra occurrences
    mudra_counts = Counter([m['mudra'] for m in mudra_detections])
    
    narrative_parts = []
    narrative_parts.append(f"Throughout the performance, {len(mudra_detections)} mudra gestures were identified.")
    
    # Top 5 mudras
    top_mudras = mudra_counts.most_common(5)
    
    if top_mudras:
        narrative_parts.append("\nThe most prominent mudras were:")
        for idx, (mudra, count) in enumerate(top_mudras, 1):
            meaning = get_mudra_meaning(mudra)
            category = get_mudra_category(mudra)
            percentage = (count / len(mudra_detections)) * 100
            narrative_parts.append(
                f"{idx}. **{mudra}** ({category}) - appeared {count} times ({percentage:.1f}%), "
                f"representing {meaning}"
            )
    
    # Categorize by type
    single_hand_count = sum(1 for m in mudra_detections 
                           if m['mudra'] in MUDRA_CATEGORIES['single_hand'])
    double_hand_count = sum(1 for m in mudra_detections 
                           if m['mudra'] in MUDRA_CATEGORIES['double_hand'])
    
    if single_hand_count and double_hand_count:
        ratio = single_hand_count / double_hand_count
        narrative_parts.append(
            f"\nThe performance showed a balance of Asamyuktha (single-hand, {single_hand_count}) "
            f"and Samyuktha (double-hand, {double_hand_count}) mudras, "
            f"demonstrating versatility in expression."
        )
    
    return "\n".join(narrative_parts)
