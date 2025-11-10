"""
Aviation Question Generation System - Data Preparation Module (EXPANDED)
This script prepares an EXPANDED training dataset with 30+ aviation topics.
"""

import json
import random
from typing import List, Dict, Tuple

# EXPANDED Aviation corpus - 30+ topics with 120+ question-answer pairs
AVIATION_CORPUS = [
    # 1. AIRSPEED
    {
        "context": "Airspeed is the speed of an aircraft relative to the air through which it is moving. There are several types of airspeed: indicated airspeed (IAS), calibrated airspeed (CAS), true airspeed (TAS), and ground speed (GS). Indicated airspeed is read directly from the airspeed indicator and is affected by instrument and position errors.",
        "questions": [
            "What is airspeed?",
            "What are the different types of airspeed?",
            "What is indicated airspeed and how is it measured?",
            "What factors affect indicated airspeed readings?"
        ]
    },
    # 2. FOUR FORCES
    {
        "context": "The four fundamental forces acting on an aircraft in flight are lift, weight, thrust, and drag. Lift is the upward force created by the wings as air flows over and under them. Weight is the downward force due to gravity. Thrust is the forward force produced by the engine or propeller. Drag is the resistance force that opposes the motion of the aircraft through the air.",
        "questions": [
            "What are the four fundamental forces acting on an aircraft?",
            "How is lift generated on an aircraft?",
            "What produces thrust in an aircraft?",
            "Define drag in aviation terminology."
        ]
    },
    # 3. ANGLE OF ATTACK
    {
        "context": "Angle of attack (AOA) is the angle between the chord line of the wing and the direction of the relative wind. As the angle of attack increases, lift increases up to a critical point called the critical angle of attack, typically around 15-20 degrees. Beyond this point, the airflow separates from the wing's upper surface, causing a stall.",
        "questions": [
            "What is angle of attack?",
            "What happens when angle of attack increases?",
            "What is the critical angle of attack?",
            "What occurs when an aircraft exceeds the critical angle of attack?"
        ]
    },
    # 4. ALTIMETER
    {
        "context": "The altimeter is an instrument that measures the height of an aircraft above a given pressure level, typically mean sea level (MSL). It works by measuring atmospheric pressure, which decreases with altitude. Pilots must adjust the altimeter setting (QNH) to account for local barometric pressure variations. Standard pressure is 29.92 inches of mercury or 1013.25 hectopascals.",
        "questions": [
            "What does an altimeter measure?",
            "How does an altimeter work?",
            "What is the standard pressure setting for altimeters?",
            "Why must pilots adjust altimeter settings?"
        ]
    },
    # 5. ATTITUDE INDICATOR
    {
        "context": "The attitude indicator, also called artificial horizon, displays the aircraft's orientation relative to the earth's horizon. It shows pitch (nose up or down) and bank (roll left or right). This instrument is critical for instrument flight when visual references are not available. It operates using a gyroscope that maintains a fixed reference in space.",
        "questions": [
            "What is an attitude indicator?",
            "What information does the attitude indicator provide?",
            "When is the attitude indicator most critical?",
            "How does an attitude indicator operate?"
        ]
    },
    # 6. NAVIGATION
    {
        "context": "Navigation in aviation involves determining the aircraft's position and maintaining the desired flight path. There are several methods including pilotage (visual reference to landmarks), dead reckoning (calculations based on heading, speed, and time), radio navigation (using VOR, NDB), and satellite navigation (GPS). Modern aircraft typically use a combination of these methods.",
        "questions": [
            "What is aviation navigation?",
            "What are the main navigation methods used in aviation?",
            "What is pilotage?",
            "What navigation systems do modern aircraft use?"
        ]
    },
    # 7. STALL
    {
        "context": "Aerodynamic stall occurs when the angle of attack exceeds the critical angle and smooth airflow over the wing is disrupted. Warning signs include buffeting, reduced control effectiveness, and activation of the stall warning system. Recovery requires reducing the angle of attack by lowering the nose and increasing airspeed. Stalls can occur at any airspeed if the critical angle of attack is exceeded.",
        "questions": [
            "What is an aerodynamic stall?",
            "What are the warning signs of an impending stall?",
            "How do pilots recover from a stall?",
            "Can a stall occur at high airspeed?"
        ]
    },
    # 8. VERTICAL SPEED INDICATOR
    {
        "context": "The vertical speed indicator (VSI) shows the rate of climb or descent in feet per minute. It uses the rate of change of atmospheric pressure to determine vertical velocity. There is typically a slight lag in the instrument's response, which pilots must account for. Modern aircraft often have instantaneous vertical speed indicators (IVSI) that reduce this lag.",
        "questions": [
            "What does the vertical speed indicator measure?",
            "How does a VSI work?",
            "What is the main limitation of traditional VSI instruments?",
            "What improvement does an IVSI provide?"
        ]
    },
    # 9. RUNWAY VISUAL RANGE
    {
        "context": "Runway visual range (RVR) is the distance over which a pilot can see runway markings from the approach end. It is measured by transmissometers or forward scatter meters and reported in feet or meters. RVR is critical for determining landing minimums in instrument meteorological conditions (IMC). Values below minimums require a missed approach.",
        "questions": [
            "What is runway visual range?",
            "How is RVR measured?",
            "Why is RVR important for pilots?",
            "What must pilots do if RVR is below minimums?"
        ]
    },
    # 10. MAGNETIC COMPASS
    {
        "context": "The magnetic compass is a primary navigation instrument that indicates magnetic heading. It contains a magnetized needle that aligns with Earth's magnetic field. Compass errors include variation (difference between magnetic and true north), deviation (errors caused by aircraft magnetic fields), and turning errors during acceleration and deceleration.",
        "questions": [
            "What is a magnetic compass?",
            "How does a magnetic compass work?",
            "What is magnetic variation?",
            "What types of errors affect magnetic compass readings?"
        ]
    },
    # 11. CARBURETOR ICING
    {
        "context": "Carburetor icing occurs when moisture in the air freezes in the carburetor venturi, restricting airflow and reducing engine power. It can occur even in above-freezing temperatures when humidity is high. Signs include loss of RPM and rough engine operation. Carburetor heat is used to prevent or remove ice by directing warm air through the carburetor.",
        "questions": [
            "What is carburetor icing?",
            "Under what conditions can carburetor icing occur?",
            "What are the signs of carburetor icing?",
            "How do pilots prevent or remove carburetor ice?"
        ]
    },
    # 12. WEIGHT AND BALANCE
    {
        "context": "Weight and balance calculations are critical for safe flight operations. The aircraft's center of gravity (CG) must remain within specified limits throughout the flight. Loading beyond forward or aft CG limits affects stability and control. Pilots must calculate total weight, moment arms, and ensure the loaded CG position is within the aircraft's envelope before flight.",
        "questions": [
            "Why are weight and balance calculations important?",
            "What is center of gravity in aviation?",
            "What happens if CG is outside limits?",
            "What must pilots calculate before flight?"
        ]
    },
    # 13. CONTROLLED AIRSPACE
    {
        "context": "Controlled airspace includes Class A, B, C, D, and E airspace, each with specific requirements for entry and operations. Class A extends from 18,000 feet MSL to FL600 and requires an instrument rating and clearance. Class B surrounds major airports with specific VFR requirements. Class C and D have progressively less restrictive requirements. Class E is controlled airspace not classified as A, B, C, or D.",
        "questions": [
            "What are the classes of controlled airspace?",
            "What are the requirements for Class A airspace?",
            "Where is Class B airspace typically found?",
            "What is Class E airspace?"
        ]
    },
    # 14. GO-AROUND
    {
        "context": "A go-around or missed approach is a safety maneuver performed when a landing cannot be safely completed. Common reasons include unstable approach, poor visibility, runway obstruction, or loss of visual reference. The procedure involves applying full power, retracting flaps incrementally, establishing a positive rate of climb, and following published missed approach procedures or ATC instructions.",
        "questions": [
            "What is a go-around?",
            "When should pilots execute a go-around?",
            "What are the key steps in a go-around procedure?",
            "What must pilots follow during a missed approach?"
        ]
    },
    # 15. TRANSPONDER
    {
        "context": "The transponder is an electronic device that responds to radar interrogation from air traffic control. Mode A transmits a four-digit code assigned by ATC. Mode C adds altitude information from the aircraft's encoding altimeter. Mode S provides additional data link capability. Pilots must set the assigned squawk code and ensure the transponder is on the appropriate mode for their flight.",
        "questions": [
            "What is a transponder?",
            "What are the different transponder modes?",
            "What does Mode C transmit?",
            "What must pilots do with their transponder?"
        ]
    },
    # 16. VOR NAVIGATION
    {
        "context": "VOR (VHF Omnidirectional Range) is a radio navigation system that provides azimuth information to aircraft. VOR stations transmit signals on frequencies between 108.0 and 117.95 MHz. The system works by comparing the phase difference between a reference signal and a variable signal. Pilots can track TO or FROM a VOR station by selecting a radial and following the course deviation indicator (CDI).",
        "questions": [
            "What does VOR stand for and what does it provide?",
            "How does the VOR navigation system work?",
            "What frequency range do VOR stations use?",
            "How do pilots use VOR for navigation?"
        ]
    },
    # 17. ILS APPROACH
    {
        "context": "The Instrument Landing System (ILS) provides precision guidance for aircraft approaching a runway. It consists of a localizer (horizontal guidance), glideslope (vertical guidance), and marker beacons (distance information). The localizer provides lateral guidance aligned with the runway centerline. The glideslope provides vertical guidance, typically at a 3-degree angle. ILS enables approaches in low visibility conditions.",
        "questions": [
            "What is an ILS and what does it provide?",
            "What are the main components of an ILS?",
            "What does the localizer provide?",
            "What is the typical glideslope angle?"
        ]
    },
    # 18. DENSITY ALTITUDE
    {
        "context": "Density altitude is pressure altitude corrected for non-standard temperature. It is used to determine aircraft performance. High density altitude reduces aircraft performance by decreasing engine power, propeller efficiency, and wing lift. Factors that increase density altitude include high temperature, high elevation, and low atmospheric pressure. Pilots must account for density altitude when calculating takeoff distance, climb rate, and landing distance.",
        "questions": [
            "What is density altitude?",
            "How does high density altitude affect aircraft performance?",
            "What factors increase density altitude?",
            "Why must pilots calculate density altitude?"
        ]
    },
    # 19. CROSSWIND LANDING
    {
        "context": "Crosswind landing techniques are essential for safe operations when wind is not aligned with the runway. The two primary methods are the crab method and the wing-low (sideslip) method. The crab method involves pointing the nose into the wind to maintain runway alignment. The wing-low method uses aileron to lower the upwind wing while applying opposite rudder to maintain runway centerline. Most pilots use a combination of both techniques.",
        "questions": [
            "What are crosswind landing techniques?",
            "What is the crab method?",
            "What is the wing-low method?",
            "How do pilots handle crosswind landings?"
        ]
    },
    # 20. HYDROPLANING
    {
        "context": "Hydroplaning occurs when a layer of water builds up between the aircraft tires and the runway surface, causing loss of tire contact and reduced braking effectiveness. There are three types: dynamic hydroplaning (high speed on wet surfaces), viscous hydroplaning (thin film on smooth surfaces), and reverted rubber hydroplaning (locked wheels on wet surfaces). The minimum hydroplaning speed can be estimated as 9 times the square root of tire pressure in PSI.",
        "questions": [
            "What is hydroplaning?",
            "What are the three types of hydroplaning?",
            "How can minimum hydroplaning speed be estimated?",
            "What are the dangers of hydroplaning?"
        ]
    },
    # 21. PITOT-STATIC SYSTEM
    {
        "context": "The pitot-static system provides pressure information to flight instruments including the airspeed indicator, altimeter, and vertical speed indicator. The pitot tube measures ram air pressure (total pressure) while the static port measures ambient atmospheric pressure (static pressure). Blockage of the pitot tube affects only the airspeed indicator, while static port blockage affects all three instruments. Alternate static sources are provided in case of primary static port failure.",
        "questions": [
            "What is the pitot-static system?",
            "What instruments use the pitot-static system?",
            "What does the pitot tube measure?",
            "What happens if the static port is blocked?"
        ]
    },
    # 22. GROUND EFFECT
    {
        "context": "Ground effect is the increased aerodynamic efficiency that occurs when an aircraft flies within one wingspan height of the ground. In ground effect, induced drag decreases and lift increases due to interference with wingtip vortices. This phenomenon is most noticeable during takeoff and landing. Pilots may experience a floating sensation during landing as the aircraft requires less power to maintain altitude. When leaving ground effect during takeoff, additional thrust is required.",
        "questions": [
            "What is ground effect?",
            "When does ground effect occur?",
            "How does ground effect affect aircraft performance?",
            "What must pilots consider when leaving ground effect?"
        ]
    },
    # 23. WAKE TURBULENCE
    {
        "context": "Wake turbulence is caused by wingtip vortices generated by all aircraft in flight. These rotating cylinders of air can persist for several minutes and pose a hazard to following aircraft, especially during takeoff and landing. Heavy aircraft generate stronger wake turbulence. Vortices sink at approximately 300-500 feet per minute and tend to drift with wind. Pilots must maintain proper separation and avoid flying directly behind or below preceding aircraft.",
        "questions": [
            "What causes wake turbulence?",
            "How long can wake turbulence persist?",
            "Which aircraft generate the strongest wake turbulence?",
            "How should pilots avoid wake turbulence?"
        ]
    },
    # 24. HYPOXIA
    {
        "context": "Hypoxia is a state of oxygen deficiency in the body sufficient to impair functions of the brain and other organs. Types include hypoxic hypoxia (insufficient oxygen in air), hypemic hypoxia (blood's inability to carry oxygen), stagnant hypoxia (inadequate blood circulation), and histotoxic hypoxia (cells' inability to use oxygen). Symptoms include euphoria, impaired judgment, decreased reaction time, and visual impairment. At cabin altitudes above 12,500 feet MSL, supplemental oxygen is required after 30 minutes.",
        "questions": [
            "What is hypoxia?",
            "What are the types of hypoxia?",
            "What are the symptoms of hypoxia?",
            "When is supplemental oxygen required?"
        ]
    },
    # 25. SPATIAL DISORIENTATION
    {
        "context": "Spatial disorientation occurs when a pilot's sensory perception of position, attitude, or motion conflicts with reality. The vestibular system (inner ear) can provide false sensations during flight, especially in instrument meteorological conditions. Common illusions include the leans (false perception of bank), graveyard spiral (failure to detect gradual turn), and somatogravic illusion (acceleration perceived as pitch up). Prevention requires trusting flight instruments over bodily sensations.",
        "questions": [
            "What is spatial disorientation?",
            "What causes spatial disorientation?",
            "What is the leans illusion?",
            "How do pilots prevent spatial disorientation?"
        ]
    },
    # 26. WINDSHEAR
    {
        "context": "Windshear is a sudden change in wind speed or direction over a short distance. It can occur at any altitude but is most hazardous during takeoff and landing. Low-level windshear, particularly from microbursts, can cause rapid changes in airspeed and vertical speed. Pilots encountering windshear should apply maximum thrust, adjust pitch to maintain safe airspeed, and be prepared to execute a go-around. Modern aircraft are equipped with windshear detection systems.",
        "questions": [
            "What is windshear?",
            "Why is windshear hazardous during takeoff and landing?",
            "What is a microburst?",
            "What should pilots do when encountering windshear?"
        ]
    },
    # 27. ATIS
    {
        "context": "Automatic Terminal Information Service (ATIS) is a continuous broadcast of recorded aeronautical information at towered airports. ATIS includes weather information, active runways, approach procedures in use, and other pertinent data. Each ATIS broadcast is identified by a phonetic letter (Alpha, Bravo, Charlie, etc.) that changes when information is updated. Pilots must acknowledge receipt of ATIS information when contacting approach control or tower.",
        "questions": [
            "What is ATIS?",
            "What information does ATIS provide?",
            "How are ATIS broadcasts identified?",
            "When must pilots acknowledge ATIS?"
        ]
    },
    # 28. NOTAM
    {
        "context": "A NOTAM (Notice to Airmen) is a notice containing information concerning the establishment, condition, or change in any aeronautical facility, service, procedure, or hazard. NOTAMs are issued for runway closures, navaid outages, airspace restrictions, and other time-critical information. Types include NOTAM-D (distant), FDC NOTAMs (regulatory), pointer NOTAMs, and military NOTAMs. Pilots are required to check NOTAMs as part of preflight planning.",
        "questions": [
            "What is a NOTAM?",
            "What information do NOTAMs provide?",
            "What are the different types of NOTAMs?",
            "When must pilots check NOTAMs?"
        ]
    },
    # 29. V-SPEEDS
    {
        "context": "V-speeds are standardized airspeeds used for aircraft operation. Important V-speeds include VS0 (stall speed in landing configuration), VS1 (stall speed in clean configuration), VR (rotation speed), VX (best angle of climb speed), VY (best rate of climb speed), VA (maneuvering speed), VFE (maximum flap extended speed), VNO (maximum structural cruising speed), and VNE (never exceed speed). These speeds vary with aircraft weight and configuration.",
        "questions": [
            "What are V-speeds?",
            "What is VY?",
            "What is the difference between VX and VY?",
            "Why do V-speeds vary with weight?"
        ]
    },
    # 30. PREFLIGHT INSPECTION
    {
        "context": "The preflight inspection is a systematic visual and physical check of the aircraft before flight. It includes checking fuel quantity and quality, oil level, control surface movement, tire condition, lights, and pitot-static system. The inspection follows a standardized pattern specific to each aircraft type. Any discrepancies must be addressed before flight. The preflight inspection is the pilot's responsibility and cannot be delegated.",
        "questions": [
            "What is a preflight inspection?",
            "What items are checked during preflight inspection?",
            "Who is responsible for the preflight inspection?",
            "What must pilots do if discrepancies are found?"
        ]
    },
    # 31. FUEL MANAGEMENT
    {
        "context": "Proper fuel management is critical for flight safety. Pilots must calculate fuel required for the flight including taxi, takeoff, climb, cruise, descent, approach, landing, and required reserves. Federal regulations require fuel for VFR day flight to the destination plus 30 minutes, and for IFR flight to the destination, alternate (if required), and 45 minutes. Pilots must monitor fuel consumption during flight and verify fuel quantity before engine start.",
        "questions": [
            "Why is fuel management important?",
            "What fuel reserves are required for VFR flight?",
            "What fuel reserves are required for IFR flight?",
            "When must pilots verify fuel quantity?"
        ]
    },
    # 32. EMERGENCY LOCATOR TRANSMITTER
    {
        "context": "An Emergency Locator Transmitter (ELT) is a device that transmits a distress signal on 121.5 MHz and 406 MHz frequencies when activated by impact or manually. Modern 406 MHz ELTs transmit identification and position data to satellites. ELTs must be inspected within 12 calendar months and batteries replaced when 50% of useful life has expired. Pilots should ensure the ELT switch is in the armed position before flight.",
        "questions": [
            "What is an ELT?",
            "What frequencies do ELTs use?",
            "How often must ELTs be inspected?",
            "When are ELT batteries replaced?"
        ]
    },
    # 33. RADIO COMMUNICATION PROCEDURES
    {
        "context": "Standard radio communication procedures ensure clear and efficient communication between pilots and air traffic control. Communications should be concise, clear, and use standard phraseology. Pilots must listen before transmitting, identify the facility being called, identify their aircraft, and state their request. Read back all runway assignments, hold short instructions, and clearances. The phonetic alphabet is used to avoid confusion when spelling words or callsigns.",
        "questions": [
            "Why are standard radio procedures important?",
            "What should pilots include when initiating radio contact?",
            "What instructions must be read back?",
            "When is the phonetic alphabet used?"
        ]
    },
    # 34. SQUAWK CODES
    {
        "context": "Squawk codes are four-digit transponder codes assigned by air traffic control. Standard codes include 1200 (VFR flight), 7500 (hijacking), 7600 (communication failure), and 7700 (emergency). Pilots must enter the assigned code and verify MODE C altitude reporting is enabled. When changing codes, pilots should avoid passing through 7500, 7600, or 7700 to prevent inadvertent emergency declaration.",
        "questions": [
            "What are squawk codes?",
            "What does squawk 1200 indicate?",
            "What are the emergency squawk codes?",
            "What should pilots avoid when changing transponder codes?"
        ]
    },
    # 35. AIRWORTHINESS DIRECTIVES
    {
        "context": "Airworthiness Directives (ADs) are legally enforceable rules issued by aviation authorities to correct unsafe conditions in aircraft, engines, propellers, or appliances. Compliance with ADs is mandatory and must be recorded in the aircraft maintenance records. ADs may require immediate action, compliance within a specified time, or compliance at the next scheduled maintenance. Pilots must ensure all applicable ADs have been complied with before flight.",
        "questions": [
            "What are Airworthiness Directives?",
            "Is compliance with ADs mandatory?",
            "Where is AD compliance recorded?",
            "When must pilots verify AD compliance?"
        ]
    }
]


def prepare_training_data(corpus: List[Dict], output_file: str = "aviation_training_data.json"):
    """
    Prepare training data in the format required for T5 fine-tuning.

    Args:
        corpus: List of dictionaries containing contexts and questions
        output_file: Path to save the prepared data

    Returns:
        List of training examples
    """
    training_data = []

    for item in corpus:
        context = item["context"]
        questions = item["questions"]

        for question in questions:
            # Format for T5: "generate question: <context>"
            training_example = {
                "input_text": f"generate question: {context}",
                "target_text": question,
                "context": context
            }
            training_data.append(training_example)

    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Prepared {len(training_data)} training examples")
    print(f"✓ Saved to {output_file}")

    return training_data


def split_train_validation(data: List[Dict], val_split: float = 0.15,
                           output_prefix: str = "aviation"):
    """
    Split data into training and validation sets.

    Args:
        data: List of training examples
        val_split: Fraction of data for validation
        output_prefix: Prefix for output files
    """
    random.shuffle(data)
    split_idx = int(len(data) * (1 - val_split))

    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Save splits
    with open(f"{output_prefix}_train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(f"{output_prefix}_val.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Training set: {len(train_data)} examples")
    print(f"✓ Validation set: {len(val_data)} examples")

    return train_data, val_data


def generate_statistics(data: List[Dict]):
    """Generate statistics about the dataset."""
    total_examples = len(data)
    contexts = set([item['context'] for item in data])

    avg_context_length = sum(len(item['context'].split()) for item in data) / total_examples
    avg_question_length = sum(len(item['target_text'].split()) for item in data) / total_examples

    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total training examples: {total_examples}")
    print(f"Unique contexts (topics): {len(contexts)}")
    print(f"Average context length: {avg_context_length:.1f} words")
    print(f"Average question length: {avg_question_length:.1f} words")
    print(f"Questions per topic: {total_examples / len(contexts):.1f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    print("="*50)
    print("Aviation Question Generation - Data Preparation")
    print("EXPANDED CORPUS - 35 Topics")
    print("="*50 + "\n")

    # Prepare training data
    training_data = prepare_training_data(AVIATION_CORPUS)

    # Split into train/validation
    train_data, val_data = split_train_validation(training_data)

    # Generate statistics
    generate_statistics(training_data)

    # Show sample
    print("SAMPLE TRAINING EXAMPLE:")
    print("-"*50)
    sample = random.choice(training_data)
    print(f"Input: {sample['input_text'][:100]}...")
    print(f"Target: {sample['target_text']}")
    print("-"*50)

    print("\n✅ EXPANDED dataset ready!")
    print(f"   - 35 aviation topics")
    print(f"   - {len(training_data)} total examples")
    print(f"   - {len(train_data)} training examples")
    print(f"   - {len(val_data)} validation examples")
    print("\nThis will significantly improve model training!")