"""
Synthetic Content Description Generator
Generates 500 content descriptions labeled against YouTube's 14 policy categories.
Each description is traceable to a specific guideline example from:
https://support.google.com/youtube/answer/6162278

Author: Bhuvan Dontha
"""

import json
import random
import csv
import os

random.seed(42)

# Load policy taxonomy
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR)

with open(os.path.join(DATA_DIR, "policy_taxonomy.json"), "r") as f:
    taxonomy = json.load(f)

# ============================================================
# CONTENT TEMPLATES: Each maps to a specific guideline example
# Format: (description_template, policy_id, severity_tier, source_note)
# ============================================================

CONTENT_TEMPLATES = [
    # --- INAPPROPRIATE LANGUAGE ---
    # GREEN
    ("Comedy sketch where host says 'what the hell' and 'damn' throughout", "POL-LANG", "GREEN", "Abbreviated/obscured profanity"),
    ("Music video with backing track containing bleeped profanity", "POL-LANG", "GREEN", "Obscured profanity in music"),
    ("Stand-up comedy set with frequent use of 'bitch' and 'asshole'", "POL-LANG", "GREEN", "Moderate profanity in comedy"),
    ("Podcast episode where hosts use 'shit' multiple times in conversation", "POL-LANG", "GREEN", "Moderate profanity in video"),
    ("Vlog intro music containing the word 'damn' in lyrics", "POL-LANG", "GREEN", "Moderate profanity in music"),
    # YELLOW
    ("Video titled 'This is absolute shit' reviewing a bad product", "POL-LANG", "YELLOW", "Moderate profanity in title"),
    ("Thumbnail with text overlay reading 'BULL***T'", "POL-LANG", "YELLOW", "Moderate profanity in thumbnail"),
    ("Rap music video titled 'That Sh*t Hits Different'", "POL-LANG", "YELLOW", "Profanity in music title"),
    ("Commentary video where profanity is used in most sentences throughout", "POL-LANG", "YELLOW", "Focal profanity usage"),
    # RED
    ("Video titled 'F*ck This Product' with uncensored profanity", "POL-LANG", "RED", "Strong profanity in title"),
    ("Thumbnail displaying the f-word prominently in large text", "POL-LANG", "RED", "Strong profanity in thumbnail"),
    ("Video containing racial slurs used casually in conversation", "POL-LANG", "RED", "Hateful language/slurs"),
    ("Content repeatedly using homophobic slurs as insults", "POL-LANG", "RED", "Extreme profanity/slurs"),

    # --- VIOLENCE ---
    # GREEN
    ("Bodycam footage of routine police arrest with forcible handcuffing", "POL-VIOL", "GREEN", "Law enforcement in action"),
    ("Gameplay video of Call of Duty with combat starting after intro", "POL-VIOL", "GREEN", "Unedited gameplay after 15s"),
    ("Action movie clip showing character getting punched with minimal blood", "POL-VIOL", "GREEN", "Dramatized non-graphic violence"),
    ("Documentary about World War 2 showing fully blurred archival bodies", "POL-VIOL", "GREEN", "Historical dead bodies obscured"),
    ("Boxing match at professional arena with visible face punches", "POL-VIOL", "GREEN", "Combat sports professional setting"),
    ("Self-defense tutorial demonstrating basic blocking techniques", "POL-VIOL", "GREEN", "Fights educational context"),
    ("News report mentioning a local homicide without graphic descriptions", "POL-VIOL", "GREEN", "Coverage of tragedies limited"),
    ("Hunting video where the moment of kill is indiscernible in the distance", "POL-VIOL", "GREEN", "Hunting non-graphic"),
    # YELLOW
    ("History channel showing dead bodies with visible wounds in documentary", "POL-VIOL", "YELLOW", "Dead bodies injury educational"),
    ("Gaming video with graphic beheading scene in the thumbnail", "POL-VIOL", "YELLOW", "Graphic game violence thumbnail"),
    ("Raw footage of soldiers firing weapons in a conflict zone", "POL-VIOL", "YELLOW", "Armed conflict raw footage"),
    ("Movie scene showing character with visibly broken bones and blood", "POL-VIOL", "YELLOW", "Dramatized severe injuries"),
    ("Street fight compilation with visible injuries in educational breakdown", "POL-VIOL", "YELLOW", "Street fights educational"),
    ("Documentary on recent homicide with descriptive language about death", "POL-VIOL", "YELLOW", "Tragedy graphic details"),
    # RED
    ("Video showing graphic dead bodies with exposed wounds no context", "POL-VIOL", "RED", "Graphic dead bodies non-educational"),
    ("Content glorifying gang violence and encouraging retaliation", "POL-VIOL", "RED", "Glorification of violence"),
    ("Gameplay where player aggregates NPCs for mass killing compilation", "POL-VIOL", "RED", "Manufactured shocking gameplay"),
    ("Video showing real-world execution footage from conflict zone", "POL-VIOL", "RED", "Ultra graphic violent acts"),
    ("Content depicting graphic torture of a restrained person", "POL-VIOL", "RED", "Graphic torture"),
    ("Graphic depiction of violence between children in schoolyard", "POL-VIOL", "RED", "Violence concerning minors"),

    # --- ADULT CONTENT ---
    # GREEN
    ("Romantic comedy scene with characters kissing passionately", "POL-ADULT", "GREEN", "Romance or kissing"),
    ("Sex education video explaining contraception methods with diagrams", "POL-ADULT", "GREEN", "Non-graphic sex education"),
    ("Music video featuring choreographed hip-hop dancing in a studio", "POL-ADULT", "GREEN", "Sexually graphic dance professional"),
    ("Discussion video about navigating relationships and dating", "POL-ADULT", "GREEN", "Discussions of relationships"),
    ("Woman breastfeeding her infant in a parenting vlog", "POL-ADULT", "GREEN", "Breastfeeding with child"),
    # YELLOW
    ("Classical art painting showing discernible sexual acts in thumbnail", "POL-ADULT", "YELLOW", "Classical art sexual thumbnail"),
    ("Educational video about STDs using animated depictions of sex", "POL-ADULT", "YELLOW", "Non-arousing sexual education"),
    ("Prank video where friends pretend to be caught in sexual situation", "POL-ADULT", "YELLOW", "Sexual theme pranks"),
    ("Dance video with deliberate camera zoom on sexual body parts", "POL-ADULT", "YELLOW", "Deliberate focus sexual parts dance"),
    # RED
    ("Video showing exposed nudity with clearly visible sexual body parts", "POL-ADULT", "RED", "Full nudity"),
    ("Content discussing sexual fetishes with explicit descriptions", "POL-ADULT", "RED", "Discussion of fetishes"),
    ("Video featuring sex toys being demonstrated and reviewed", "POL-ADULT", "RED", "Sex toys or devices"),
    ("Erotic dance performance explicitly intended to arouse the audience", "POL-ADULT", "RED", "Erotic dance arousal intent"),
    ("Video with thumbnail showing 18+ adult content warning", "POL-ADULT", "RED", "Adult content thumbnail"),

    # --- SHOCKING CONTENT ---
    # GREEN
    ("Dermatologist explaining ear wax removal procedure educationally", "POL-SHOCK", "GREEN", "Medical procedure educational"),
    ("Cooking tutorial showing raw fish being prepared on cutting board", "POL-SHOCK", "GREEN", "Unsensational food preparation"),
    ("Home video of person accidentally walking into glass door", "POL-SHOCK", "GREEN", "Accident no visible injury"),
    ("Tattoo procedure video showing the tattooing process with minimal blood", "POL-SHOCK", "GREEN", "Cosmetic procedure educational"),
    # YELLOW
    ("Pimple popping compilation by a medical professional with explanations", "POL-SHOCK", "YELLOW", "Medical procedure focal bodily"),
    ("Car crash dashcam footage with strong moment of impact", "POL-SHOCK", "YELLOW", "High impact accident"),
    ("Mukbang video eating live octopus tentacles in theatrical manner", "POL-SHOCK", "YELLOW", "Sensational mukbang"),
    # RED
    ("Close-up video of extreme surgical procedure with exposed organs no context", "POL-SHOCK", "RED", "Gruesome gory no context"),
    ("Content showing person eating live insects solely to shock viewers", "POL-SHOCK", "RED", "Eating live animals shock"),
    ("Video focused entirely on graphic wound close-ups with no medical context", "POL-SHOCK", "RED", "Disgusting no context"),

    # --- HARMFUL ACTS ---
    # GREEN
    ("Professional wingsuit flying over mountains filmed with drone", "POL-HARM", "GREEN", "Professional stunts"),
    ("Compilation of people failing at skateboard tricks with minor falls", "POL-HARM", "GREEN", "Fail compilation non-graphic"),
    ("Ice bucket challenge video with friends laughing", "POL-HARM", "GREEN", "Pranks no risk"),
    ("Video debunking common health misinformation myths", "POL-HARM", "GREEN", "Debunking misinformation"),
    # YELLOW
    ("Parkour video on top of skyscrapers without safety equipment", "POL-HARM", "YELLOW", "High risk activities"),
    ("Prank where person is told their spouse died causing extreme distress", "POL-HARM", "YELLOW", "Emotional distress prank"),
    ("Challenge video eating Carolina Reaper pepper with visible distress", "POL-HARM", "YELLOW", "Eating harmful substances"),
    ("Review comparing different vaping juice flavors and nicotine levels", "POL-HARM", "YELLOW", "Tobacco product reviews"),
    # RED
    ("Video challenging viewers to drink bleach for supposed health benefits", "POL-HARM", "RED", "Dangerous challenge critical harm"),
    ("Content promoting anti-vaccination conspiracy theories as facts", "POL-HARM", "RED", "Anti-vaccination misinformation"),
    ("Video claiming COVID-19 vaccines contain microchips", "POL-HARM", "RED", "COVID misinformation"),
    ("Prank involving fake bomb scare in a crowded public place", "POL-HARM", "RED", "Terrorism prank"),
    ("Content claiming that cancer can be cured through diet alone", "POL-HARM", "RED", "Medical misinformation"),
    ("Video making false claims about election fraud to undermine voting", "POL-HARM", "RED", "Election misinformation"),

    # --- HATEFUL & DEROGATORY ---
    # GREEN
    ("News report covering discrimination against a religious minority", "POL-HATE", "GREEN", "News discrimination"),
    ("Comedy sketch satirizing and condemning racist stereotypes", "POL-HATE", "GREEN", "Comedy condemning hate"),
    ("Educational video about the history of the civil rights movement", "POL-HATE", "GREEN", "Educational with obscured slurs"),
    ("Public debate about immigration policy without hateful rhetoric", "POL-HATE", "GREEN", "Public debate protected groups"),
    # YELLOW
    ("Documentary with unbleeped racial slurs for educational impact", "POL-HATE", "YELLOW", "Unobscured slurs educational"),
    ("Political debate using offensive language about ethnic minorities", "POL-HATE", "YELLOW", "Offensive language educational debate"),
    # RED
    ("Video stating all people of a certain nationality are inferior", "POL-HATE", "RED", "Disparaging protected group"),
    ("Content promoting white supremacist ideology and symbols", "POL-HATE", "RED", "Promoting hate groups"),
    ("Video singling out a specific person for targeted harassment campaign", "POL-HATE", "RED", "Singling out for harassment"),
    ("Content denying a documented genocide and calling survivors liars", "POL-HATE", "RED", "Denying tragic events"),
    ("Video inciting viewers to commit violence against disabled people", "POL-HATE", "RED", "Inciting violence protected group"),

    # --- RECREATIONAL DRUGS ---
    # GREEN
    ("Documentary about the history of the war on drugs in America", "POL-DRUG", "GREEN", "Drug education documentary"),
    ("Personal story about recovering from opioid addiction", "POL-DRUG", "GREEN", "Addiction recovery"),
    ("Music video with rapper shown briefly near a bag of white powder", "POL-DRUG", "GREEN", "Music drugs fleeting"),
    ("GTA gameplay where character completes drug dealing mission", "POL-DRUG", "GREEN", "Gaming drug deals"),
    # YELLOW
    ("Movie scene showing character injecting heroin in dramatic context", "POL-DRUG", "YELLOW", "Drug consumption dramatized"),
    ("Dramatized TV series episode depicting cocaine manufacturing lab", "POL-DRUG", "YELLOW", "Drug creation dramatized"),
    # RED
    ("Video reviewing different strains of cannabis and rating their effects", "POL-DRUG", "RED", "Drug reviews"),
    ("Tutorial on how to grow marijuana plants at home", "POL-DRUG", "RED", "Cannabis farming tips"),
    ("Video sharing links to websites where drugs can be purchased online", "POL-DRUG", "RED", "Selling drugs online"),
    ("Content glorifying a drug cartel leader and their lifestyle", "POL-DRUG", "RED", "Drug trafficking glorification"),

    # --- FIREARMS ---
    # GREEN
    ("Gun review video at a professional shooting range with safety gear", "POL-FIRE", "GREEN", "Shooting range gun review"),
    ("Tutorial on cleaning and maintaining a hunting rifle", "POL-FIRE", "GREEN", "Firearm maintenance"),
    ("Discussion video about proposed gun control legislation", "POL-FIRE", "GREEN", "Gun legislation discussion"),
    ("Airsoft team competition video with proper protective equipment", "POL-FIRE", "GREEN", "Airsoft responsible use"),
    # YELLOW
    ("Person shooting targets in their backyard near residential area", "POL-FIRE", "YELLOW", "Guns uncontrolled environment"),
    ("BB gun battle without protective gear in public park", "POL-FIRE", "YELLOW", "BB guns no protection"),
    # RED
    ("Step-by-step guide on how to 3D print a functional firearm", "POL-FIRE", "RED", "3D printed guns"),
    ("Video promoting a specific online gun store with discount codes", "POL-FIRE", "RED", "Gun sales promotion"),
    ("Tutorial on how to convert semi-automatic rifle to fully automatic", "POL-FIRE", "RED", "Gun modification instructions"),
    ("Video showing a 12-year-old child firing a rifle without adult present", "POL-FIRE", "RED", "Minors guns unsupervised"),

    # --- CONTROVERSIAL ISSUES ---
    # GREEN
    ("PSA video about suicide prevention hotline and mental health resources", "POL-CONTROV", "GREEN", "Preventing controversial issues"),
    ("News segment covering abortion legislation changes factually", "POL-CONTROV", "GREEN", "Non-graphic abortion coverage"),
    ("Movie scene showing character jumping from bridge in larger narrative", "POL-CONTROV", "GREEN", "Dramatized controversial"),
    ("Brief mention 'next week we discuss declining suicide rates' in outro", "POL-CONTROV", "GREEN", "Fleeting reference"),
    # YELLOW
    ("Documentary showing a child being verbally abused by parent", "POL-CONTROV", "YELLOW", "Child abuse non-graphic"),
    ("Personal recovery story from someone who overcame bulimia", "POL-CONTROV", "YELLOW", "Eating disorder recovery"),
    ("Thumbnail showing graphic image of self-harm scars for awareness video", "POL-CONTROV", "YELLOW", "Controversial thumbnail graphic"),
    # RED
    ("Video titled 'how to end your life with dignity' with instructions", "POL-CONTROV", "RED", "Promote suicide"),
    ("Content showing close-up of fresh self-harm cuts with visible blood", "POL-CONTROV", "RED", "Graphic self-harm"),
    ("First person account describing child abuse in graphic sexual detail", "POL-CONTROV", "RED", "Graphic child abuse descriptive"),
    ("Video promoting extreme caloric restriction and showing emaciated body", "POL-CONTROV", "RED", "Eating disorder triggering"),

    # --- SENSITIVE EVENTS ---
    # GREEN
    ("News reporter covering a natural disaster aftermath respectfully", "POL-SENSIT", "GREEN", "News reporting sensitive"),
    ("Documentary discussing the impact of a mass shooting on community", "POL-SENSIT", "GREEN", "Documentary sensitive event"),
    # RED
    ("Video using clickbait title about recent tragedy to get views", "POL-SENSIT", "RED", "Clickbait tragedy"),
    ("Content selling merchandise themed around a recent terrorist attack", "POL-SENSIT", "RED", "Profiting sensitive event"),
    ("Video exploiting recent earthquake footage to promote survival gear", "POL-SENSIT", "RED", "Exploiting tragedy"),

    # --- ENABLING DISHONEST BEHAVIOR ---
    # GREEN
    ("Urban exploration of abandoned hospital with documented permission", "POL-DISHON", "GREEN", "Urban exploration permitted"),
    ("Ethical hacking tutorial explaining penetration testing methodology", "POL-DISHON", "GREEN", "Ethical hacking"),
    ("Documentary about infamous bank robbers and their capture", "POL-DISHON", "GREEN", "Crime documentary"),
    ("Video about phone jailbreaking and custom ROM installation", "POL-DISHON", "GREEN", "Digital lifehacks"),
    # RED
    ("Tutorial on how to install spyware on someone's phone secretly", "POL-DISHON", "RED", "Tracking without consent"),
    ("Video promoting essay writing services for university assignments", "POL-DISHON", "RED", "Essay writing service"),
    ("Guide on creating fake government identification documents", "POL-DISHON", "RED", "Fake ID creation"),
    ("Video showing how to break into houses using lockpicking tools", "POL-DISHON", "RED", "Trespassing promotion"),

    # --- INAPPROPRIATE FOR KIDS ---
    # GREEN
    ("Animated educational video teaching kids about stranger danger", "POL-KIDS", "GREEN", "Kids educational safety"),
    ("Family-friendly cooking tutorial for kids making cupcakes", "POL-KIDS", "GREEN", "Safe DIY kids"),
    ("Kids yoga and fitness routine with animated characters", "POL-KIDS", "GREEN", "Sports fitness kids"),
    # RED
    ("Kids channel video showing realistic toy guns being fired", "POL-KIDS", "RED", "Guns kids content"),
    ("Animated content for kids featuring adult horror characters like Momo", "POL-KIDS", "RED", "Horror kids Momo"),
    ("Kids challenge video encouraging eating non-edible craft supplies", "POL-KIDS", "RED", "Eating non-edible kids"),
    ("Content disguised as kids cartoon but containing sexual innuendos", "POL-KIDS", "RED", "Adult themes kids disguised"),
    ("Video encouraging kids to bully other students as a prank", "POL-KIDS", "RED", "Kids bullying promotion"),

    # --- INCENDIARY & DEMEANING ---
    # YELLOW
    ("Video focused entirely on publicly shaming a specific person", "POL-INCEND", "YELLOW", "Shaming individual"),
    ("Content claiming a school shooting was staged with crisis actors", "POL-INCEND", "YELLOW", "Crisis actors claim"),
    ("Compilation video dedicated to insulting and mocking a public figure", "POL-INCEND", "YELLOW", "Malicious personal attacks"),
    ("Video harassing and intimidating a small business owner repeatedly", "POL-INCEND", "YELLOW", "Harassment intimidation"),

    # --- TOBACCO ---
    # YELLOW
    ("Video promoting a new line of flavored vape pens and e-liquids", "POL-TOBAC", "YELLOW", "Promoting tobacco products"),
    ("Content reviewing and comparing different cigar brands favorably", "POL-TOBAC", "YELLOW", "Cigarettes cigars promotion"),
    ("Influencer promoting herbal cigarettes as a healthy alternative", "POL-TOBAC", "YELLOW", "Herbal cigarettes promotion"),
]


def add_variation(template: str) -> str:
    """Add slight randomized variation to templates for realism."""
    prefixes = [
        "", "New upload: ", "Trending: ", "Reupload of ", "Part 2: ",
        "Full video: ", "Extended cut: ", "Viewer requested: ", ""
    ]
    suffixes = [
        "", " (2024)", " - full version", " | unedited",
        " - viewer discretion advised", " #trending", ""
    ]
    return random.choice(prefixes) + template + random.choice(suffixes)


def generate_content_descriptions(n: int = 500) -> list:
    """Generate n synthetic content descriptions with ground truth labels."""
    descriptions = []
    content_id = 1

    # First pass: use all templates once
    for template, policy_id, severity, source in CONTENT_TEMPLATES:
        descriptions.append({
            "content_id": f"CNT-{content_id:04d}",
            "description": add_variation(template),
            "true_policy_id": policy_id,
            "true_policy_name": next(
                c["policy_name"] for c in taxonomy["categories"]
                if c["policy_id"] == policy_id
            ),
            "true_severity": severity,
            "source_note": source,
        })
        content_id += 1

    # Second pass: duplicate with variation until we reach n
    while len(descriptions) < n:
        template, policy_id, severity, source = random.choice(CONTENT_TEMPLATES)
        descriptions.append({
            "content_id": f"CNT-{content_id:04d}",
            "description": add_variation(template),
            "true_policy_id": policy_id,
            "true_policy_name": next(
                c["policy_name"] for c in taxonomy["categories"]
                if c["policy_id"] == policy_id
            ),
            "true_severity": severity,
            "source_note": source,
        })
        content_id += 1

    random.shuffle(descriptions)
    return descriptions[:n]



def save_csv(data: list, filename: str):
    """Save list of dicts to CSV."""
    filepath = os.path.join(DATA_DIR, filename)
    if not data:
        print(f"Warning: No data to save for {filename}")
        return
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved {len(data)} rows to {filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("Policy Enforcement Auditor — Data Generation")
    print("=" * 60)

    # Generate content descriptions
    print("\n[1/2] Generating 500 content descriptions...")
    descriptions = generate_content_descriptions(500)
    save_csv(descriptions, "synthetic_content.csv")

    # Save ground truth separately (for evaluation)
    ground_truth = [
        {
            "content_id": d["content_id"],
            "true_policy_id": d["true_policy_id"],
            "true_policy_name": d["true_policy_name"],
            "true_severity": d["true_severity"],
        }
        for d in descriptions
    ]
    save_csv(ground_truth, "ground_truth.csv")

    # Summary stats
    print("\n[2/2] Summary Statistics:")
    print(f"  Content descriptions: {len(descriptions)}")
    print(f"  Unique policies covered: {len(set(d['true_policy_id'] for d in descriptions))}")
    print(f"  Severity distribution:")
    for sev in ["GREEN", "YELLOW", "RED"]:
        count = sum(1 for d in descriptions if d["true_severity"] == sev)
        print(f"    {sev}: {count} ({count/len(descriptions)*100:.1f}%)")
    print(f"\n  Source: YouTube Advertiser-Friendly Content Guidelines")
    print(f"  URL: https://support.google.com/youtube/answer/6162278")
    print("\n✅ Data generation complete.")
