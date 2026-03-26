import re

CHAMPION_PROFILE = {
    "aatrox": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "ahri": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},
    "akali": {"subclass": "assassin", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "akshan": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "late"},
    "alistar": {"subclass": "engage", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "ambessa": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "amumu": {"subclass": "tank", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "anivia": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},
    "annie": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},
    "aphelios": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "late"},
    "ashe": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "early"},
    "aurelionsol": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},
    "aurora": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},
    "azir": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},

    "bard": {"subclass": "enchanter", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},
    "belveth": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "early"},
    "blitzcrank": {"subclass": "engage", "damage_type": "ap", "range_type": "melee", "scaling_type": "early"},
    "brand": {"subclass": "battlemage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},
    "braum": {"subclass": "engage", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "briar": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},

    "caitlyn": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "early"},
    "camille": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "cassiopeia": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},
    "chogath": {"subclass": "tank", "damage_type": "ap", "range_type": "melee", "scaling_type": "late"},
    "corki": {"subclass": "marksman", "damage_type": "mixed", "range_type": "ranged", "scaling_type": "mid"},

    "darius": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "early"},
    "diana": {"subclass": "assassin", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "drmundo": {"subclass": "tank", "damage_type": "ap", "range_type": "melee", "scaling_type": "late"},
    "draven": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "early"},

    "ekko": {"subclass": "assassin", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "elise": {"subclass": "assassin", "damage_type": "ap", "range_type": "mixed", "scaling_type": "early"},
    "evelynn": {"subclass": "assassin", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "ezreal": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "mid"},

    "fiddlesticks": {"subclass": "assassin", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},
    "fiora": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "late"},
    "fizz": {"subclass": "assassin", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},

    "galio": {"subclass": "mage", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "gangplank": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "late"},
    "garen": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "gnar": {"subclass": "bruiser", "damage_type": "ad", "range_type": "mixed", "scaling_type": "mid"},
    "gragas": {"subclass": "tank", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "graves": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "mid"},
    "gwen": {"subclass": "bruiser", "damage_type": "ap", "range_type": "melee", "scaling_type": "late"},

    "hecarim": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "heimerdinger": {"subclass": "battlemage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},
    "hwei": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},

    "illaoi": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "irelia": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "ivern": {"subclass": "enchanter", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},

    "janna": {"subclass": "enchanter", "damage_type": "ap", "range_type": "ranged", "scaling_type": "early"},
    "jarvaniv": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "early"},
    "jax": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "late"},
    "jayce": {"subclass": "assassin", "damage_type": "ad", "range_type": "mixed", "scaling_type": "early"},
    "jhin": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "mid"},
    "jinx": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "late"},

    "kaisa": {"subclass": "marksman", "damage_type": "mixed", "range_type": "ranged", "scaling_type": "late"},
    "kalista": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "early"},
    "karma": {"subclass": "enchanter", "damage_type": "ap", "range_type": "ranged", "scaling_type": "early"},
    "karthus": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},
    "kassadin": {"subclass": "assassin", "damage_type": "ap", "range_type": "melee", "scaling_type": "late"},
    "katarina": {"subclass": "assassin", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "kayle": {"subclass": "battlemage", "damage_type": "mixed", "range_type": "mixed", "scaling_type": "late"},
    "kayn": {"subclass": "assassin", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "kennen": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},
    "khazix": {"subclass": "assassin", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "kindred": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "late"},
    "kled": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "early"},
    "kogmaw": {"subclass": "marksman", "damage_type": "mixed", "range_type": "ranged", "scaling_type": "mid"},
    "ksante": {"subclass": "tank", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},

    "leblanc": {"subclass": "assassin", "damage_type": "ap", "range_type": "ranged", "scaling_type": "early"},
    "leesin": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "early"},
    "leona": {"subclass": "engage", "damage_type": "ap", "range_type": "melee", "scaling_type": "early"},
    "lillia": {"subclass": "mage", "damage_type": "ap", "range_type": "melee", "scaling_type": "late"},
    "lissandra": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},
    "lucian": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "early"},
    "lulu": {"subclass": "enchanter", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},
    "lux": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},

    "malphite": {"subclass": "tank", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "malzahar": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},
    "maokai": {"subclass": "tank", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "masteryi": {"subclass": "assassin", "damage_type": "ad", "range_type": "melee", "scaling_type": "late"},
    "mel": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},
    "milio": {"subclass": "enchanter", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},
    "missfortune": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "mid"},
    "mordekaiser": {"subclass": "bruiser", "damage_type": "ap", "range_type": "melee", "scaling_type": "late"},
    "morgana": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},

    "naafiri": {"subclass": "assassin", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "nami": {"subclass": "enchanter", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},
    "nasus": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "late"},
    "nautilus": {"subclass": "engage", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "neeko": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "early"},
    "nidalee": {"subclass": "battlemage", "damage_type": "ap", "range_type": "mixed", "scaling_type": "early"},
    "nilah": {"subclass": "marksman", "damage_type": "ad", "range_type": "melee", "scaling_type": "late"},
    "nocturne": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "nunu": {"subclass": "tank", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},

    "olaf": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "early"},
    "orianna": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},
    "ornn": {"subclass": "tank", "damage_type": "ap", "range_type": "melee", "scaling_type": "late"},

    "pantheon": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "early"},
    "poppy": {"subclass": "engage", "damage_type": "ad", "range_type": "melee", "scaling_type": "early"},
    "pyke": {"subclass": "engage", "damage_type": "ad", "range_type": "melee", "scaling_type": "early"},

    "qiyana": {"subclass": "assassin", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "quinn": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "early"},

    "rakan": {"subclass": "engage", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "rammus": {"subclass": "tank", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "reksai": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "early"},
    "rell": {"subclass": "engage", "damage_type": "ap", "range_type": "melee", "scaling_type": "early"},
    "renataglasc": {"subclass": "enchanter", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},
    "renekton": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "early"},
    "rengar": {"subclass": "assassin", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "riven": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "rumble": {"subclass": "mage", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "ryze": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},

    "samira": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "mid"},
    "sejuani": {"subclass": "tank", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "senna": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "late"},
    "seraphine": {"subclass": "enchanter", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},
    "sett": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "shaco": {"subclass": "assassin", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "shen": {"subclass": "tank", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "shyvana": {"subclass": "bruiser", "damage_type": "mixed", "range_type": "melee", "scaling_type": "late"},
    "singed": {"subclass": "tank", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "sion": {"subclass": "tank", "damage_type": "ad", "range_type": "melee", "scaling_type": "late"},
    "sivir": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "late"},
    "skarner": {"subclass": "tank", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "smolder": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "late"},
    "sona": {"subclass": "enchanter", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},
    "soraka": {"subclass": "enchanter", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},
    "swain": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},
    "sylas": {"subclass": "bruiser", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "syndra": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},

    "tahmkench": {"subclass": "tank", "damage_type": "ap", "range_type": "melee", "scaling_type": "mid"},
    "taliyah": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},
    "talon": {"subclass": "assassin", "damage_type": "ad", "range_type": "melee", "scaling_type": "early"},
    "taric": {"subclass": "enchanter", "damage_type": "ap", "range_type": "melee", "scaling_type": "late"},
    "teemo": {"subclass": "battlemage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},
    "thresh": {"subclass": "engage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},
    "tristana": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "late"},
    "trundle": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "tryndamere": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "early"},
    "twistedfate": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},
    "twitch": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "late"},

    "udyr": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "urgot": {"subclass": "bruiser", "damage_type": "ad", "range_type": "ranged", "scaling_type": "mid"},

    "varus": {"subclass": "marksman", "damage_type": "mixed", "range_type": "ranged", "scaling_type": "mid"},
    "vayne": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "late"},
    "veigar": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},
    "velkoz": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},
    "vex": {"subclass": "battlemage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},
    "vi": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "viego": {"subclass": "assassin", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "viktor": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},
    "vladimir": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},
    "volibear": {"subclass": "bruiser", "damage_type": "mixed", "range_type": "melee", "scaling_type": "early"},

    "warwick": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "early"},
    "monkeyking": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},

    "xayah": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "late"},
    "xerath": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},
    "xinzhao": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "early"},

    "yasuo": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "early"},
    "yone": {"subclass": "assassin", "damage_type": "ad", "range_type": "melee", "scaling_type": "late"},
    "yorick": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "yunara": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "late"},
    "yuumi": {"subclass": "enchanter", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},

    "zaahen": {"subclass": "bruiser", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "zac": {"subclass": "tank", "damage_type": "ap", "range_type": "melee", "scaling_type": "early"},
    "zed": {"subclass": "assassin", "damage_type": "ad", "range_type": "melee", "scaling_type": "mid"},
    "zeri": {"subclass": "marksman", "damage_type": "ad", "range_type": "ranged", "scaling_type": "late"},
    "ziggs": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},
    "zilean": {"subclass": "enchanter", "damage_type": "ap", "range_type": "ranged", "scaling_type": "late"},
    "zoe": {"subclass": "mage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "early"},
    "zyra": {"subclass": "battlemage", "damage_type": "ap", "range_type": "ranged", "scaling_type": "mid"},
}

def get_champion_profile(champion_name: str) -> dict[str, str]:
    """
    get the dict
    """
    key = normalize_champion_name(champion_name)
    if key not in CHAMPION_PROFILE:
        raise KeyError(
            f"Champion '{champion_name}' normalized to '{key}' is missing from CHAMPION_PROFILE."
        )
    return CHAMPION_PROFILE[key]

def get_champion_subclass(champion_name: str) -> str:
    return get_champion_profile(champion_name)["subclass"]

def get_champion_damage_type(champion_name: str) -> str:
    return get_champion_profile(champion_name)["damage_type"]

def get_champion_range_type(champion_name: str) -> str:
    return get_champion_profile(champion_name)["range_type"]

def get_champion_scaling_type(champion_name: str) -> str:
    return get_champion_profile(champion_name)["scaling_type"]

def normalize_champion_name(name: str) -> str:
    """
    Normalize champion names so small punctuation/spacing differences still map.
    Examples:
    - "Kai'Sa" -> "kaisa"
    - "Nunu & Willump" -> "nunuwillump" (handled below)
    """
    s = str(name).strip().lower()

    # common manual aliases before stripping
    alias_map = {
        "wukong": "monkeyking",
        "nunu & willump": "nunu",
        "nunu and willump": "nunu",
        "nunuwillump": "nunu",
        "dr. mundo": "drmundo",
        "dr mundo": "drmundo",
        "cho'gath": "chogath",
        "kai'sa": "kaisa",
        "kha'zix": "khazix",
        "kog'maw": "kogmaw",
        "rek'sai": "reksai",
        "vel'koz": "velkoz",
        "bel'veth": "belveth",
        "aurelion sol": "aurelionsol",
        "tahm kench": "tahmkench",
        "twisted fate": "twistedfate",
        "xin zhao": "xinzhao",
        "jarvan iv": "jarvaniv",
        "renata glasc": "renataglasc",
        "renata": "renataglasc",
        "master yi": "masteryi",
        "k'sante": "ksante",
    }

    if s in alias_map:
        return alias_map[s]

    return re.sub(r"[^a-z0-9]", "", s)