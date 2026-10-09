"""
Stage 3 & 7: Understand & clean parcellation, assign structure importance.
Maps SynthSeg / FreeSurfer label IDs to anatomical names and risk weights.
"""

SYNTHSEG_LABELS = {
    0: ("Background", "Background", 0.0),
    2: ("Left-Cerebral-White-Matter", "General", 0.3),
    3: ("Left-Cerebral-Cortex", "Important", 0.7),
    4: ("Left-Lateral-Ventricle", "Critical", 1.0),
    5: ("Left-Inf-Lat-Vent", "Critical", 1.0),
    7: ("Left-Cerebellum-White-Matter", "General", 0.3),
    8: ("Left-Cerebellum-Cortex", "General", 0.3),
    10: ("Left-Thalamus", "Important", 0.7),
    11: ("Left-Caudate", "Important", 0.7),
    12: ("Left-Putamen", "Important", 0.7),
    13: ("Left-Pallidum", "Important", 0.7),
    14: ("3rd-Ventricle", "Critical", 1.0),
    15: ("4th-Ventricle", "Critical", 1.0),
    16: ("Brain-Stem", "Critical", 1.0),
    17: ("Left-Hippocampus", "Important", 0.7),
    18: ("Left-Amygdala", "Important", 0.7),
    24: ("CSF", "Background", 0.0),
    26: ("Left-Accumbens-area", "Important", 0.7),
    28: ("Left-VentralDC", "General", 0.3),
    41: ("Right-Cerebral-White-Matter", "General", 0.3),
    42: ("Right-Cerebral-Cortex", "Important", 0.7),
    43: ("Right-Lateral-Ventricle", "Critical", 1.0),
    44: ("Right-Inf-Lat-Vent", "Critical", 1.0),
    46: ("Right-Cerebellum-White-Matter", "General", 0.3),
    47: ("Right-Cerebellum-Cortex", "General", 0.3),
    49: ("Right-Thalamus", "Important", 0.7),
    50: ("Right-Caudate", "Important", 0.7),
    51: ("Right-Putamen", "Important", 0.7),
    52: ("Right-Pallidum", "Important", 0.7),
    53: ("Right-Hippocampus", "Important", 0.7),
    54: ("Right-Amygdala", "Important", 0.7),
    58: ("Right-Accumbens-area", "Important", 0.7),
    60: ("Right-VentralDC", "General", 0.3),
}

def get_label_info(label_id):
    """
    Returns (name, tier, importance_weight) for a given label ID.
    If cortical parcellation labels (1000-2035) are present, defaults to Important tier.
    """
    label_id = int(label_id)
    if label_id in SYNTHSEG_LABELS:
        return SYNTHSEG_LABELS[label_id]
    elif 1000 <= label_id < 3000:
        return (f"Cortical-Parcel-{label_id}", "Important", 0.7)
    else:
        return (f"Unknown-{label_id}", "General", 0.3)
