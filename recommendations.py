def get_recommendations(severity):

    if severity == "MILD":
        return {
            "Allopathy": ["Short-acting bronchodilators"],
            "Lifestyle": [
                "Quit smoking",
                "Light exercise",
                "Avoid pollution"
            ],
            "Homeopathy": ["Consult certified practitioner"]
        }

    elif severity == "MODERATE":
        return {
            "Allopathy": [
                "Long-acting bronchodilators",
                "Inhaled corticosteroids"
            ],
            "Lifestyle": [
                "Pulmonary rehab",
                "Breathing exercises"
            ],
            "Homeopathy": ["Complementary only"]
        }

    else:
        return {
            "Allopathy": [
                "Combination inhalers",
                "Oxygen therapy"
            ],
            "Lifestyle": [
                "Strict monitoring",
                "Avoid exertion"
            ],
            "Homeopathy": ["Supportive only"]
        }