medical_data = [
    {"patient_id": 1, "symptoms": ["fever", "cough"], "diagnosis": None, "expert_feedback": None},
    {"patient_id": 2, "symptoms": ["headache", "fatigue"], "diagnosis": "Migraine", "expert_feedback": None},
    {"patient_id": 3, "symptoms": ["chest pain", "shortness of breath"], "diagnosis": "Possible Heart Condition", "expert_feedback": None},
]

# Simulate a discussion with a domain expert
def collaborate_with_expert(data):
    for record in data:

        # Case 1: No diagnosis yet
        if record["diagnosis"] is None:
            if "fever" in record["symptoms"] and "cough" in record["symptoms"]:
                record["diagnosis"] = "Flu"
                record["expert_feedback"] = "Consistent with seasonal flu."
            else:
                record["diagnosis"] = "Unknown"
                record["expert_feedback"] = "Further investigation needed."

        # Case 2: Diagnosis already exists
        else:
            if record["diagnosis"] == "Possible Heart Condition" and "chest pain" in record["symptoms"]:
                record["expert_feedback"] = "Likely heart condition, further tests required."
            else:
                record["expert_feedback"] = "Diagnosis confirmed."
    return data

updated_data = collaborate_with_expert(medical_data)

for record in updated_data:
    print(record)
