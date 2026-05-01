def classify_attention(is_eyes_closed, is_looking_away, is_yawning=False):
    """
    Classifies the user's attention state based on eye state, head pose, and yawning.
    Returns the attention status and a corresponding score (0-100%).
    """
    if is_eyes_closed or is_yawning:
        return "Drowsy", 20
    elif is_looking_away:
        return "Distracted", 40
    else:
        return "Focused", 100
