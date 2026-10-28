"""Story A (BTP Scenario 1), Gidea Park Platform 4, GP02 view: beat table and generation prompts.

Source: 2026-10-16_S1_StoryA_script_for_partner_review.docx, section "The story, beat by beat".
Each beat is 5 s of continuous footage from the one fixed camera (80 frames at 16 fps). `see` is the script's
"What the camera sees" text verbatim; `prompt` is the motion description given to the video model, which
also sees the previous beat's last frame.

Hard limits from the script (checked by eye on every selected take): adults only; nothing sexual; no kiss,
no groping, no hand on her chest or under her coat, no nudity; her response is a shove, not a strike; no
weapons, no injuries, no other people.
"""

CAMERA = (
    "Real CCTV security camera footage, fixed camera high above the platform looking steeply down and along an outdoor "
    "UK railway station platform (track on the left, grey asphalt walkway, dark slatted benches with purple armrests "
    "against a black metal railing on the right, yellow grit bin) in bright afternoon sun with hard shadows. "
    "The camera is completely static: no camera movement, no zoom, no cuts; the "

    "framing stays wide and fixed, showing the whole platform. "
)
WOMAN = ("the young woman in the red coat (early twenties, slim, dark hair in a low ponytail under a white knitted "
         "beanie, knee-length red coat, black jeans, white trainers, small black cross-body bag, phone in her hand)")
MAN = ("the man in the green jacket (in his fifties, heavy-set and broad-shouldered, short grey hair, bottle-green "
       "quilted jacket, tan chinos, brown shoes, hands empty)")
STYLE = (" Only these two people are on the platform. Realistic, natural human motion at real-time speed, "
         "realistic security camera image quality.")
EMPTY_STYLE = " Realistic, natural human motion at real-time speed, realistic security camera image quality."


def P(action, two=True):
    return CAMERA + action + (STYLE if two else EMPTY_STYLE)


# Pre-roll (not part of the 90 s): puts her on the middle seat of the near bench, starting from the empty plate.
PREROLL = {
    "P0a": P("A young woman in a red coat (early twenties, slim, dark hair in a low ponytail under a white knitted "
             "beanie, knee-length red coat, black jeans, white trainers, small black cross-body bag) walks into view "
             "from the bottom of the frame, walks to the near dark slatted bench on the right, turns and sits down on its "
             "middle seat facing the track, puts her bag on her lap and looks down at the phone in her hand.", two=False),
    "P0b": P("A young woman in a red coat (early twenties, slim, dark hair in a low ponytail under a white knitted "
             "beanie, knee-length red coat, black jeans, white trainers, small black cross-body bag) walks down the "
             "platform towards the camera, reaches the near dark slatted bench on the right, sits down on its middle seat "
             "facing the track, puts her bag on her lap and looks down at the phone in her hand.", two=False),
}

BEATS = [
    # id, t0, panel, see (verbatim), behaviour, tier, prompt
    dict(id="A01", t0=0, panel="1 Baseline", behaviour="waiting stationary", tier="T0",
         see="A young woman sits alone on the middle seat of the near bench, her bag on her lap, looking at her phone. "
             "Nobody else is on the platform.",
         prompt=P(f"{WOMAN.capitalize()} sits alone on the middle seat of the near dark slatted bench on the right, her bag on "
                  "her lap, looking down at her phone and scrolling with her thumb. She sits still. Nobody else is on "
                  "the platform; the rest of the platform is empty.", two=False)),
    dict(id="A02", t0=5, panel="1 Baseline", behaviour="waiting stationary", tier="T0",
         see="A man comes into view at the far end of the platform and walks down the walkway towards the camera at an "
             "ordinary pace. She does not look up.",
         prompt=P(f"{MAN.capitalize()} comes into view at the far end of the platform and walks down the grey walkway "
                  f"towards the camera at an ordinary walking pace. {WOMAN.capitalize()} stays seated on the middle seat "
                  "of the near bench, looking down at her phone; she does not look up.")),
    dict(id="A03", t0=10, panel="1 Baseline", behaviour="walking through", tier="T0",
         see="He slows as he reaches the gap between the two benches and pauses, facing the benches and glancing along "
             "them. The far bench is completely empty.",
         prompt=P(f"{MAN.capitalize()} slows down as he reaches the gap between the two benches, stops and stands still "
                  "facing the benches, turning his head to glance along the empty far bench and then at the near bench. "
                  f"The far bench is completely empty. {WOMAN.capitalize()} stays seated on the near bench looking at "
                  "her phone.")),
    dict(id="A04", t0=15, panel="2 Space encroachment", behaviour="personal space encroachment", tier="T1",
         see="He walks past the empty far bench and across to the near bench, stopping in front of its end seat, right "
             "next to her.",
         prompt=P(f"{WOMAN.capitalize()} stays seated on the bench the whole time, looking down at her phone; she does "
                  f"not stand up and does not look at him. {MAN.capitalize()} ignores the empty far bench and walks "
                  "slowly towards the camera along the front of her bench, passing in front of her knees, and stops "
                  "standing in front of the empty end seat of the bench nearest the camera, right next to her. "
                  "The framing stays wide and fixed, showing the whole platform.")),
    dict(id="A05", t0=20, panel="2 Space encroachment", behaviour="unwanted proximity seated", tier="T2",
         see="He sits down on the end seat beside her, a hand's width from her, although the rest of the bench and the "
             "whole far bench are empty. She keeps her eyes on her phone.",
         prompt=P(f"{MAN.capitalize()} turns and sits down on the empty end seat of the bench nearest the camera, "
                  f"right beside {WOMAN}, only a hand's width from her, although the rest of the bench and the whole "
                  "far bench are empty. He sits facing the track like her. She keeps her eyes on her phone and does "
                  "not look at him.")),
    dict(id="A06", t0=25, panel="3 Unsolicited engagement", behaviour="low level harassment gesture", tier="T2",
         see="He turns his head and speaks to her. She gives a brief nod without looking up and angles her body slightly "
             "away from him.",
         prompt=P(f"Both stay seated side by side on the bench. {MAN.capitalize()} turns his head and shoulders "
                  f"towards {WOMAN}, who sits beside him further along the bench, and talks to her, his mouth moving. "
                  "She keeps her head bowed and her eyes on the phone in her lap the whole time; she does not look "
                  "up and does not look at him. She gives one small nod, still looking down, and turns her shoulders "
                  "slightly away from him.")),
    dict(id="A07", t0=30, panel="3 Unsolicited engagement (she is avoidant)", behaviour="low level harassment gesture",
         tier="T2",
         see="He keeps talking, leaning towards her. She leans away, holds her phone closer and does not reply.",
         prompt=P(f"{MAN.capitalize()} keeps talking to {WOMAN}, leaning his upper body towards her. She leans away "
                  "from him, holds her phone closer to her face and does not reply.")),
    dict(id="A08", t0=35, panel="3 Unsolicited engagement", behaviour="low level harassment gesture", tier="T2",
         see="He gestures as he talks, his head now close to her shoulder. She stays turned away, still on her phone. "
             "They are not yet touching.",
         prompt=P(f"{MAN.capitalize()} gestures with one hand as he talks, his head now close to her shoulder. "
                  f"{WOMAN.capitalize()} stays turned away from him, still looking at her phone. They are not touching.")),
    dict(id="A09", t0=40, panel="4 Escalating animation", behaviour="low level harassment gesture", tier="T2",
         see="He becomes more animated: hands out to his sides, chest turned square to her. She pulls her bag round onto "
             "her lap as a barrier and shrinks away.",
         prompt=P(f"{MAN.capitalize()} becomes more animated as he talks, holding both hands out to his sides with his "
                  f"chest turned square to her. {WOMAN.capitalize()} pulls her black bag round onto her lap as a barrier "
                  "and shrinks away from him.")),
    dict(id="A10", t0=45, panel="4 Moves closer still", behaviour="unwanted proximity seated", tier="T2",
         see="He slides along the seat until his thigh is against hers and lays his arm along the backrest behind her, "
             "his hand on the rail, not yet on her. She leans further away.",
         prompt=P(f"{MAN.capitalize()} slides along the bench seat until his thigh is pressed against hers and lays his "
                  "arm along the top of the backrest behind her, his hand resting on the rail, not on her. "
                  f"{WOMAN.capitalize()} leans further away from him.")),
    dict(id="A11", t0=50, panel="5 Physical contact", behaviour="unwanted touching", tier="T3",
         see="His hand comes down onto the top of her far upper arm. She flinches and turns her face away from him, up "
             "the bench.",
         prompt=P(f"The arm of {MAN} that lies along the backrest comes down and his hand rests on the top of her far "
                  f"upper arm, around her shoulders. {WOMAN.capitalize()} flinches and turns her face away from him, "
                  "towards the far end of the bench.")),
    dict(id="A12", t0=55, panel="5 Physical contact", behaviour="unwanted touching", tier="T3",
         see="He puts his other hand on her knee and leaves it there. She lifts it off with her free hand, turns her "
             "knees away and clamps her bag to her chest.",
         prompt=P(f"With his arm still around her shoulders, {MAN} puts his other hand on her knee. "
                  f"{WOMAN.capitalize()} lifts his hand off her knee with her free hand, turns her knees away from him "
                  "and clamps her black bag to her chest with both arms.")),
    dict(id="A13", t0=60, panel="5 Physical contact (she remains avoidant)", behaviour="unwanted touching", tier="T3",
         see="He tightens the arm around her shoulders and pulls her against him. She goes rigid, her face turned hard "
             "away over her far shoulder.",
         prompt=P(f"{MAN.capitalize()} tightens his arm around her shoulders and pulls her sideways against him. "
                  f"{WOMAN.capitalize()} goes rigid, her bag clamped to her chest, her face turned hard away from him "
                  "over her far shoulder.")),
    dict(id="A14", t0=65, panel="6 Attempted assault", behaviour="grab or assault attempt", tier="T4",
         see="He grips her forearm, pulls it towards him and pushes his face towards hers. Her face stays turned away, "
             "head and shoulders pulled back from him.",
         prompt=P(f"{MAN.capitalize()} grips her forearm with his free hand, pulls her arm towards him and pushes his "
                  f"face towards her face. {WOMAN.capitalize()} keeps her face turned away and pulls her head and "
                  "shoulders back from him; their faces do not touch.")),
    dict(id="A15", t0=70, panel="6 Attempted assault (she pushes back)", behaviour="grab or assault attempt", tier="T4",
         see="She wrenches her arm free and shoves him away with both hands on his shoulder, arms straight. He is "
             "pushed back along the seat, his arm comes off her shoulders and a gap opens between them.",
         prompt=P(f"{WOMAN.capitalize()} wrenches her arm free and shoves {MAN} away with both hands on his shoulder, "
                  "arms straight. He is pushed back along the bench seat, his arm comes off her shoulders and a gap "
                  "opens between them.")),
    dict(id="A16", t0=75, panel="6 Attempted assault (aftermath begins)", behaviour="grab or assault attempt",
         tier="T4",
         see="She stands up fast, bag clutched to her chest, steps out into the walkway and turns to face him. He stays "
             "on the bench, rocked back, and reaches a hand out towards her, well short of her.",
         prompt=P(f"{WOMAN.capitalize()} stands up quickly from the bench, her bag clutched to her chest, steps out "
                  f"onto the grey walkway and turns to face him. {MAN.capitalize()} stays seated on the bench, rocked "
                  "back, and reaches one hand out towards her, well short of her.")),
    dict(id="A17", t0=80, panel="6 Aftermath", behaviour="confrontational posture", tier="T3",
         see="Still facing him, she backs away up the platform two or three steps, holds a palm out towards him "
             "(“stay back”) and lifts her phone to her ear. He stands up from the bench, arms spread wide, "
             "remonstrating, without stepping towards her.",
         prompt=P(f"Still facing him, {WOMAN} backs away up the platform, away from the camera, two or three steps, "
                  "holds one palm out towards him to tell him to stay back, and lifts her phone to her ear. "
                  f"{MAN.capitalize()} stands up from the bench with his arms spread wide, remonstrating, without "
                  "stepping towards her.")),
    dict(id="A18", t0=85, panel="6 Aftermath", behaviour="confrontational posture", tier="T3",
         see="She takes one more step back and stands still, phone to her ear, palm out. He stays by the bench and "
             "points at her. The clip ends on this tableau, the two of them well apart.",
         prompt=P(f"{WOMAN.capitalize()} takes one more step back and stands still on the walkway, phone to her ear, "
                  f"her other palm held out towards him. {MAN.capitalize()} stays standing by the bench and points at "
                  "her. The two of them stay well apart.")),
]

# Benign twin: same A01-A03, then he takes the empty far bench.
TWIN = [
    dict(id="AT1", t0=15, panel="Benign twin", behaviour="waiting stationary", tier="T0",
         see="He steps up the platform to the empty far bench and sits on it, about three metres from her, facing the "
             "track.",
         prompt=P(f"{MAN.capitalize()} steps up the platform to the empty far bench and sits down on it, about three "
                  f"metres from {WOMAN}, facing the track. She keeps looking at her phone on the near bench.")),
    dict(id="AT2", t0=20, panel="Benign twin", behaviour="walking through", tier="T0",
         see="A little later she stands, grips her bag strap and steps out into the walkway, unhurried, turning towards "
             "the camera end. He is looking along the track, not at her.",
         prompt=P(f"{WOMAN.capitalize()} stands up from the near bench, grips her bag strap and steps out onto the "
                  f"walkway, unhurried, turning towards the camera. {MAN.capitalize()} stays seated on the far bench "
                  "looking along the track, not at her.")),
    dict(id="AT3", t0=25, panel="Benign twin", behaviour="walking through", tier="T0",
         see="She walks down the platform at an ordinary pace past the near bench and stops beside the grit bin to wait, "
             "facing the track. Nobody follows her.",
         prompt=P(f"{WOMAN.capitalize()} walks down the platform towards the camera at an ordinary pace, past the near "
                  "bench, and stops beside the yellow grit bin to wait, facing the track. "
                  f"{MAN.capitalize()} stays seated on the far bench.")),
    dict(id="AT4", t0=30, panel="Benign twin", behaviour="walking through", tier="T0",
         see="He gets up and walks the same way with his eyes ahead, stops three to four metres short of her and turns to "
             "face the track: two passengers waiting for a train.",
         prompt=P(f"{MAN.capitalize()} gets up from the far bench and walks down the platform towards the camera with his "
                  f"eyes ahead, stops three to four metres short of {WOMAN} and turns to face the track. Two passengers "
                  "waiting for a train.")),
]

ALL = {b["id"]: b for b in BEATS + TWIN}

TIER_COLOURS = {"T0": "#2e7d32", "T1": "#9e9d24", "T2": "#ef8f00", "T3": "#e65100", "T4": "#c62828"}
PANELS = [
    ("1 Baseline", 0, 15), ("2 Space encroachment", 15, 25), ("3 Unsolicited engagement", 25, 40),
    ("4 Escalating animation, moves closer", 40, 50), ("5 Physical contact", 50, 65),
    ("6 Attempted assault and aftermath", 65, 90),
]


# ---- v2 prompts (2026-10-22): explicit posture, gaze and side for each person; hard limits restated ----
_W = "the young woman in the red coat"
_M = "the man in the green jacket"
_V2 = {
    "A07": (f"Both stay seated side by side on the bench. {_M.capitalize()} keeps talking to her, leaning his upper "
            f"body towards her. {_W.capitalize()}, who sits further along the bench, leans her upper body away from "
            "him towards the far end of the bench, holding her phone with both hands in front of her chest, head "
            "bowed over it; she does not reply and does not look at him. Her hands stay low, at chest height."),
    "A08": (f"Both stay seated. {_M.capitalize()} gestures with one hand as he talks, leaning in so that his head is "
            f"close to her shoulder. {_W.capitalize()} stays turned away from him, looking down at her phone. They "
            "are not touching."),
    "A09": (f"Both stay seated. {_M.capitalize()} becomes more animated as he talks: he turns his chest square "
            f"towards her and spreads both hands out to his sides, palms up. {_W.capitalize()} pulls her black bag "
            "onto her lap in front of her as a barrier and shrinks away from him towards the far end of the bench, "
            "looking down."),
    "A10": (f"Both stay seated. {_M.capitalize()} slides along the bench seat towards her until his thigh presses "
            "against her thigh, and lays his arm along the top of the slatted backrest behind her shoulders, his hand "
            f"resting on the backrest, not touching her. {_W.capitalize()} leans further away from him, looking "
            "down at her phone."),
    "A11": (f"Both stay seated. The arm of {_M} that lies along the backrest comes down and his hand rests on the "
            f"top of her far upper arm, so his arm is around her shoulders. {_W.capitalize()} flinches and turns "
            "her face away from him, towards the far end of the bench."),
    "A12": (f"{_W.capitalize()} remains seated on the bench for the whole clip; she does not stand up. "
            f"{_M.capitalize()} keeps his arm around her shoulders, reaches across with his other hand and places it "
            "on her knee, and leaves it resting there. She pushes his hand off her knee with her hand, turns her "
            "knees away from him while staying seated, and clamps her black bag to her chest."),
    "A13": (f"Both people stay sitting on the bench for the entire clip; nobody stands up and nobody walks. "
            f"{_M.capitalize()}, sitting at the near end of the bench, wraps his arm around the shoulders of {_W} and "
            "pulls her close against his side. She sits stiffly upright, clutching her black bag to her chest, and "
            "turns her face away from him towards the far end of the platform."),
    "A14": (f"Both people stay sitting on the bench for the entire clip; nobody stands up. {_M.capitalize()} grabs "
            f"the forearm of {_W} with his hand and yanks her arm towards him, lunging his head towards her. She "
            "resists: she turns her head sharply away from him towards the far end of the platform and leans her "
            "head and shoulders back, away from him, straining to pull her arm free, her face tense and frightened. "
            "A clear gap stays between their heads. Her white beanie stays on her head; his short grey hair stays "
            "uncovered."),
    "A15": (f"Both people are sitting on the bench. {_W.capitalize()} wrenches her arm free from his grip and shoves "
            f"{_M} away with both hands on his shoulder, her arms straight. He is pushed back along the bench seat "
            "towards the near end, his arm comes off her shoulders and a gap opens between them. Both stay seated; "
            "nobody stands up yet."),
    "A16": (f"{_W.capitalize()} stands up quickly from the bench, clutching her black bag to her chest, steps out "
            f"onto the grey walkway in front of the bench and turns to face him. {_M.capitalize()} stays seated on "
            "the bench, leaning back, and reaches one hand out towards her, but she is out of his reach."),
    "A17": (f"Wide fixed shot; both people stay small in the frame and the camera does not follow them. "
            f"{_W.capitalize()}, standing on the grey walkway, faces the man and walks slowly backwards along the middle "
            "of the walkway towards the far end of the platform, away from the camera, keeping well away from the "
            "platform edge. She holds one palm out towards him to tell him to stay back and lifts her phone to her "
            f"ear. {_M.capitalize()} stands up from the bench, spreads his arms wide and argues, but stays beside the "
            "bench and does not step towards her."),
    "A18": (f"Wide fixed shot; the camera does not move and does not follow anyone. {_W.capitalize()}, standing on "
            "the grey walkway facing the man, lifts her phone to her ear, holds her other palm out towards him, takes "
            f"one more small step back and then stands still. {_M.capitalize()} stays standing beside the bench, "
            "raises his arm and points at her. The two of them stay well apart."),
}
for _b in BEATS:
    if _b["id"] in _V2:
        _b["prompt"] = P(_V2[_b["id"]])

_V2_TWIN = {
    "AT1": (f"{_W.capitalize()} stays seated on the near bench the whole time, looking down at her phone. "
            f"{_M.capitalize()} turns away from her, walks up the platform away from the camera to the empty far "
            "bench and sits down on it, about three metres from her, facing the track."),
    "AT2": (f"{_M.capitalize()} stays seated on the far bench, looking along the track, not at her. "
            f"{_W.capitalize()} stands up from the near bench, unhurried, grips her bag strap, steps out onto the grey "
            "walkway and turns towards the camera."),
    "AT3": (f"{_W.capitalize()} walks down the platform towards the camera at an ordinary pace, past the near bench, "
            f"and stops beside the yellow grit bin to wait, facing the track. {_M.capitalize()} stays seated on the far "
            "bench. Nobody follows her."),
    "AT4": (f"{_M.capitalize()} gets up from the far bench and walks down the platform towards the camera with his "
            f"eyes ahead, stops three to four metres short of {_W} and turns to face the track. Two passengers "
            "waiting for a train, well apart."),
}
for _b in TWIN:
    if _b["id"] in _V2_TWIN:
        _b["prompt"] = P(_V2_TWIN[_b["id"]])


# ---- v3 (2026-10-22): A17 and A18 pinned at both ends (first + last frame), matching the script's timing:
# she lifts the phone to her ear during A17; in A18 she takes one more step back and stands still.
# End frames: work/keyframes/v2/A17_end.png, work/keyframes/v2/A18_end.png (scripts/make_keyframe.py).
_V3 = {
    "A17": (f"Still facing him, {_W} backs away up the grey walkway two or three steps, away from him, holds one palm "
            "out towards him to tell him to stay back, and lifts her phone to her ear with her other hand. "
            f"{_M.capitalize()} stands up from the bench with his arms spread wide, remonstrating, but stays beside the "
            "bench and does not step towards her. The camera does not move."),
    "A18": (f"{_W.capitalize()} takes one more step back up the walkway and stands still, her phone to her ear, her "
            f"other palm held out towards him. {_M.capitalize()} stays standing beside the bench, lowers his arms, then "
            "raises one arm and points at her. The two of them stay well apart. The camera does not move."),
}
for _b in BEATS:
    if _b["id"] in _V3:
        _b["prompt"] = P(_V3[_b["id"]])
