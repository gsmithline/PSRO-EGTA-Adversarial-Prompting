from rainbow_teaming.descriptors import ATTACK_STYLES, RISK_CATEGORIES, build_descriptor, category_to_scalar


def test_category_to_scalar_known_unknown():
    assert category_to_scalar("violence", RISK_CATEGORIES) == 0.0
    assert category_to_scalar("unknown", RISK_CATEGORIES) == 1.0


def test_build_descriptor_bounds():
    desc = build_descriptor("violence", "direct")
    assert desc == (0.0, 0.0)
    desc_unknown = build_descriptor("mystery", "strange")
    assert desc_unknown == (1.0, 1.0)
    assert len(desc_unknown) == 2
