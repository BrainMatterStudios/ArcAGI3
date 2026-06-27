from arcagi3 import invisible_state as IS


def _t(mkey, ukey, action, next_mkey, feats):
    return IS.Transition(mkey=mkey, ukey=ukey, action=action, next_mkey=next_mkey, feats=feats)


def test_find_violations():
    ts = [
        _t(b"A", b"A", ("S", 1), b"X", {}),
        _t(b"A", b"A", ("S", 1), b"Y", {}),   # same (mkey,action) -> different next => violation
        _t(b"A", b"A", ("S", 2), b"Z", {}),    # different action, single next => not a violation
    ]
    v = IS.find_violations(ts)
    assert set(v) == {(b"A", ("S", 1))}
    assert v[(b"A", ("S", 1))] == {b"X", b"Y"}


def test_invisible_state_resolved_by_history():
    # same pixels (ukey constant), next depends on a hidden counter exposed via feats["rot_mod4"]
    ts = [
        _t(b"G", b"G", ("S", 1), b"WIN", {"rot_mod4": 0, "noise_mod2": 0}),
        _t(b"G", b"G", ("S", 1), b"BLOCK", {"rot_mod4": 1, "noise_mod2": 1}),
        _t(b"G", b"G", ("S", 1), b"WIN", {"rot_mod4": 0, "noise_mod2": 1}),
        _t(b"G", b"G", ("S", 1), b"BLOCK", {"rot_mod4": 1, "noise_mod2": 0}),
    ]
    report = IS.classify(ts)
    pair = (b"G", ("S", 1))
    assert report[pair]["category"] == "INVISIBLE_STATE"
    assert report[pair]["resolver"] == "rot_mod4"      # noise_mod2 does NOT separate win/block


def test_over_merge_detected():
    # masking merged two visibly-different from-states (distinct ukey)
    ts = [
        _t(b"M", b"U1", ("S", 1), b"X", {"f_mod2": 0}),
        _t(b"M", b"U2", ("S", 1), b"Y", {"f_mod2": 1}),
    ]
    report = IS.classify(ts)
    assert report[(b"M", ("S", 1))]["category"] == "OVER_MERGE"


def test_unexplained_when_no_feature_resolves():
    ts = [
        _t(b"N", b"N", ("S", 1), b"X", {"f_mod2": 0}),
        _t(b"N", b"N", ("S", 1), b"Y", {"f_mod2": 0}),   # identical feats, different next => unresolved
    ]
    report = IS.classify(ts)
    assert report[(b"N", ("S", 1))]["category"] == "UNEXPLAINED"


def test_summary_rates():
    ts = [
        _t(b"G", b"G", ("S", 1), b"WIN", {"rot_mod4": 0}),
        _t(b"G", b"G", ("S", 1), b"BLOCK", {"rot_mod4": 1}),
        _t(b"P", b"P", ("S", 1), b"Q", {"rot_mod4": 0}),   # non-violation
    ]
    s = IS.summary(ts)
    assert s["n_pairs"] == 2 and s["n_violations"] == 1
    assert s["invisible_state"] == 1 and s["over_merge"] == 0 and s["unexplained"] == 0
