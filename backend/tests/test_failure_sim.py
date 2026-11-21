from app.failure_sim import get_simulated_status, parse_simulate_failure_header


def test_parse_single_pair() -> None:
    assert parse_simulate_failure_header("key_1=429") == {"key_1": 429}


def test_parse_multiple_pairs_with_spaces() -> None:
    assert parse_simulate_failure_header(" key_1 = 429 , key_2 = 500 ") == {
        "key_1": 429,
        "key_2": 500,
    }


def test_parse_wildcard() -> None:
    rules = parse_simulate_failure_header("*=503")
    assert rules == {"*": 503}
    assert get_simulated_status(rules, "key_1") == 503
    assert get_simulated_status(rules, "key_2") == 503


def test_exact_key_takes_precedence_over_wildcard() -> None:
    rules = parse_simulate_failure_header("*=500,key_2=429")
    assert get_simulated_status(rules, "key_1") == 500
    assert get_simulated_status(rules, "key_2") == 429


def test_unsupported_statuses_and_garbage_ignored() -> None:
    assert parse_simulate_failure_header("key_1=200,key_2=404,broken,key_3=abc,key_4=502") == {
        "key_4": 502
    }
    assert parse_simulate_failure_header(None) == {}
    assert parse_simulate_failure_header("") == {}
